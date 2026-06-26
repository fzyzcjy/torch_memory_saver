#include "hardware_xpu_support.h"
#include "core.h"

#if defined(USE_XPU)

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

// Intel XPU backend for torch_memory_saver using Level Zero Virtual Memory
// Management (VMM). Implements the same pause/resume semantics as CUDA/ROCm
// backends, but using Intel's Level Zero API instead.
//
// Architecture:
//   - malloc:  zeVirtualMemReserve → zePhysicalMemCreate → zeVirtualMemMap
//   - pause:   zeVirtualMemUnmap + zePhysicalMemDestroy (VA stays reserved)
//   - resume:  zePhysicalMemCreate + zeVirtualMemMap (to same VA)
//   - free:    unmap + destroy (if ACTIVE), then zeVirtualMemFree
//
// Functions return cudaError_t (typedef'd to int for XPU); ze_result_t is
// translated to those codes at the boundary. See macro.h for the rationale.

namespace {

#ifdef TMS_DEBUG_LOG
#define XPU_LOG(x)                                                             \
  do {                                                                         \
    std::cout << "[torch_memory_saver:xpu] " << x << std::endl;                \
  } while (0)
#else
#define XPU_LOG(x)                                                             \
  do {                                                                         \
  } while (0)
#endif

#define XPU_ERR(x)                                                             \
  do {                                                                         \
    std::cerr << "[torch_memory_saver:xpu] " << x << std::endl;                \
  } while (0)

size_t align_up(size_t size, size_t align) {
  return (size + align - 1) & ~(align - 1);
}

ze_context_handle_t ze_context_of(const sycl::context &ctx) {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
}

ze_device_handle_t ze_device_of(const sycl::device &dev) {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
}

size_t ze_alloc_granularity(ze_context_handle_t ze_ctx,
                            ze_device_handle_t ze_dev) {
  size_t page_size = 0;
  ze_result_t rc = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, 1, &page_size);
  if (rc != ZE_RESULT_SUCCESS || page_size == 0)
    page_size = 2 * 1024 * 1024; // fallback 2 MiB
  return page_size;
}

// --------------------------------------------------- per-device SYCL context
// CRITICAL: use the SYCL platform default context (the one sycl::queue(device)
// picks up), NOT a fresh sycl::context(device). Level Zero virtual address
// mappings are per-ze_context: a VA mapped in one context is invisible in
// another. PyTorch's XPU streams use the platform default context, so our VMM
// allocations must live there too or torch kernels won't see them.
//
// Contexts are intentionally leaked (raw new, never deleted). sycl/L0 static
// destructors run in an undefined order at process exit; destroying a
// sycl::context after the runtime has begun teardown yields
// "UR_RESULT_ERROR_DEVICE_LOST". Leaking avoids the destructor; the OS reclaims
// memory at exit anyway.
struct PerDeviceContext {
  sycl::device sycl_dev;
  sycl::context sycl_ctx;
  sycl::queue sycl_queue; // reused; creating queues inside the alloc lock can
                          // re-enter the allocator and deadlock
  ze_context_handle_t ze_ctx;
  ze_device_handle_t ze_dev;
};

// On runtimes that expose the same physical GPUs through BOTH a Level-Zero and
// an OpenCL SYCL platform, get_devices(gpu) returns a mixed list (e.g. 8 entries
// for 4 GPUs). torch.xpu enumerates Level-Zero devices only, so we must filter
// to the Level-Zero backend before indexing by a torch device ordinal;
// otherwise an ordinal can land on an OpenCL device and get_native<level_zero>
// fails ("Backends mismatch"). On old single-platform runtimes every GPU is
// already Level-Zero, so this is an order-preserving no-op.
const std::vector<sycl::device> &l0_gpu_devices() {
  // Cache once populated. Do NOT memoize an empty result: a call that races
  // ahead of SYCL/L0 device enumeration must be able to recover on a later
  // call. All callers reach here under the get_device_context mutex, so the
  // non-atomic pointer/contents are race-free in practice.
  static std::vector<sycl::device> *cache = nullptr;
  if (cache && !cache->empty())
    return *cache;
  if (!cache)
    cache = new std::vector<sycl::device>(); // leaked; matches ctx_map pattern
  cache->clear();
  for (const auto &d :
       sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (d.get_backend() == sycl::backend::ext_oneapi_level_zero)
      cache->push_back(d);
  }
  XPU_LOG("l0_gpu_devices: " << cache->size() << " Level-Zero GPU device(s)");
  return *cache;
}

PerDeviceContext &get_device_context(int device_id) {
  static auto *ctx_map = new std::unordered_map<int, PerDeviceContext *>();
  static auto *ctx_mutex = new std::mutex();

  std::lock_guard<std::mutex> lock(*ctx_mutex);
  auto it = ctx_map->find(device_id);
  if (it != ctx_map->end())
    return *(it->second);

  const std::vector<sycl::device> &devices = l0_gpu_devices();
  if (device_id < 0 || device_id >= (int)devices.size())
    throw std::runtime_error("[torch_memory_saver:xpu] invalid XPU device id: " +
                             std::to_string(device_id));

  auto *pdc = new PerDeviceContext();
  pdc->sycl_dev = devices[device_id];
  pdc->sycl_queue = sycl::queue(pdc->sycl_dev); // default platform context
  pdc->sycl_ctx = pdc->sycl_queue.get_context();
  pdc->ze_ctx = ze_context_of(pdc->sycl_ctx);
  pdc->ze_dev = ze_device_of(pdc->sycl_dev);
  ctx_map->emplace(device_id, pdc);
  return *pdc;
}

} // namespace

namespace XPUImplementation {

// ------------------------------------------------------------------ malloc
cudaError_t xpu_malloc(
    void **ptr,
    CUdevice device,
    size_t size,
    const std::string &tag,
    bool enable_cpu_backup,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  try {
    PerDeviceContext &pdc = get_device_context(device);
    size_t granularity = ze_alloc_granularity(pdc.ze_ctx, pdc.ze_dev);
    size_t aligned = align_up(size, granularity);

    // 1. Reserve virtual address (stays fixed for the allocation's lifetime).
    void *vptr = nullptr;
    ze_result_t rc = zeVirtualMemReserve(pdc.ze_ctx, nullptr, aligned, &vptr);
    if (rc != ZE_RESULT_SUCCESS || !vptr) {
      XPU_ERR("zeVirtualMemReserve failed: 0x" << std::hex << rc);
      return cudaErrorMemoryAllocation;
    }

    // 2. Create physical memory.
    ze_physical_mem_desc_t pdesc = {};
    pdesc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
    pdesc.size = aligned;
    ze_physical_mem_handle_t phys = {};
    rc = zePhysicalMemCreate(pdc.ze_ctx, pdc.ze_dev, &pdesc, &phys);
    if (rc != ZE_RESULT_SUCCESS) {
      zeVirtualMemFree(pdc.ze_ctx, vptr, aligned);
      XPU_ERR("zePhysicalMemCreate failed: 0x" << std::hex << rc);
      return cudaErrorMemoryAllocation;
    }

    // 3. Map physical into the reserved virtual range.
    rc = zeVirtualMemMap(pdc.ze_ctx, vptr, aligned, phys, 0,
                         ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
    if (rc != ZE_RESULT_SUCCESS) {
      zePhysicalMemDestroy(pdc.ze_ctx, phys);
      zeVirtualMemFree(pdc.ze_ctx, vptr, aligned);
      XPU_ERR("zeVirtualMemMap failed: 0x" << std::hex << rc);
      return cudaErrorMemoryAllocation;
    }

    *ptr = vptr;
    {
      const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
      AllocationMetadata metadata;
      metadata.size = size;
      metadata.device = device;
      metadata.tag = tag;
      metadata.state = AllocationState::ACTIVE;
      metadata.enable_cpu_backup = enable_cpu_backup;
      metadata.cpu_backup = nullptr;
      metadata.aligned_size = aligned;
      metadata.xpu.ze_ctx = pdc.ze_ctx;
      metadata.xpu.ze_dev = pdc.ze_dev;
      metadata.xpu.ze_phys = phys;
      allocation_metadata.emplace(*ptr, metadata);
    }
    XPU_LOG("malloc ptr=" << *ptr << " size=" << size << " aligned=" << aligned
                          << " tag=" << tag);
    return cudaSuccess;
  } catch (const std::exception &e) {
    XPU_ERR("xpu_malloc exception: " << e.what());
    return cudaErrorMemoryAllocation;
  }
}

// ------------------------------------------------------------------ free
cudaError_t xpu_free(
    void *ptr,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  AllocationMetadata metadata;
  {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
    auto it = allocation_metadata.find(ptr);
    if (it == allocation_metadata.end())
      return cudaErrorInvalidDevicePointer;
    metadata = it->second;
    allocation_metadata.erase(it);
  }

  try {
    ze_context_handle_t ze_ctx = metadata.xpu.ze_ctx;
    size_t aligned = metadata.aligned_size;
    if (metadata.state == AllocationState::ACTIVE) {
      zeVirtualMemUnmap(ze_ctx, ptr, aligned);
      zePhysicalMemDestroy(ze_ctx, metadata.xpu.ze_phys);
    }
    // If PAUSED, the physical handle was already destroyed in xpu_pause().
    zeVirtualMemFree(ze_ctx, ptr, aligned);
    if (metadata.cpu_backup)
      std::free(metadata.cpu_backup);
    XPU_LOG("free ptr=" << ptr << " size=" << metadata.size);
    return cudaSuccess;
  } catch (const std::exception &e) {
    XPU_ERR("xpu_free exception: " << e.what());
    return cudaErrorMemoryAllocation;
  }
}

// ------------------------------------------------------------------ pause
void xpu_pause(
    const std::string &tag,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  try {
    for (auto &[ptr, metadata] : allocation_metadata) {
      if (!tag.empty() && metadata.tag != tag)
        continue;
      if (metadata.state != AllocationState::ACTIVE)
        continue;

      ze_context_handle_t ze_ctx = metadata.xpu.ze_ctx;
      size_t aligned = metadata.aligned_size;

      if (metadata.enable_cpu_backup) {
        if (!metadata.cpu_backup)
          metadata.cpu_backup = std::malloc(metadata.size);
        if (!metadata.cpu_backup) {
          XPU_ERR("cpu backup malloc failed for ptr=" << ptr
                  << " size=" << metadata.size << "; skipping pause");
          continue;
        }
        bool memcpy_ok = false;
        try {
          PerDeviceContext &pdc = get_device_context(metadata.device);
          pdc.sycl_queue.memcpy(metadata.cpu_backup, ptr, metadata.size).wait();
          memcpy_ok = true;
        } catch (...) {
          XPU_ERR("cpu backup memcpy failed for ptr=" << ptr << "; skipping pause");
        }
        if (!memcpy_ok) {
          std::free(metadata.cpu_backup);
          metadata.cpu_backup = nullptr;
          continue;
        }
      }

      ze_result_t rc_unmap = zeVirtualMemUnmap(ze_ctx, ptr, aligned);
      ze_result_t rc_destroy =
          zePhysicalMemDestroy(ze_ctx, metadata.xpu.ze_phys);
      if (rc_unmap != ZE_RESULT_SUCCESS)
        XPU_ERR("pause zeVirtualMemUnmap failed: 0x" << std::hex << rc_unmap);
      if (rc_destroy != ZE_RESULT_SUCCESS)
        XPU_ERR("pause zePhysicalMemDestroy failed: 0x" << std::hex
                                                        << rc_destroy);
      metadata.xpu.ze_phys = {};
      metadata.state = AllocationState::PAUSED;
      XPU_LOG("pause ptr=" << ptr << " size=" << metadata.size
                           << " tag=" << metadata.tag);
    }
  } catch (const std::exception &e) {
    XPU_ERR("xpu_pause exception: " << e.what());
  }
}

// ------------------------------------------------------------------ resume
void xpu_resume(
    const std::string &tag,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  try {
    for (auto &[ptr, metadata] : allocation_metadata) {
      if (!tag.empty() && metadata.tag != tag)
        continue;
      if (metadata.state != AllocationState::PAUSED)
        continue;

      ze_context_handle_t ze_ctx = metadata.xpu.ze_ctx;
      ze_device_handle_t ze_dev = metadata.xpu.ze_dev;
      size_t aligned = metadata.aligned_size;

      ze_physical_mem_desc_t pdesc = {};
      pdesc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
      pdesc.size = aligned;
      ze_physical_mem_handle_t phys = {};
      ze_result_t rc = zePhysicalMemCreate(ze_ctx, ze_dev, &pdesc, &phys);
      if (rc != ZE_RESULT_SUCCESS) {
        XPU_ERR("resume zePhysicalMemCreate failed: 0x" << std::hex << rc);
        continue;
      }
      rc = zeVirtualMemMap(ze_ctx, ptr, aligned, phys, 0,
                           ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
      if (rc != ZE_RESULT_SUCCESS) {
        zePhysicalMemDestroy(ze_ctx, phys);
        XPU_ERR("resume zeVirtualMemMap failed: 0x" << std::hex << rc);
        continue;
      }
      metadata.xpu.ze_phys = phys;
      metadata.state = AllocationState::ACTIVE;

      if (metadata.enable_cpu_backup && metadata.cpu_backup) {
        try {
          PerDeviceContext &pdc = get_device_context(metadata.device);
          pdc.sycl_queue.memcpy(ptr, metadata.cpu_backup, metadata.size).wait();
        } catch (...) {
          XPU_ERR("cpu restore memcpy failed for ptr=" << ptr);
        }
        std::free(metadata.cpu_backup);
        metadata.cpu_backup = nullptr;
      }
      XPU_LOG("resume ptr=" << ptr << " size=" << metadata.size
                            << " tag=" << metadata.tag);
    }
  } catch (const std::exception &e) {
    XPU_ERR("xpu_resume exception: " << e.what());
  }
}

// --------------------------------------------------------- passthrough alloc
void *xpu_passthrough_malloc(CUdevice device, size_t size) {
  try {
    PerDeviceContext &pdc = get_device_context(device);
    return sycl::malloc_device(size, pdc.sycl_dev, pdc.sycl_ctx);
  } catch (const std::exception &e) {
    XPU_ERR("xpu_passthrough_malloc exception: " << e.what());
    return nullptr;
  }
}

void xpu_passthrough_free(void *ptr, CUdevice device) {
  try {
    PerDeviceContext &pdc = get_device_context(device);
    sycl::free(ptr, pdc.sycl_ctx);
  } catch (const std::exception &e) {
    XPU_ERR("xpu_passthrough_free exception: " << e.what());
  }
}

bool xpu_is_managed(
    void *ptr,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  return allocation_metadata.find(ptr) != allocation_metadata.end();
}

void xpu_prewarm_devices(int n_devices) {
  for (int i = 0; i < n_devices; i++) {
    try {
      get_device_context(i);
    } catch (const std::exception &e) {
      XPU_ERR("prewarm device " << i << " exception: " << e.what());
    }
  }
}

uint64_t xpu_device_free_bytes(int device_id) {
  try {
    static std::once_flag zes_init_flag;
    static bool zes_ok = false;
    std::call_once(zes_init_flag,
                   [] { zes_ok = (zesInit(0) == ZE_RESULT_SUCCESS); });
    if (!zes_ok)
      return 0;

    PerDeviceContext &pdc = get_device_context(device_id);
    ze_device_properties_t core_props = {};
    core_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    if (zeDeviceGetProperties(pdc.ze_dev, &core_props) != ZE_RESULT_SUCCESS)
      return 0;

    uint32_t sdc = 0;
    if (zesDriverGet(&sdc, nullptr) != ZE_RESULT_SUCCESS || sdc == 0)
      return 0;
    std::vector<zes_driver_handle_t> sdrivers(sdc);
    if (zesDriverGet(&sdc, sdrivers.data()) != ZE_RESULT_SUCCESS)
      return 0;

    for (auto sd : sdrivers) {
      uint32_t sdev = 0;
      if (zesDeviceGet(sd, &sdev, nullptr) != ZE_RESULT_SUCCESS)
        continue;
      std::vector<zes_device_handle_t> sdevs(sdev);
      if (zesDeviceGet(sd, &sdev, sdevs.data()) != ZE_RESULT_SUCCESS)
        continue;
      for (auto cand : sdevs) {
        zes_device_properties_t sp = {};
        sp.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (zesDeviceGetProperties(cand, &sp) != ZE_RESULT_SUCCESS)
          continue;
        if (std::memcmp(sp.core.uuid.id, core_props.uuid.id,
                        ZE_MAX_DEVICE_UUID_SIZE) != 0)
          continue;
        uint32_t count = 0;
        if (zesDeviceEnumMemoryModules(cand, &count, nullptr) !=
                ZE_RESULT_SUCCESS ||
            count == 0)
          return 0;
        std::vector<zes_mem_handle_t> mems(count);
        if (zesDeviceEnumMemoryModules(cand, &count, mems.data()) !=
                ZE_RESULT_SUCCESS)
          return 0;
        uint64_t free_total = 0;
        for (auto m : mems) {
          zes_mem_state_t st = {};
          st.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
          if (zesMemoryGetState(m, &st) == ZE_RESULT_SUCCESS)
            free_total += st.free;
        }
        return free_total;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    XPU_ERR("xpu_device_free_bytes exception: " << e.what());
    return 0;
  }
}

uint64_t xpu_committed_bytes(
    int device_id,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  uint64_t total = 0;
  for (const auto &kv : allocation_metadata) {
    const AllocationMetadata &metadata = kv.second;
    if ((int)metadata.device != device_id)
      continue;
    if (metadata.state == AllocationState::ACTIVE)
      total += metadata.aligned_size;
  }
  return total;
}

} // namespace XPUImplementation

#endif // defined(USE_XPU)
