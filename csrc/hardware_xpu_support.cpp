#include "hardware_xpu_support.h"
#include "core.h"

#if defined(USE_XPU)

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>


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

bool xpu_test_fault(const char *env_name) {
  const char *v = std::getenv(env_name);
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

ze_context_handle_t ze_context_of(const sycl::context &ctx) {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
}

ze_device_handle_t ze_device_of(const sycl::device &dev) {
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
}

size_t ze_alloc_granularity(ze_context_handle_t ze_ctx,
                            ze_device_handle_t ze_dev, size_t size) {
  size_t page_size = 0;
  ze_result_t rc = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, size, &page_size);
  if (rc != ZE_RESULT_SUCCESS || page_size == 0)
    page_size = 2 * 1024 * 1024; // fallback 2 MiB
  return page_size;
}

struct PerDeviceContext {
  sycl::device sycl_dev;
  sycl::context sycl_ctx;
  sycl::queue sycl_queue;
  ze_context_handle_t ze_ctx;
  ze_device_handle_t ze_dev;
};

const std::vector<sycl::device> &l0_gpu_devices() {
  static std::vector<sycl::device> *cache = nullptr;
  if (cache && !cache->empty())
    return *cache;
  if (!cache)
    cache = new std::vector<sycl::device>();
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
  pdc->sycl_queue = sycl::queue(pdc->sycl_dev);
  pdc->sycl_ctx = pdc->sycl_queue.get_context();
  pdc->ze_ctx = ze_context_of(pdc->sycl_ctx);
  pdc->ze_dev = ze_device_of(pdc->sycl_dev);
  ctx_map->emplace(device_id, pdc);
  return *pdc;
}

} // namespace

namespace XPUImplementation {

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
    size_t granularity = ze_alloc_granularity(pdc.ze_ctx, pdc.ze_dev, size);
    size_t aligned = align_up(size, granularity);

    void *vptr = nullptr;
    ze_result_t rc = zeVirtualMemReserve(pdc.ze_ctx, nullptr, aligned, &vptr);
    if (rc != ZE_RESULT_SUCCESS || !vptr) {
      XPU_ERR("zeVirtualMemReserve failed: 0x" << std::hex << rc);
      return cudaErrorMemoryAllocation;
    }

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
      metadata.raw_size = size;
      metadata.device = device;
      metadata.tag = tag;
      metadata.state = AllocationState::ACTIVE;
      metadata.enable_cpu_backup = enable_cpu_backup;
      metadata.cpu_backup = nullptr;
      metadata.enable_disk_backup = false;
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

cudaError_t xpu_free(
    void *ptr,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  auto it = allocation_metadata.find(ptr);
  if (it == allocation_metadata.end())
    return cudaErrorInvalidDevicePointer;
  AllocationMetadata &metadata = it->second;

  try {
    ze_context_handle_t ze_ctx = metadata.xpu.ze_ctx;
    size_t aligned = metadata.aligned_size;

    if (metadata.state == AllocationState::ACTIVE) {
      ze_result_t rc;
      if (xpu_test_fault("TMS_XPU_FAULT_FREE_UNMAP")) {
        rc = ZE_RESULT_ERROR_UNKNOWN; // fault WITHOUT unmapping: stays retryable
      } else {
        rc = zeVirtualMemUnmap(ze_ctx, ptr, aligned);
      }
      if (rc != ZE_RESULT_SUCCESS) {
        XPU_ERR("free zeVirtualMemUnmap failed: 0x" << std::hex << rc
                                                    << "; keeping ACTIVE");
        return cudaErrorMemoryAllocation;
      }
      metadata.state = AllocationState::PAUSED;
      metadata.xpu.leaked = true;
    }

    if (metadata.xpu.leaked) {
      ze_result_t rc = ZE_RESULT_SUCCESS;
      if (xpu_test_fault("TMS_XPU_FAULT_FREE_DESTROY")) {
        rc = ZE_RESULT_ERROR_UNKNOWN; // fault WITHOUT destroying: ze_phys stays valid
      } else {
        rc = zePhysicalMemDestroy(ze_ctx, metadata.xpu.ze_phys);
      }
      if (rc != ZE_RESULT_SUCCESS) {
        XPU_ERR("free zePhysicalMemDestroy failed: 0x" << std::hex << rc
                << "; physical handle retained + leaked (tracked)");
        return cudaErrorMemoryAllocation;
      }
      metadata.xpu.ze_phys = {};
      metadata.xpu.leaked = false;
    }

    ze_result_t rc = ZE_RESULT_SUCCESS;
    if (xpu_test_fault("TMS_XPU_FAULT_FREE_VA")) {
      rc = ZE_RESULT_ERROR_UNKNOWN;
    } else {
      rc = zeVirtualMemFree(ze_ctx, ptr, aligned);
    }
    if (rc != ZE_RESULT_SUCCESS) {
      XPU_ERR("free zeVirtualMemFree failed: 0x" << std::hex << rc
                                                 << "; VA retained (tracked)");
      return cudaErrorMemoryAllocation;
    }

    // Fully released: only now drop ownership.
    if (metadata.cpu_backup)
      std::free(metadata.cpu_backup);
    XPU_LOG("free ptr=" << ptr << " size=" << metadata.raw_size);
    allocation_metadata.erase(it);
    return cudaSuccess;
  } catch (const std::exception &e) {
    XPU_ERR("xpu_free exception: " << e.what());
    return cudaErrorMemoryAllocation;
  }
}

cudaError_t xpu_pause(
    const std::string &tag,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  cudaError_t first_err = cudaSuccess;
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
          metadata.cpu_backup = std::malloc(metadata.raw_size);
        if (!metadata.cpu_backup) {
          XPU_ERR("cpu backup malloc failed for ptr=" << ptr
                  << " size=" << metadata.raw_size << "; keeping ACTIVE");
          if (first_err == cudaSuccess)
            first_err = cudaErrorMemoryAllocation;
          continue;
        }
        bool memcpy_ok = false;
        try {
          PerDeviceContext &pdc = get_device_context(metadata.device);
          pdc.sycl_queue.memcpy(metadata.cpu_backup, ptr, metadata.raw_size).wait();
          memcpy_ok = true;
        } catch (...) {
          XPU_ERR("cpu backup memcpy failed for ptr=" << ptr << "; keeping ACTIVE");
        }
        if (!memcpy_ok) {
          std::free(metadata.cpu_backup);
          metadata.cpu_backup = nullptr;
          if (first_err == cudaSuccess)
            first_err = cudaErrorMemoryAllocation;
          continue;
        }
      }

      ze_result_t rc_unmap = zeVirtualMemUnmap(ze_ctx, ptr, aligned);
      if (rc_unmap != ZE_RESULT_SUCCESS) {
        XPU_ERR("pause zeVirtualMemUnmap failed: 0x" << std::hex << rc_unmap
                                                     << "; keeping ACTIVE");
        if (first_err == cudaSuccess)
          first_err = cudaErrorMemoryAllocation;
        continue;
      }

      ze_result_t rc_destroy;
      if (xpu_test_fault("TMS_XPU_FAULT_PAUSE_DESTROY")) {
        rc_destroy = ZE_RESULT_ERROR_UNKNOWN;
      } else {
        rc_destroy = zePhysicalMemDestroy(ze_ctx, metadata.xpu.ze_phys);
      }
      if (rc_destroy != ZE_RESULT_SUCCESS) {
        XPU_ERR("pause zePhysicalMemDestroy failed: 0x"
                << std::hex << rc_destroy
                << "; physical handle retained + leaked (tracked)");
        metadata.state = AllocationState::PAUSED;
        metadata.xpu.leaked = true;
        if (first_err == cudaSuccess)
          first_err = cudaErrorMemoryAllocation;
        continue;
      }

      metadata.xpu.ze_phys = {};
      metadata.xpu.leaked = false;
      metadata.state = AllocationState::PAUSED;
      XPU_LOG("pause ptr=" << ptr << " size=" << metadata.raw_size
                           << " tag=" << metadata.tag);
    }
  } catch (const std::exception &e) {
    XPU_ERR("xpu_pause exception: " << e.what());
    if (first_err == cudaSuccess)
      first_err = cudaErrorMemoryAllocation;
  }
  return first_err;
}

cudaError_t xpu_resume(
    const std::string &tag,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  cudaError_t first_err = cudaSuccess;
  try {
    for (auto &[ptr, metadata] : allocation_metadata) {
      if (!tag.empty() && metadata.tag != tag)
        continue;
      if (metadata.state != AllocationState::PAUSED)
        continue;

      ze_context_handle_t ze_ctx = metadata.xpu.ze_ctx;
      ze_device_handle_t ze_dev = metadata.xpu.ze_dev;
      size_t aligned = metadata.aligned_size;

      const bool reuse_leaked = metadata.xpu.leaked;
      ze_physical_mem_handle_t phys = {};
      if (reuse_leaked) {
        phys = metadata.xpu.ze_phys;
      } else {
        ze_physical_mem_desc_t pdesc = {};
        pdesc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
        pdesc.size = aligned;
        ze_result_t rc = zePhysicalMemCreate(ze_ctx, ze_dev, &pdesc, &phys);
        if (rc == ZE_RESULT_SUCCESS &&
            xpu_test_fault("TMS_XPU_FAULT_RESUME_CREATE")) {
          zePhysicalMemDestroy(ze_ctx, phys); // undo the real create
          rc = ZE_RESULT_ERROR_UNKNOWN;       // simulate create failure
        }
        if (rc != ZE_RESULT_SUCCESS) {
          XPU_ERR("resume zePhysicalMemCreate failed: 0x" << std::hex << rc);
          if (first_err == cudaSuccess)
            first_err = cudaErrorMemoryAllocation;
          continue;
        }
      }

      ze_result_t rc = zeVirtualMemMap(ze_ctx, ptr, aligned, phys, 0,
                                       ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
      if (rc == ZE_RESULT_SUCCESS &&
          xpu_test_fault("TMS_XPU_FAULT_RESUME_MAP")) {
        zeVirtualMemUnmap(ze_ctx, ptr, aligned); // undo the real map
        rc = ZE_RESULT_ERROR_UNKNOWN;            // simulate map failure
      }
      if (rc != ZE_RESULT_SUCCESS) {
        if (!reuse_leaked)
          zePhysicalMemDestroy(ze_ctx, phys);
        XPU_ERR("resume zeVirtualMemMap failed: 0x" << std::hex << rc);
        if (first_err == cudaSuccess)
          first_err = cudaErrorMemoryAllocation;
        continue;
      }

      if (metadata.enable_cpu_backup && metadata.cpu_backup) {
        bool restore_ok = false;
        try {
          if (xpu_test_fault("TMS_XPU_FAULT_RESUME_RESTORE"))
            throw std::runtime_error("injected restore fault");
          PerDeviceContext &pdc = get_device_context(metadata.device);
          pdc.sycl_queue.memcpy(ptr, metadata.cpu_backup, metadata.raw_size).wait();
          restore_ok = true;
        } catch (const std::exception &e) {
          XPU_ERR("cpu restore memcpy failed for ptr=" << ptr << ": "
                                                       << e.what());
        } catch (...) {
          XPU_ERR("cpu restore memcpy failed for ptr=" << ptr);
        }
        if (!restore_ok) {
          zeVirtualMemUnmap(ze_ctx, ptr, aligned);
          if (!reuse_leaked)
            zePhysicalMemDestroy(ze_ctx, phys);
          if (first_err == cudaSuccess)
            first_err = cudaErrorMemoryAllocation;
          continue;
        }
      }

      metadata.xpu.ze_phys = phys;
      metadata.xpu.leaked = false;
      metadata.state = AllocationState::ACTIVE;
      if (metadata.cpu_backup) {
        std::free(metadata.cpu_backup);
        metadata.cpu_backup = nullptr;
      }
      XPU_LOG("resume ptr=" << ptr << " size=" << metadata.raw_size
                            << " tag=" << metadata.tag);
    }
  } catch (const std::exception &e) {
    XPU_ERR("xpu_resume exception: " << e.what());
    if (first_err == cudaSuccess)
      first_err = cudaErrorMemoryAllocation;
  }
  return first_err;
}

uint32_t xpu_affected_devices(
    const char *tag,
    int *out_device_ids,
    uint32_t capacity,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  const bool all = (tag == nullptr || tag[0] == '\0');
  std::vector<int> devices;
  for (const auto &kv : allocation_metadata) {
    const AllocationMetadata &metadata = kv.second;
    if (!all && metadata.tag != tag)
      continue;
    int dev = (int)metadata.device;
    if (std::find(devices.begin(), devices.end(), dev) == devices.end())
      devices.push_back(dev);
  }
  std::sort(devices.begin(), devices.end());
  if (out_device_ids != nullptr) {
    uint32_t n = std::min<uint32_t>(capacity, (uint32_t)devices.size());
    for (uint32_t i = 0; i < n; i++)
      out_device_ids[i] = devices[i];
  }
  return (uint32_t)devices.size();
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

uint64_t xpu_tracked_bytes(
    int device_id,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  uint64_t total = 0;
  for (const auto &kv : allocation_metadata) {
    const AllocationMetadata &metadata = kv.second;
    if ((int)metadata.device != device_id)
      continue;
    total += metadata.aligned_size;
  }
  return total;
}

uint64_t xpu_leaked_bytes(
    int device_id,
    std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
    std::mutex &allocator_metadata_mutex) {
  const std::lock_guard<std::mutex> lock(allocator_metadata_mutex);
  uint64_t total = 0;
  for (const auto &kv : allocation_metadata) {
    const AllocationMetadata &metadata = kv.second;
    if ((int)metadata.device != device_id)
      continue;
    if (metadata.xpu.leaked)
      total += metadata.aligned_size;
  }
  return total;
}

} // namespace XPUImplementation

#endif // defined(USE_XPU)
