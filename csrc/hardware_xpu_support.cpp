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

// Intel XPU backend for torch_memory_saver via Level Zero Virtual Memory
// Management (VMM): same pause/resume semantics as the CUDA/ROCm backends.
//
// Architecture:
//   - malloc:  zeVirtualMemReserve → zePhysicalMemCreate → zeVirtualMemMap
//   - pause:   zeVirtualMemUnmap + zePhysicalMemDestroy (VA stays reserved)
//   - resume:  zePhysicalMemCreate + zeVirtualMemMap (to same VA)
//   - free:    unmap + destroy (if ACTIVE), then zeVirtualMemFree
//
// Functions return cudaError_t (typedef'd to int for XPU); ze_result_t is
// translated at the boundary. See macro.h for the rationale.

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

// Test-only fault injection for the transactional pause/resume/free cleanup
// paths. True when the named fault is armed (env var == "1"). Read fresh each
// call (uncached) so a test can arm, observe rollback, disarm, and retry within
// one process. One getenv when unset; never fires in production. Faults:
//   resume: TMS_XPU_FAULT_RESUME_CREATE / _MAP / _RESTORE
//   pause:  TMS_XPU_FAULT_PAUSE_DESTROY
//   free:   TMS_XPU_FAULT_FREE_UNMAP / _DESTROY / _VA
// Each simulates failure WITHOUT the destructive Level Zero call, so the
// resource stays valid and the leak is recoverable on retry -- mirroring the
// driver erroring while leaving the allocation alive.
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
  // Query with the actual size: zeVirtualMemQueryPageSize returns the page size
  // recommended for it (larger allocations may want larger pages). 2 MiB on fail.
  size_t page_size = 0;
  ze_result_t rc = zeVirtualMemQueryPageSize(ze_ctx, ze_dev, size, &page_size);
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

// When the same GPUs are exposed through BOTH a Level-Zero and an OpenCL SYCL
// platform, get_devices(gpu) returns a mixed list (e.g. 8 entries for 4 GPUs).
// torch.xpu enumerates Level-Zero only, so filter to the Level-Zero backend
// before indexing by a torch device ordinal; otherwise an ordinal can land on
// an OpenCL device and get_native<level_zero> fails ("Backends mismatch"). On
// single-platform runtimes every GPU is already Level-Zero: order-preserving
// no-op.
const std::vector<sycl::device> &l0_gpu_devices() {
  // Cache once populated. Do NOT memoize an empty result: a call racing ahead of
  // SYCL/L0 device enumeration must be able to recover later. All callers reach
  // here under the get_device_context mutex, so the non-atomic pointer/contents
  // are race-free.
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
    size_t granularity = ze_alloc_granularity(pdc.ze_ctx, pdc.ze_dev, size);
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
//
// Ownership is dropped (metadata erased, backup freed) ONLY after every needed
// Level Zero release step succeeds. On a step failure the entry is RETAINED,
// its state advanced to what did succeed, and an error returned -- so the held
// resource stays tracked (via xpu_committed_bytes / xpu_leaked_bytes) instead
// of silently dropped, and a later retry resumes where it stopped. Steps by
// state:
//   ACTIVE              -> unmap, destroy, free-VA
//   PAUSED + leaked      -> destroy (handle survived a failed pause), free-VA
//   PAUSED (not leaked)  -> free-VA (handle already destroyed in pause)
// Lock held across the driver calls (as in pause/resume) so no thread observes
// a partially-released entry.
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

    // Step 1: unmap the VA if still mapped (ACTIVE). On failure keep ACTIVE --
    // the physical range is still live, so freeing the VA now would be illegal.
    if (metadata.state == AllocationState::ACTIVE) {
      ze_result_t rc;
      if (xpu_test_fault("TMS_XPU_FAULT_FREE_UNMAP")) {
        // Fault WITHOUT unmapping: VA stays mapped and handle live, so the entry
        // must stay ACTIVE and NOT destroy a still-mapped handle. A later retry
        // (fault cleared) does the real unmap.
        rc = ZE_RESULT_ERROR_UNKNOWN;
      } else {
        rc = zeVirtualMemUnmap(ze_ctx, ptr, aligned);
      }
      if (rc != ZE_RESULT_SUCCESS) {
        XPU_ERR("free zeVirtualMemUnmap failed: 0x" << std::hex << rc
                                                    << "; keeping ACTIVE");
        return cudaErrorMemoryAllocation;
      }
      // Unmapped: handle now orphaned until destroyed. Record it so a failure
      // below leaves a correct (leaked) state to retry from.
      metadata.state = AllocationState::PAUSED;
      metadata.xpu.leaked = true;
    }

    // Step 2: destroy the physical handle if still alive (ACTIVE just unmapped,
    // or a pause whose destroy failed left leaked=true). A plain PAUSED alloc
    // already destroyed its handle in pause -> skip.
    if (metadata.xpu.leaked) {
      ze_result_t rc = ZE_RESULT_SUCCESS;
      if (xpu_test_fault("TMS_XPU_FAULT_FREE_DESTROY")) {
        // Fault WITHOUT destroying: ze_phys stays valid so the retained-leaked
        // state is recoverable by a later retry (mirrors the pause fault).
        rc = ZE_RESULT_ERROR_UNKNOWN;
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

    // Step 3: release the reserved VA. On failure the VA is leaked (tracked via
    // the retained entry, now with no physical handle) and retry-able; do not
    // erase the entry.
    ze_result_t rc = ZE_RESULT_SUCCESS;
    if (xpu_test_fault("TMS_XPU_FAULT_FREE_VA")) {
      // Fault WITHOUT freeing: the reserved VA stays valid for a later retry.
      rc = ZE_RESULT_ERROR_UNKNOWN;
    } else {
      rc = zeVirtualMemFree(ze_ctx, ptr, aligned);
    }
    if (rc != ZE_RESULT_SUCCESS) {
      XPU_ERR("free zeVirtualMemFree failed: 0x" << std::hex << rc
                                                 << "; VA retained (tracked)");
      return cudaErrorMemoryAllocation;
    }

    // Fully released: safe to drop ownership and free the backup.
    if (metadata.cpu_backup)
      std::free(metadata.cpu_backup);
    XPU_LOG("free ptr=" << ptr << " size=" << metadata.size);
    allocation_metadata.erase(it);
    return cudaSuccess;
  } catch (const std::exception &e) {
    // Keep the (possibly state-advanced) entry so nothing is silently dropped.
    XPU_ERR("xpu_free exception: " << e.what());
    return cudaErrorMemoryAllocation;
  }
}

// ------------------------------------------------------------------ pause
//
// Transactional per allocation. A handle is released (state -> PAUSED, ze_phys
// cleared) ONLY after both the backup copy and the unmap+destroy succeed. On a
// step failure the allocation is left in a recoverable state, not force-PAUSED:
//   - backup/unmap failure -> stays ACTIVE (still fully mapped, retry-able)
//   - destroy failure       -> VA unmapped but handle STILL ALIVE, so it is
//     retained (ze_phys kept, xpu.leaked=true) not cleared. Clearing it (old
//     bug) made resume create a second handle and orphan this one. Leaked keeps
//     the bytes visible via tms_xpu_leaked_bytes and lets resume re-map the same
//     handle (see xpu_resume) or free reclaim it.
//
// cudaSuccess only if every matching allocation paused cleanly; else the first
// error, so the Python layer surfaces the leak instead of reporting success.
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
          metadata.cpu_backup = std::malloc(metadata.size);
        if (!metadata.cpu_backup) {
          XPU_ERR("cpu backup malloc failed for ptr=" << ptr
                  << " size=" << metadata.size << "; keeping ACTIVE");
          if (first_err == cudaSuccess)
            first_err = cudaErrorMemoryAllocation;
          continue;
        }
        bool memcpy_ok = false;
        try {
          PerDeviceContext &pdc = get_device_context(metadata.device);
          pdc.sycl_queue.memcpy(metadata.cpu_backup, ptr, metadata.size).wait();
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
        // Stay ACTIVE (do NOT destroy the handle): destroying now orphans the
        // still-mapped VA, and the next resume's zeVirtualMemMap fails on an
        // already-mapped range, breaking the allocation permanently. ACTIVE
        // lets it be retried or freed cleanly.
        XPU_ERR("pause zeVirtualMemUnmap failed: 0x" << std::hex << rc_unmap
                                                     << "; keeping ACTIVE");
        if (first_err == cudaSuccess)
          first_err = cudaErrorMemoryAllocation;
        continue;
      }

      ze_result_t rc_destroy;
      if (xpu_test_fault("TMS_XPU_FAULT_PAUSE_DESTROY")) {
        // Fault WITHOUT destroying: ze_phys stays valid, exercising the
        // retained-handle recovery path end-to-end (resume must re-map it, not
        // create a new one).
        rc_destroy = ZE_RESULT_ERROR_UNKNOWN;
      } else {
        rc_destroy = zePhysicalMemDestroy(ze_ctx, metadata.xpu.ze_phys);
      }
      if (rc_destroy != ZE_RESULT_SUCCESS) {
        // VA already unmapped (memory inaccessible) but handle not released.
        // Retain it (do NOT clear ze_phys) and mark leaked, so committed bytes
        // stay tracked and resume re-maps this exact handle, not a new one.
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
      XPU_LOG("pause ptr=" << ptr << " size=" << metadata.size
                           << " tag=" << metadata.tag);
    }
  } catch (const std::exception &e) {
    XPU_ERR("xpu_pause exception: " << e.what());
    if (first_err == cudaSuccess)
      first_err = cudaErrorMemoryAllocation;
  }
  return first_err;
}

// ------------------------------------------------------------------ resume
//
// Transactional per allocation: committed to ACTIVE (and cpu_backup freed) only
// after obtaining a handle, mapping it, AND restoring data all succeed. The
// handle is freshly created, EXCEPT when pause() could not destroy the previous
// one (metadata.xpu.leaked) -- then that retained handle is re-mapped so it is
// reclaimed, not orphaned. A failing step rolls back to clean PAUSED with backup
// intact (and any retained handle still retained) -- never mapped-but-
// uninitialized nor ACTIVE-but-unmapped. Each allocation is all-or-nothing, so
// resume() is idempotent: on error just retry; committed allocations are skipped
// (state != PAUSED).
//
// cudaSuccess only if every matching allocation committed; else the first error,
// so the Python layer surfaces a half-resumed tag instead of reporting success.
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

      // Step 1: obtain the handle to map. Normally pause() destroyed the old
      // one, so create fresh. But if pause() could not destroy it
      // (metadata.xpu.leaked), the original is STILL alive and merely unmapped
      // -- re-map that exact handle, not a second one (which would orphan the
      // retained one). On create failure nothing changed; stays PAUSED, backup
      // intact.
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

      // Step 2: map the handle into the reserved VA. On failure roll back and
      // stay PAUSED. Destroy the handle only if WE created it this resume; a
      // reused leaked handle must be retained (destroying it would drop the leak
      // and leave ze_phys dangling for the next retry).
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

      // Step 3: restore backed-up contents BEFORE committing, so a failed copy
      // rolls back with the backup still available. state / ze_phys / cpu_backup
      // untouched until commit.
      if (metadata.enable_cpu_backup && metadata.cpu_backup) {
        bool restore_ok = false;
        try {
          if (xpu_test_fault("TMS_XPU_FAULT_RESUME_RESTORE"))
            throw std::runtime_error("injected restore fault");
          PerDeviceContext &pdc = get_device_context(metadata.device);
          pdc.sycl_queue.memcpy(ptr, metadata.cpu_backup, metadata.size).wait();
          restore_ok = true;
        } catch (const std::exception &e) {
          XPU_ERR("cpu restore memcpy failed for ptr=" << ptr << ": "
                                                       << e.what());
        } catch (...) {
          XPU_ERR("cpu restore memcpy failed for ptr=" << ptr);
        }
        if (!restore_ok) {
          // Roll back to clean PAUSED, backup preserved (still the only copy).
          // Unmap, and destroy the handle only if we created it this resume; a
          // reused leaked handle stays retained for a later retry (see Step 1).
          zeVirtualMemUnmap(ze_ctx, ptr, aligned);
          if (!reuse_leaked)
            zePhysicalMemDestroy(ze_ctx, phys);
          if (first_err == cudaSuccess)
            first_err = cudaErrorMemoryAllocation;
          continue;
        }
      }

      // Commit: publish the handle, mark ACTIVE, and only now release the
      // backup. Clearing leaked: a reused handle is fully owned/mapped again.
      metadata.xpu.ze_phys = phys;
      metadata.xpu.leaked = false;
      metadata.state = AllocationState::ACTIVE;
      if (metadata.cpu_backup) {
        std::free(metadata.cpu_backup);
        metadata.cpu_backup = nullptr;
      }
      XPU_LOG("resume ptr=" << ptr << " size=" << metadata.size
                            << " tag=" << metadata.tag);
    }
  } catch (const std::exception &e) {
    XPU_ERR("xpu_resume exception: " << e.what());
    if (first_err == cudaSuccess)
      first_err = cudaErrorMemoryAllocation;
  }
  return first_err;
}

// Distinct devices with an allocation matching `tag`, read from the same map
// (same mutex) xpu_pause/xpu_resume iterate, so the set is exactly what those
// unmap/remap. Matches xpu_pause's tag filter: null/empty tag == all. State is
// irrelevant -- pause unmaps ACTIVE, resume re-maps PAUSED; both need their
// device drained. See the header for the capacity/retry contract.
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

// Bytes still TRACKED on a device: sum of aligned_size over every entry that
// still owns its VA, any state (ACTIVE, PAUSED, PAUSED+leaked). Unlike committed
// (ACTIVE-only) and leaked (leaked-only), this stays > 0 while the ownership
// RECORD exists -- including after a free() whose VA release failed (PAUSED, not
// leaked, no handle), which both other counters report as 0. Lets a test prove
// a failing free retains ownership rather than dropping the record before Level
// Zero confirms release. Drops to 0 only once the entry is released and erased.
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

// Physical bytes whose handle could not be released (a pause/free destroy
// failure retained the handle, leaked=true). NOT counted by xpu_committed_bytes
// (ACTIVE only), so a test watching only committed would see 0 and miss the
// leak. Exposed directly so failure paths assert against real retained
// ownership, not the self-reported ACTIVE state. Drops to 0 once the handle is
// reclaimed (resume re-maps it, or free finally destroys it).
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
