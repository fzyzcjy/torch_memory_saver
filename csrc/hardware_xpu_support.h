#pragma once
#include "macro.h"
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(USE_XPU)

#include <level_zero/ze_api.h>

// Forward declarations (defined in core.h, which includes this header).
enum class AllocationState;
struct AllocationMetadata;

// Per-allocation Level Zero state, embedded in AllocationMetadata (see core.h).
// The virtual address (the map key in allocation_metadata) stays reserved for
// the lifetime of the allocation; only the physical handle is created/destroyed
// across pause/resume.
struct XPUAllocExtra {
    ze_context_handle_t ze_ctx = {};
    ze_device_handle_t ze_dev = {};
    ze_physical_mem_handle_t ze_phys = {};
    // Set when a physical handle can neither be released nor rolled back
    // (zePhysicalMemDestroy failed AND unmap couldn't re-map): unusable but kept
    // tracked so orphaned bytes stay visible via tms_xpu_leaked_bytes, not dropped.
    bool leaked = false;
};

// High-level XPU implementation, mirroring ROCmHIPImplementation. The shared
// allocator metadata map + mutex live in the TorchMemorySaver instance and are
// passed in so this backend stays free of global state beyond the per-device
// SYCL contexts. Return cudaError_t (int on XPU) to match core.h's
// platform-agnostic interface; see macro.h for why.
namespace XPUImplementation {
    cudaError_t xpu_malloc(
        void **ptr,
        CUdevice device,
        size_t size,
        const std::string &tag,
        bool enable_cpu_backup,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    cudaError_t xpu_free(
        void *ptr,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // cudaSuccess only if every matching allocation cleanly paused; else an error,
    // left recoverable (a failed destroy RETAINS the handle, see xpu_leaked_bytes).
    cudaError_t xpu_pause(
        const std::string &tag,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // cudaSuccess only if every matching allocation fully resumed; else an error.
    // Transactional: a failed allocation is left cleanly PAUSED, so resume retries.
    cudaError_t xpu_resume(
        const std::string &tag,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // Distinct device ids with an allocation matching `tag` (null/empty = all), so
    // Python drains what pause/resume unmap. Writes sorted ids up to `capacity`.
    uint32_t xpu_affected_devices(
        const char *tag,
        int *out_device_ids,
        uint32_t capacity,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // Pre-warm per-device SYCL contexts before the pluggable allocator is
    // registered (creating a sycl::context inside an allocator callback can
    // deadlock the SYCL runtime).
    void xpu_prewarm_devices(int n_devices);

    // Authoritative free-device-memory reading via sysman. torch's allocator
    // accounting does not reflect physical pages released by zeVirtualMemUnmap.
    uint64_t xpu_device_free_bytes(int device_id);

    // Committed physical bytes held on a device: sum of aligned_size over ACTIVE
    // allocations. Driver-independent (unlike sysman free-bytes, frozen/deprecated
    // on newer drivers); drops to 0 on pause(), restored on resume().
    uint64_t xpu_committed_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // Physical bytes whose Level Zero handle couldn't be released, retained
    // (leaked=true); separate from committed so a cleanup-failure leak is visible.
    uint64_t xpu_leaked_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // Bytes still TRACKED on a device (any state); unlike committed/leaked, stays > 0
    // while the ownership record exists -- proving a failing free retains it.
    uint64_t xpu_tracked_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );
}

#endif  // defined(USE_XPU)
