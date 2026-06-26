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
};

// High-level XPU implementation, mirroring ROCmHIPImplementation. The shared
// allocator metadata map + mutex live in the TorchMemorySaver instance and are
// passed in so this backend stays free of global state beyond the per-device
// SYCL contexts. Functions return cudaError_t (int on XPU) to match the
// platform-agnostic interface in core.h; see macro.h for why.
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

    void xpu_pause(
        const std::string &tag,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    void xpu_resume(
        const std::string &tag,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    // Plain device alloc/free for allocations made OUTSIDE an interesting
    // region. The XPUPluggableAllocator is global and intercepts every device
    // allocation, so non-region allocations must bypass the VMM path.
    void *xpu_passthrough_malloc(CUdevice device, size_t size);
    void xpu_passthrough_free(void *ptr, CUdevice device);

    // Whether a pointer is managed by the VMM path (vs a passthrough alloc).
    bool xpu_is_managed(
        void *ptr,
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

    // Committed physical bytes the saver currently holds on a device: sum of
    // aligned_size over ACTIVE allocations on that device. Driver-independent
    // (unlike sysman free-bytes, which is frozen/deprecated on newer drivers);
    // drops to 0 on pause() and is restored on resume().
    uint64_t xpu_committed_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );
}

#endif  // defined(USE_XPU)
