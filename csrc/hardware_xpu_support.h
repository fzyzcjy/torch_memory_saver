#pragma once
#include "macro.h"
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(USE_XPU)

#include <level_zero/ze_api.h>

enum class AllocationState;
struct AllocationMetadata;

struct XPUAllocExtra {
    ze_context_handle_t ze_ctx = {};
    ze_device_handle_t ze_dev = {};
    ze_physical_mem_handle_t ze_phys = {};
    bool leaked = false;
};

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

    cudaError_t xpu_pause(
        const std::string &tag,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    cudaError_t xpu_resume(
        const std::string &tag,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );


    uint32_t xpu_affected_devices(
        const char *tag,
        int *out_device_ids,
        uint32_t capacity,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    void xpu_prewarm_devices(int n_devices);

    uint64_t xpu_device_free_bytes(int device_id);

    uint64_t xpu_committed_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    uint64_t xpu_leaked_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );

    uint64_t xpu_tracked_bytes(
        int device_id,
        std::unordered_map<void *, AllocationMetadata> &allocation_metadata,
        std::mutex &allocator_metadata_mutex
    );
}

#endif  // defined(USE_XPU)
