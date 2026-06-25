#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "utils.h"
#include "macro.h"
#include "disk_backend.h"

#if TMS_ROCM_LEGACY_CHUNKED
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

#if defined(USE_XPU)
#include "hardware_xpu_support.h"
#endif

struct AllocationMetadata {
    size_t raw_size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;
    bool enable_disk_backup;
    DiskBackupSlot disk;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#elif defined(USE_XPU)
    // Intel XPU (Level Zero): keep the reserved virtual address mapped to a
    // physical handle that can be unmapped (pause) and re-created (resume).
    // Fields are defined in hardware_xpu_support.h (XPUAllocExtra).
    size_t aligned_size;
    XPUAllocExtra xpu;
#else
    // CUDA and ROCm 7.0+: Single allocation handle
    size_t allocation_size;
    CUmemGenericAllocationHandle allocHandle;
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(
        void** ptr,
        CUdevice device,
        size_t raw_size,
        const std::string& tag,
        bool enable_cpu_backup,
        bool enable_disk_backup);
    cudaError_t free(void *ptr);

    // Whether ptr is tracked by the VMM path (vs a passthrough device alloc).
    // Used by the XPU torch-mode hook to route frees of non-region allocations.
    bool is_managed(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);
    void set_disk_backup_dir(const std::string& dir) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backend_.set_dir(dir);
    }

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;

    // Guarded by allocator_metadata_mutex_.
    DiskBackend disk_backend_;
};
