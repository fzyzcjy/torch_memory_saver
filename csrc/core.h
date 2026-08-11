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
#include "cpu_backup.h"
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
    CpuBackupSlot cpu_backup;
    bool enable_disk_backup;
    DiskBackupSlot disk;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#elif defined(USE_XPU)
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
        CpuBackupKind cpu_backup_kind,
        bool enable_disk_backup);
    cudaError_t free(void *ptr);

    cudaError_t pause(const std::string& tag);
    cudaError_t resume(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
#if defined(USE_XPU)
        if (value != 0) {
            std::cerr << "[torch_memory_saver.cpp] set_memory_margin_bytes("
                      << value << ") ignored: NOT supported on Intel XPU "
                         "(OOM-margin guard needs device free-bytes the driver "
                         "reports frozen). Manage headroom outside torch_memory_saver."
                      << std::endl;
        }
        (void)value;
#else
        memory_margin_bytes_.store(value);
#endif
    }
    void set_retain_cpu_backup(bool value) {
        retain_cpu_backup_.store(value);
    }
    bool get_retain_cpu_backup() const {
        return retain_cpu_backup_.load();
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);
    void set_disk_backup_dir(const std::string& dir) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backend_.set_dir(dir);
    }

#if defined(USE_XPU)
    uint64_t xpu_committed_bytes(int device_id);
    uint64_t xpu_leaked_bytes(int device_id);
    uint64_t xpu_tracked_bytes(int device_id);
    uint32_t xpu_affected_devices(const char* tag, int* out_device_ids, uint32_t capacity);
#endif

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;
    std::atomic<bool> retain_cpu_backup_ = false;

    // Guarded by allocator_metadata_mutex_.
    DiskBackend disk_backend_;
};
