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

#if TMS_ROCM_LEGACY_CHUNKED
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;
    // Disk backup: when enabled, the paused allocation is spilled to a fixed
    // node-local file instead of a pinned CPU buffer. disk_fd is opened lazily
    // on the first pause and reused for the allocation's lifetime; the file is
    // preallocated once to `size` and overwritten in place every pause (so disk
    // usage never grows across RL steps).
    bool enable_disk_backup;
    int disk_fd;
    std::string disk_path;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    // CUDA and ROCm 7.0+: Single allocation handle
    CUmemGenericAllocationHandle allocHandle;
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup, bool enable_disk_backup);
    cudaError_t free(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);
    void set_disk_backup_dir(const std::string& dir) {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        disk_backup_dir_ = dir;
    }

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    // Lazily allocate the shared pinned staging buffer used to stream
    // GPU<->disk in fixed-size chunks (bounds host RAM regardless of the total
    // amount offloaded). Callers must hold allocator_metadata_mutex_.
    void ensure_disk_staging_();
    void disk_offload_(void* ptr, AllocationMetadata& metadata);
    void disk_reload_(void* ptr, AllocationMetadata& metadata);

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;

    // Disk-backup state (guarded by allocator_metadata_mutex_).
    std::string disk_backup_dir_;
    size_t disk_chunk_bytes_ = 0;
    void* disk_staging_ = nullptr;      // pinned host, size == disk_chunk_bytes_
    uint64_t disk_backup_counter_ = 0;  // unique suffix per allocation
};
