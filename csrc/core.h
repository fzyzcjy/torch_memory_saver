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

// Per-chunk state for chunked allocations
struct ChunkInfo {
    AllocationState state;
    CUmemGenericAllocationHandle allocHandle;
    void* cpu_backup;
    size_t size;    // aligned size of this chunk
    size_t offset;  // offset within the VA range
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;

    // Chunked allocation fields (used when num_chunks > 1)
    size_t num_chunks;
    size_t aligned_total_size;  // total aligned size (may be > size due to alignment)
    std::vector<ChunkInfo> chunks;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    // CUDA and ROCm 7.0+: Single allocation handle (used when num_chunks <= 1)
    CUmemGenericAllocationHandle allocHandle;
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup, size_t num_chunks = 1);
    cudaError_t free(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);

    // Per-chunk pause/resume
    void pause_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices);
    void resume_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices);

    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;
};
