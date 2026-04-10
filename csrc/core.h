#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <cstdint>
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

// Per-chunk state for chunked allocations. chunk.size is guaranteed to be
// a multiple of the device allocation granularity (enforced at malloc time).
struct ChunkInfo {
    AllocationState state;
    CUmemGenericAllocationHandle allocHandle;
    void* cpu_backup;
    size_t size;
    size_t offset;
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    // Only meaningful when !is_chunked. For chunked allocations the per-chunk
    // state in `chunks[i].state` is the source of truth.
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;

    // True if this allocation was created with chunk_size > 0. When true, the
    // `chunks` vector is populated and all mapping state lives there; the
    // single-handle fields below are unused.
    bool is_chunked;
    std::vector<ChunkInfo> chunks;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    // CUDA and ROCm 7.0+: Single allocation handle (only when !is_chunked)
    CUmemGenericAllocationHandle allocHandle;
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup, size_t chunk_size = 0);
    cudaError_t free(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);

    // Per-chunk pause/resume. Silently no-op on chunks already in the target
    // state, so these compose freely with pause()/resume().
    void pause_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices);
    void resume_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices);

    // Query number of chunks for a given tag. Requires a non-empty tag that
    // uniquely identifies a single allocation; errors out otherwise. Returns
    // 0 if no allocation matches, 1 for non-chunked allocations.
    size_t get_num_chunks(const std::string& tag);

    // Fill `out_active` (length num_chunks) with 1 for ACTIVE chunks and 0
    // for PAUSED chunks. Same uniqueness requirement as get_num_chunks.
    void get_chunk_states(const std::string& tag, uint8_t* out_active, size_t num_chunks);

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
