#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"
#include <algorithm>

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

#if !TMS_ROCM_LEGACY_CHUNKED
namespace {

// Pause a single chunk. Caller must hold allocator_metadata_mutex_.
// Silent no-op if the chunk is already paused, so pause_chunks() and the
// whole-allocation pause() compose freely.
void pause_chunk_locked(ChunkInfo& chunk, void* alloc_ptr, bool enable_cpu_backup) {
    if (chunk.state != AllocationState::ACTIVE) return;

    void* chunk_ptr = static_cast<uint8_t*>(alloc_ptr) + chunk.offset;
    if (enable_cpu_backup) {
        if (chunk.cpu_backup == nullptr) {
            CUDA_ERROR_CHECK(cudaMallocHost(&chunk.cpu_backup, chunk.size));
        }
        CUDA_ERROR_CHECK(cudaMemcpy(chunk.cpu_backup, chunk_ptr, chunk.size, cudaMemcpyDeviceToHost));
    }
    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) chunk_ptr, chunk.size));
    CURESULT_CHECK(cuMemRelease(chunk.allocHandle));
    chunk.state = AllocationState::PAUSED;
}

// Resume a single chunk. Caller must hold allocator_metadata_mutex_.
// Silent no-op if the chunk is already active.
void resume_chunk_locked(ChunkInfo& chunk, void* alloc_ptr, CUdevice device, bool enable_cpu_backup) {
    if (chunk.state != AllocationState::PAUSED) return;

    void* chunk_ptr = static_cast<uint8_t*>(alloc_ptr) + chunk.offset;
    CUmemGenericAllocationHandle newAllocHandle;
    CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, chunk.size, device));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) chunk_ptr, chunk.size, 0, newAllocHandle, 0));
    CUDAUtils::cu_mem_set_access(chunk_ptr, chunk.size, device);

    if (enable_cpu_backup) {
        SIMPLE_CHECK(chunk.cpu_backup != nullptr, "cpu_backup should not be nullptr (chunk)");
        CUDA_ERROR_CHECK(cudaMemcpy(chunk_ptr, chunk.cpu_backup, chunk.size, cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaFreeHost(chunk.cpu_backup));
        chunk.cpu_backup = nullptr;
    }

    chunk.allocHandle = newAllocHandle;
    chunk.state = AllocationState::ACTIVE;
}

// Look up the unique allocation matching `tag` using the O(1) tag index.
// Errors out if `tag` is empty or if multiple allocations share the tag —
// the chunk-level APIs use per-allocation chunk indices, so ambiguity would
// be silently wrong. Returns {nullptr, nullptr} if no allocation matches.
std::pair<void*, AllocationMetadata*> find_unique_by_tag_locked(
    std::unordered_map<void*, AllocationMetadata>& allocations,
    const std::unordered_map<std::string, std::vector<void*>>& tag_index,
    const std::string& tag
) {
    SIMPLE_CHECK(!tag.empty(), "chunk-level operations require a non-empty tag");

    auto it = tag_index.find(tag);
    if (it == tag_index.end() || it->second.empty()) {
        return {nullptr, nullptr};
    }
    SIMPLE_CHECK(it->second.size() == 1,
        "chunk-level operations require exactly one allocation per tag, but found "
        + std::to_string(it->second.size()) + " matching tag=" + tag);

    void* ptr = it->second[0];
    auto meta_it = allocations.find(ptr);
    SIMPLE_CHECK(meta_it != allocations.end(), "tag index is out of sync with allocation map");
    return {ptr, &meta_it->second};
}

}  // namespace
#endif

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup, size_t chunk_size) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(chunk_size == 0,
        "chunked allocations (chunk_size > 0) are not supported on ROCm < 7.0 "
        "(TMS_ROCM_LEGACY_CHUNKED build). Use a newer ROCm or omit chunk_size.");
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " (alloc)size=" << size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    // Chunked allocation: multiple independent physical allocations mapped to one contiguous VA range.
    if (chunk_size > 0) {
        // chunk_size must be a multiple of the device allocation granularity.
        // Rounding up silently would make the user's logical chunk boundaries
        // (at offsets i * chunk_size) disagree with the physical chunk
        // boundaries (at offsets i * aligned_chunk_size), so pause_chunks
        // would unmap different bytes than the user expects. Reject instead.
        const size_t granularity = CUDAUtils::cu_mem_get_granularity(device);
        SIMPLE_CHECK(granularity > 0, "device reported zero allocation granularity");
        SIMPLE_CHECK(chunk_size % granularity == 0,
            "chunk_size (" + std::to_string(chunk_size) + " bytes) must be a multiple of the device "
            "allocation granularity (" + std::to_string(granularity) + " bytes).");
        SIMPLE_CHECK(size % chunk_size == 0,
            "Allocation size (" + std::to_string(size) + " bytes) is not divisible by chunk_size ("
            + std::to_string(chunk_size) + " bytes). "
            "Ensure your tensor's total byte size is a multiple of chunk_size.");

        const size_t num_chunks = size / chunk_size;
        SIMPLE_CHECK(num_chunks > 0, "chunk_size is larger than allocation size");

        // Because chunk_size is already granularity-aligned, the total VA
        // reservation equals `size` exactly — no hidden padding.
        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, granularity, 0, 0));

        std::vector<ChunkInfo> chunks(num_chunks);
        for (size_t i = 0; i < num_chunks; i++) {
            chunks[i].offset = i * chunk_size;
            chunks[i].size = chunk_size;
            chunks[i].state = AllocationState::ACTIVE;
            chunks[i].cpu_backup = nullptr;

            cudaError_t ret = CUDAUtils::cu_mem_create(&chunks[i].allocHandle, chunk_size, device);
            if (ret != cudaSuccess) {
                for (size_t j = 0; j < i; j++) {
                    CURESULT_CHECK(cuMemUnmap((CUdeviceptr)((uint8_t*)*ptr + chunks[j].offset), chunks[j].size));
                    CURESULT_CHECK(cuMemRelease(chunks[j].allocHandle));
                }
                CURESULT_CHECK(cuMemAddressFree((CUdeviceptr)*ptr, size));
                return ret;
            }

            CURESULT_CHECK(cuMemMap((CUdeviceptr)((uint8_t*)*ptr + chunks[i].offset),
                                    chunk_size, 0, chunks[i].allocHandle, 0));
            CUDAUtils::cu_mem_set_access((uint8_t*)*ptr + chunks[i].offset, chunk_size, device);
        }

        {
            const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
            AllocationMetadata meta;
            meta.size = size;
            meta.device = device;
            meta.tag = tag;
            meta.state = AllocationState::ACTIVE;
            meta.enable_cpu_backup = enable_cpu_backup;
            meta.cpu_backup = nullptr;
            meta.pauseable_chunks = true;
            meta.chunks = std::move(chunks);
            allocation_metadata_.emplace(*ptr, std::move(meta));
            if (!tag.empty()) tag_index_[tag].push_back(*ptr);
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc (chunked)"
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " chunk_size=" << chunk_size << " num_chunks=" << num_chunks
                  << " granularity=" << granularity << " tag=" << tag
                  << std::endl;
#endif
        return cudaSuccess;
    }

    // Non-chunked allocation (original path)
    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        AllocationMetadata meta;
        meta.size = size;
        meta.device = device;
        meta.tag = tag;
        meta.state = AllocationState::ACTIVE;
        meta.enable_cpu_backup = enable_cpu_backup;
        meta.cpu_backup = nullptr;
        meta.pauseable_chunks = false;
        meta.allocHandle = allocHandle;
        allocation_metadata_.emplace(*ptr, std::move(meta));
        if (!tag.empty()) tag_index_[tag].push_back(*ptr);
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = std::move(allocation_metadata_[ptr]);
        allocation_metadata_.erase(ptr);

        if (!metadata.tag.empty()) {
            auto idx_it = tag_index_.find(metadata.tag);
            if (idx_it != tag_index_.end()) {
                auto& ptrs = idx_it->second;
                ptrs.erase(std::remove(ptrs.begin(), ptrs.end(), ptr), ptrs.end());
                if (ptrs.empty()) tag_index_.erase(idx_it);
            }
        }
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    if (metadata.pauseable_chunks) {
        for (ChunkInfo& chunk : metadata.chunks) {
            if (chunk.state == AllocationState::ACTIVE) {
                CURESULT_CHECK(cuMemUnmap((CUdeviceptr)((uint8_t*)ptr + chunk.offset), chunk.size));
                CURESULT_CHECK(cuMemRelease(chunk.allocHandle));
            }
            if (chunk.cpu_backup != nullptr) {
                CUDA_ERROR_CHECK(cudaFreeHost(chunk.cpu_backup));
                chunk.cpu_backup = nullptr;
            }
        }
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free (chunked)"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " num_chunks=" << metadata.chunks.size() << " tag=" << metadata.tag
                  << std::endl;
#endif
    } else {
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

        if (nullptr != metadata.cpu_backup) {
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
                  << std::endl;
#endif
    }

#endif
    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag, bool save_chunk_state) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.pauseable_chunks) {
            if (save_chunk_state) {
                metadata.saved_chunk_states.resize(metadata.chunks.size());
                for (size_t i = 0; i < metadata.chunks.size(); i++) {
                    metadata.saved_chunk_states[i] =
                        (metadata.chunks[i].state == AllocationState::ACTIVE) ? 1 : 0;
                }
            }
            // Tolerant: pause any chunk still ACTIVE. Composes with any
            // prior pause_chunks() call — already-paused chunks are no-ops.
            for (ChunkInfo& chunk : metadata.chunks) {
                pause_chunk_locked(chunk, ptr, metadata.enable_cpu_backup);
            }
        } else {
            if (metadata.state != AllocationState::ACTIVE) {
                std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                          << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            if (metadata.enable_cpu_backup) {
                if (nullptr == metadata.cpu_backup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
                }
                SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
                CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
            }

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
            metadata.state = AllocationState::PAUSED;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " pauseable_chunks=" << metadata.pauseable_chunks;
        if (!metadata.pauseable_chunks) std::cout << " allocHandle=" << metadata.allocHandle;
        std::cout << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag, bool restore_chunk_state) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.pauseable_chunks) {
            if (restore_chunk_state && !metadata.saved_chunk_states.empty()) {
                // Restore only chunks that were ACTIVE before the save-pause.
                SIMPLE_CHECK(metadata.saved_chunk_states.size() == metadata.chunks.size(),
                    "saved chunk state size mismatch");
                for (size_t i = 0; i < metadata.chunks.size(); i++) {
                    if (metadata.saved_chunk_states[i]) {
                        resume_chunk_locked(metadata.chunks[i], ptr, metadata.device, metadata.enable_cpu_backup);
                    }
                }
                metadata.saved_chunk_states.clear();
            } else {
                // Tolerant: resume any chunk still PAUSED. Composes with any
                // prior resume_chunks() call — already-active chunks are no-ops.
                for (ChunkInfo& chunk : metadata.chunks) {
                    resume_chunk_locked(chunk, ptr, metadata.device, metadata.enable_cpu_backup);
                }
            }
        } else {
            if (metadata.state != AllocationState::PAUSED) {
                std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                          << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            CUmemGenericAllocationHandle newAllocHandle;
            CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

            CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
            CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

            if (metadata.enable_cpu_backup) {
                SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
                CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));
                CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
                metadata.cpu_backup = nullptr;
            }

            metadata.allocHandle = newAllocHandle;
            metadata.state = AllocationState::ACTIVE;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " pauseable_chunks=" << metadata.pauseable_chunks;
        if (!metadata.pauseable_chunks) std::cout << " allocHandle=" << metadata.allocHandle;
        std::cout << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        const size_t total_size = metadata.aligned_size;
#else
        const size_t total_size = metadata.size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;

#if !TMS_ROCM_LEGACY_CHUNKED
            if (metadata.pauseable_chunks) {
                // Fast path: query fits within a single chunk.
                for (const ChunkInfo& chunk : metadata.chunks) {
                    if (offset >= chunk.offset && offset + query_size <= chunk.offset + chunk.size) {
                        if (chunk.state == AllocationState::ACTIVE) {
                            return nullptr;
                        }
                        SIMPLE_CHECK(nullptr != chunk.cpu_backup,
                            "get_cpu_backup_pointer: found paused chunk but cpu_backup does not exist, do you forget to enable cpu backup");
                        return (uint8_t*) chunk.cpu_backup + (offset - chunk.offset);
                    }
                }
                // Query spans a chunk boundary. If all overlapping chunks are
                // active, the GPU data is live and no CPU backup is needed.
                // If any are paused, we can't return a contiguous CPU pointer.
                for (const ChunkInfo& chunk : metadata.chunks) {
                    bool overlaps = chunk.offset < offset + query_size
                                 && offset < chunk.offset + chunk.size;
                    if (overlaps && chunk.state == AllocationState::PAUSED) {
                        std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer: query spans chunk boundary "
                                  << "and at least one overlapping chunk is paused (cannot return contiguous CPU backup)"
                                  << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
                                  << std::endl;
                        exit(1);
                    }
                }
                return nullptr;  // all overlapping chunks are active
            }
#endif

            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            } else {
                SIMPLE_CHECK(nullptr != metadata.cpu_backup,
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist, do you forget to enable cpu backup");
                return (uint8_t*) metadata.cpu_backup + offset;
            }
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}

void TorchMemorySaver::pause_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(false, "pause_chunks is not supported with TMS_ROCM_LEGACY_CHUNKED");
#else
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    auto [ptr, metadata] = find_unique_by_tag_locked(allocation_metadata_, tag_index_, tag);
    if (ptr == nullptr) return;

    SIMPLE_CHECK(metadata->pauseable_chunks, "pause_chunks called on non-chunked allocation");

    // Validate all indices up-front so a bad index doesn't leave the
    // allocation partially paused.
    for (size_t i = 0; i < num_indices; i++) {
        SIMPLE_CHECK(chunk_indices[i] < metadata->chunks.size(),
            "chunk index " + std::to_string(chunk_indices[i]) + " out of range for allocation with "
            + std::to_string(metadata->chunks.size()) + " chunks");
    }

    for (size_t i = 0; i < num_indices; i++) {
        ChunkInfo& chunk = metadata->chunks[chunk_indices[i]];
        pause_chunk_locked(chunk, ptr, metadata->enable_cpu_backup);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause_chunks"
                  << " ptr=" << ptr << " chunk_idx=" << chunk_indices[i]
                  << " chunk.offset=" << chunk.offset << " chunk.size=" << chunk.size
                  << " tag=" << metadata->tag
                  << std::endl;
#endif
    }
#endif
}

void TorchMemorySaver::resume_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(false, "resume_chunks is not supported with TMS_ROCM_LEGACY_CHUNKED");
#else
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    auto [ptr, metadata] = find_unique_by_tag_locked(allocation_metadata_, tag_index_, tag);
    if (ptr == nullptr) return;

    SIMPLE_CHECK(metadata->pauseable_chunks, "resume_chunks called on non-chunked allocation");

    for (size_t i = 0; i < num_indices; i++) {
        SIMPLE_CHECK(chunk_indices[i] < metadata->chunks.size(),
            "chunk index " + std::to_string(chunk_indices[i]) + " out of range for allocation with "
            + std::to_string(metadata->chunks.size()) + " chunks");
    }

    for (size_t i = 0; i < num_indices; i++) {
        ChunkInfo& chunk = metadata->chunks[chunk_indices[i]];
        resume_chunk_locked(chunk, ptr, metadata->device, metadata->enable_cpu_backup);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume_chunks"
                  << " ptr=" << ptr << " chunk_idx=" << chunk_indices[i]
                  << " chunk.offset=" << chunk.offset << " chunk.size=" << chunk.size
                  << " tag=" << metadata->tag
                  << std::endl;
#endif
    }
#endif
}

size_t TorchMemorySaver::get_num_chunks(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(false, "get_num_chunks is not supported with TMS_ROCM_LEGACY_CHUNKED");
    return 0;
#else
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    auto [ptr, metadata] = find_unique_by_tag_locked(allocation_metadata_, tag_index_, tag);
    if (ptr == nullptr) return 0;

    return metadata->pauseable_chunks ? metadata->chunks.size() : 1;
#endif
}

bool TorchMemorySaver::get_chunk_states(const std::string& tag, uint8_t* out_active, size_t num_chunks) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(false, "get_chunk_states is not supported with TMS_ROCM_LEGACY_CHUNKED");
    return false;
#else
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    auto [ptr, metadata] = find_unique_by_tag_locked(allocation_metadata_, tag_index_, tag);
    if (ptr == nullptr) return false;

    if (!metadata->pauseable_chunks) {
        // Non-chunked allocation: report as a single logical chunk.
        SIMPLE_CHECK(num_chunks == 1,
            "get_chunk_states: buffer length " + std::to_string(num_chunks) +
            " but non-chunked allocation has 1 logical chunk");
        out_active[0] = (metadata->state == AllocationState::ACTIVE) ? 1 : 0;
        return true;
    }

    SIMPLE_CHECK(num_chunks == metadata->chunks.size(),
        "get_chunk_states: buffer length " + std::to_string(num_chunks) +
        " does not match number of chunks " + std::to_string(metadata->chunks.size()));

    for (size_t i = 0; i < num_chunks; i++) {
        out_active[i] = (metadata->chunks[i].state == AllocationState::ACTIVE) ? 1 : 0;
    }
    return true;
#endif
}

void TorchMemorySaver::get_chunk_layout(const std::string& tag, size_t* out_offsets, size_t* out_sizes, size_t num_chunks) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(false, "get_chunk_layout is not supported with TMS_ROCM_LEGACY_CHUNKED");
#else
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    auto [ptr, metadata] = find_unique_by_tag_locked(allocation_metadata_, tag_index_, tag);
    SIMPLE_CHECK(ptr != nullptr, "get_chunk_layout: no allocation matches tag=" + tag);

    if (!metadata->pauseable_chunks) {
        SIMPLE_CHECK(num_chunks == 1,
            "get_chunk_layout: buffer length " + std::to_string(num_chunks) +
            " but non-chunked allocation has 1 logical chunk");
        out_offsets[0] = 0;
        out_sizes[0] = metadata->size;
        return;
    }

    SIMPLE_CHECK(num_chunks == metadata->chunks.size(),
        "get_chunk_layout: buffer length " + std::to_string(num_chunks) +
        " does not match number of chunks " + std::to_string(metadata->chunks.size()));

    for (size_t i = 0; i < num_chunks; i++) {
        out_offsets[i] = metadata->chunks[i].offset;
        out_sizes[i] = metadata->chunks[i].size;
    }
#endif
}
