#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup, size_t num_chunks) {
#if TMS_ROCM_LEGACY_CHUNKED
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

    // Chunked allocation: multiple independent physical allocations mapped to one contiguous VA range
    if (num_chunks > 1) {
        // Query alignment granularity (platform-agnostic)
        size_t granularity = CUDAUtils::cu_mem_get_granularity(device);

        // Each chunk must be aligned to granularity
        size_t raw_chunk_size = (size + num_chunks - 1) / num_chunks;
        size_t aligned_chunk_size = ((raw_chunk_size + granularity - 1) / granularity) * granularity;
        size_t aligned_total = aligned_chunk_size * num_chunks;

        // Reserve one contiguous VA range
        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, aligned_total, granularity, 0, 0));

        // Create + map each chunk independently
        std::vector<ChunkInfo> chunks(num_chunks);
        for (size_t i = 0; i < num_chunks; i++) {
            chunks[i].offset = i * aligned_chunk_size;
            chunks[i].size = aligned_chunk_size;
            chunks[i].state = AllocationState::ACTIVE;
            chunks[i].cpu_backup = nullptr;

            cudaError_t ret = CUDAUtils::cu_mem_create(&chunks[i].allocHandle, aligned_chunk_size, device);
            if (ret != cudaSuccess) {
                // Cleanup already-created chunks on failure
                for (size_t j = 0; j < i; j++) {
                    CURESULT_CHECK(cuMemUnmap((CUdeviceptr)((uint8_t*)*ptr + chunks[j].offset), chunks[j].size));
                    CURESULT_CHECK(cuMemRelease(chunks[j].allocHandle));
                }
                CURESULT_CHECK(cuMemAddressFree((CUdeviceptr)*ptr, aligned_total));
                return ret;
            }

            CURESULT_CHECK(cuMemMap((CUdeviceptr)((uint8_t*)*ptr + chunks[i].offset),
                                    aligned_chunk_size, 0, chunks[i].allocHandle, 0));
            CUDAUtils::cu_mem_set_access((uint8_t*)*ptr + chunks[i].offset, aligned_chunk_size, device);
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
            meta.num_chunks = num_chunks;
            meta.aligned_total_size = aligned_total;
            meta.chunks = std::move(chunks);
            meta.allocHandle = 0;
            allocation_metadata_.emplace(*ptr, std::move(meta));
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc (chunked)"
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " num_chunks=" << num_chunks << " aligned_chunk_size=" << aligned_chunk_size
                  << " aligned_total=" << aligned_total << " tag=" << tag
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
        meta.num_chunks = 1;
        meta.aligned_total_size = size;
        meta.allocHandle = allocHandle;
        allocation_metadata_.emplace(*ptr, std::move(meta));
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
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    if (metadata.num_chunks > 1) {
        // Chunked allocation: unmap and release each chunk
        for (size_t i = 0; i < metadata.chunks.size(); i++) {
            ChunkInfo& chunk = metadata.chunks[i];
            if (chunk.state == AllocationState::ACTIVE) {
                CURESULT_CHECK(cuMemUnmap((CUdeviceptr)((uint8_t*)ptr + chunk.offset), chunk.size));
                CURESULT_CHECK(cuMemRelease(chunk.allocHandle));
            }
            if (chunk.cpu_backup != nullptr) {
                CUDA_ERROR_CHECK(cudaFreeHost(chunk.cpu_backup));
                chunk.cpu_backup = nullptr;
            }
        }
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.aligned_total_size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free (chunked)"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " num_chunks=" << metadata.num_chunks << " tag=" << metadata.tag
                  << std::endl;
#endif
    } else {
        // Non-chunked allocation (original path)
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

void TorchMemorySaver::pause(const std::string& tag) {
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

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.num_chunks > 1) {
            // Chunked allocation: pause each active chunk
            for (size_t i = 0; i < metadata.chunks.size(); i++) {
                ChunkInfo& chunk = metadata.chunks[i];
                if (chunk.state != AllocationState::ACTIVE) continue;

                void* chunk_ptr = (uint8_t*)ptr + chunk.offset;

                if (metadata.enable_cpu_backup) {
                    if (nullptr == chunk.cpu_backup) {
                        CUDA_ERROR_CHECK(cudaMallocHost(&chunk.cpu_backup, chunk.size));
                    }
                    CUDA_ERROR_CHECK(cudaMemcpy(chunk.cpu_backup, chunk_ptr, chunk.size, cudaMemcpyDeviceToHost));
                }

                CURESULT_CHECK(cuMemUnmap((CUdeviceptr) chunk_ptr, chunk.size));
                CURESULT_CHECK(cuMemRelease(chunk.allocHandle));
                chunk.state = AllocationState::PAUSED;
            }
        } else {
            // Non-chunked allocation (original path)
            if (metadata.enable_cpu_backup) {
                if (nullptr == metadata.cpu_backup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
                }
                SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
                CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
            }

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        }

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " num_chunks=" << metadata.num_chunks
                  << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
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

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.num_chunks > 1) {
            // Chunked allocation: resume each paused chunk
            for (size_t i = 0; i < metadata.chunks.size(); i++) {
                ChunkInfo& chunk = metadata.chunks[i];
                if (chunk.state != AllocationState::PAUSED) continue;

                void* chunk_ptr = (uint8_t*)ptr + chunk.offset;

                CUmemGenericAllocationHandle newAllocHandle;
                CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, chunk.size, metadata.device));
                CURESULT_CHECK(cuMemMap((CUdeviceptr) chunk_ptr, chunk.size, 0, newAllocHandle, 0));
                CUDAUtils::cu_mem_set_access(chunk_ptr, chunk.size, metadata.device);

                if (metadata.enable_cpu_backup) {
                    SIMPLE_CHECK(chunk.cpu_backup != nullptr, "cpu_backup should not be nullptr (chunk)");
                    CUDA_ERROR_CHECK(cudaMemcpy(chunk_ptr, chunk.cpu_backup, chunk.size, cudaMemcpyHostToDevice));
                    CUDA_ERROR_CHECK(cudaFreeHost(chunk.cpu_backup));
                    chunk.cpu_backup = nullptr;
                }

                chunk.state = AllocationState::ACTIVE;
                chunk.allocHandle = newAllocHandle;
            }
        } else {
            // Non-chunked allocation (original path)
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
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " num_chunks=" << metadata.num_chunks
                  << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
    }
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = (metadata.num_chunks > 1) ? metadata.aligned_total_size : metadata.size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;

            if (metadata.num_chunks > 1) {
                // Chunked allocation: find which chunk contains this range
                for (size_t i = 0; i < metadata.chunks.size(); i++) {
                    const ChunkInfo& chunk = metadata.chunks[i];
                    if (offset >= chunk.offset && offset + query_size <= chunk.offset + chunk.size) {
                        if (chunk.state == AllocationState::ACTIVE) {
                            return nullptr;
                        } else {
                            SIMPLE_CHECK(nullptr != chunk.cpu_backup,
                                "get_cpu_backup_pointer: found paused chunk but cpu_backup does not exist, do you forget to enable cpu backup");
                            size_t offset_within_chunk = offset - chunk.offset;
                            return (uint8_t*) chunk.cpu_backup + offset_within_chunk;
                        }
                    }
                }
                std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer: query spans chunk boundary"
                          << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
                          << std::endl;
                exit(1);
            }

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

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        SIMPLE_CHECK(metadata.num_chunks > 1, "pause_chunks called on non-chunked allocation");

        for (size_t idx_i = 0; idx_i < num_indices; idx_i++) {
            size_t chunk_idx = chunk_indices[idx_i];
            SIMPLE_CHECK(chunk_idx < metadata.chunks.size(), "chunk index out of range");

            ChunkInfo& chunk = metadata.chunks[chunk_idx];
            if (chunk.state != AllocationState::ACTIVE) {
                std::cerr << "[torch_memory_saver.cpp] Cannot pause chunk that is not active."
                          << " tag=" << metadata.tag << " chunk_idx=" << chunk_idx
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            void* chunk_ptr = (uint8_t*)ptr + chunk.offset;

            if (metadata.enable_cpu_backup) {
                if (nullptr == chunk.cpu_backup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&chunk.cpu_backup, chunk.size));
                }
                CUDA_ERROR_CHECK(cudaMemcpy(chunk.cpu_backup, chunk_ptr, chunk.size, cudaMemcpyDeviceToHost));
            }

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) chunk_ptr, chunk.size));
            CURESULT_CHECK(cuMemRelease(chunk.allocHandle));
            chunk.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause_chunks"
                      << " ptr=" << ptr << " chunk_idx=" << chunk_idx
                      << " chunk.offset=" << chunk.offset << " chunk.size=" << chunk.size
                      << " tag=" << metadata.tag
                      << std::endl;
#endif
        }
    }
#endif
}

void TorchMemorySaver::resume_chunks(const std::string& tag, const size_t* chunk_indices, size_t num_indices) {
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(false, "resume_chunks is not supported with TMS_ROCM_LEGACY_CHUNKED");
#else
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        SIMPLE_CHECK(metadata.num_chunks > 1, "resume_chunks called on non-chunked allocation");

        for (size_t idx_i = 0; idx_i < num_indices; idx_i++) {
            size_t chunk_idx = chunk_indices[idx_i];
            SIMPLE_CHECK(chunk_idx < metadata.chunks.size(), "chunk index out of range");

            ChunkInfo& chunk = metadata.chunks[chunk_idx];
            if (chunk.state != AllocationState::PAUSED) {
                std::cerr << "[torch_memory_saver.cpp] Cannot resume chunk that is not paused."
                          << " tag=" << metadata.tag << " chunk_idx=" << chunk_idx
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            void* chunk_ptr = (uint8_t*)ptr + chunk.offset;

            CUmemGenericAllocationHandle newAllocHandle;
            CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, chunk.size, metadata.device));
            CURESULT_CHECK(cuMemMap((CUdeviceptr) chunk_ptr, chunk.size, 0, newAllocHandle, 0));
            CUDAUtils::cu_mem_set_access(chunk_ptr, chunk.size, metadata.device);

            if (metadata.enable_cpu_backup) {
                SIMPLE_CHECK(chunk.cpu_backup != nullptr, "cpu_backup should not be nullptr (chunk)");
                CUDA_ERROR_CHECK(cudaMemcpy(chunk_ptr, chunk.cpu_backup, chunk.size, cudaMemcpyHostToDevice));
                CUDA_ERROR_CHECK(cudaFreeHost(chunk.cpu_backup));
                chunk.cpu_backup = nullptr;
            }

            chunk.state = AllocationState::ACTIVE;
            chunk.allocHandle = newAllocHandle;

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume_chunks"
                      << " ptr=" << ptr << " chunk_idx=" << chunk_idx
                      << " chunk.offset=" << chunk.offset << " chunk.size=" << chunk.size
                      << " tag=" << metadata.tag
                      << std::endl;
#endif
        }
    }
#endif
}
