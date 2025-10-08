#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
#if defined(USE_ROCM)
    // hipDevice_t device;
    CURESULT_CHECK(hipCtxGetDevice(&device));

    // // Get granularity and calculate aligned size
    // size_t granularity = CUDAUtils::cu_mem_get_granularity(device);
    // size_t aligned_size = (size + granularity - 1) & ~(granularity - 1);

    // //// Reserve aligned memory address, rocm will check granularity
    // CURESULT_CHECK(hipMemAddressReserve((hipDeviceptr_t *)ptr, aligned_size, granularity, 0, 0));

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    prop.allocFlags.compressionType = 0x0;

    size_t granularity;
    CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                            hipMemAllocationGranularityMinimum));
    size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;
    aligned_size = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE * MEMCREATE_CHUNK_SIZE;

    assert(MEMCREATE_CHUNK_SIZE % granularity == 0);
    assert(aligned_size % MEMCREATE_CHUNK_SIZE == 0);
    assert(aligned_size % granularity == 0);


    // Create allocation metadata
    AllocationMetadata metadata;
    metadata.size = size;
    metadata.aligned_size = aligned_size;
    metadata.device = device;
    //// Not sure (Check these parameters)
    metadata.tag = tag;
    metadata.enable_cpu_backup = enable_cpu_backup;
    metadata.cpu_backup = nullptr;
    ////

    // Get global device ID using our utility function
    int global_device_id = DeviceUtils::get_global_device_id(device);

    // rewrite numa node 
    uint64_t node_id = 0;
    if (global_device_id > 3) {
        node_id = 1;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " granularity=" << granularity
              << " aligned_size=" << aligned_size
              << " node_id=" << node_id
              << " device=" << device
              << " global_device_id=" << global_device_id
              << std::endl;
#endif

    hipDeviceptr_t d_mem;
    // Reserve aligned memory address, rocm will check granularity
    CURESULT_CHECK(hipMemAddressReserve(&d_mem, aligned_size, granularity, 0, node_id));
    *ptr = (void*)d_mem;

    // Create and map chunks
    // CUDAUtils::cu_mem_create_and_map(device, size, (hipDeviceptr_t)*ptr, 
    CUDAUtils::cu_mem_create_and_map(device, aligned_size, (hipDeviceptr_t)*ptr, 
                                    metadata.allocHandles, metadata.chunk_sizes);
    size_t num_chunks = metadata.allocHandles.size();
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(*ptr, std::move(metadata));
    }
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " metadata.aligned_size=" << metadata.aligned_size
              << " num_chunks=" << num_chunks
              << std::endl;
#endif

#elif defined(USE_CUDA)
    CUmemGenericAllocationHandle allocHandle;
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if defined(USE_ROCM)
    AllocationMetadata metadata;
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        SIMPLE_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
        metadata = std::move(allocation_metadata_[ptr]);
        allocation_metadata_.erase(ptr);
    }

    // Unmap and release chunks
    CUDAUtils::cu_mem_unmap_and_release(metadata.device, metadata.size, 
                                        (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);

    // Free the reserved address using stored aligned_size
    CURESULT_CHECK(hipMemAddressFree((hipDeviceptr_t)ptr, metadata.aligned_size));

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.aligned_size=" << metadata.aligned_size
              << " num_chunks=" << metadata.allocHandles.size()
              << std::endl;
#endif
#elif defined(USE_CUDA)
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

#if defined(USE_ROCM)
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }
        // Copy CUDA's code supporting cpu_backup to here
        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(hipMallocHost(&metadata.cpu_backup, metadata.aligned_size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.aligned_size, hipMemcpyDeviceToHost));
        }
        //

        // Unmap and release chunks (but keep metadata for resume)
        // CUDAUtils::cu_mem_unmap_and_release(metadata.device, metadata.size,
        CUDAUtils::cu_mem_unmap_and_release(metadata.device, metadata.aligned_size,
                                            (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);

        #ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                    << " ptr=" << ptr << " metadata.size=" << metadata.size 
                    << " metadata.aligned_size=" << metadata.aligned_size
                    << " num_chunks=" << metadata.allocHandles.size()
                    << std::endl;
        #endif
    }
#elif defined(USE_CUDA)
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

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

#if defined(USE_ROCM)
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        // Create new handles and map chunks
        // CUDAUtils::cu_mem_create_and_map(metadata.device, metadata.size,
        CUDAUtils::cu_mem_create_and_map(metadata.device, metadata.aligned_size,
                                        (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);

        // Restore from CPU backup if enabled
        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.aligned_size, hipMemcpyHostToDevice));
            // maybe we can free host memory if needed (currently keep it there to reduce re-alloc time)
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                << " ptr=" << ptr << " metadata.size=" << metadata.size
                << " metadata.aligned_size=" << metadata.aligned_size
                << " num_chunks=" << metadata.allocHandles.size()
                << std::endl;
#endif
    }

#elif defined(USE_CUDA)
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

        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));
            // maybe we can free host memory if needed (currently keep it there to reduce re-alloc time)
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                  << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}
