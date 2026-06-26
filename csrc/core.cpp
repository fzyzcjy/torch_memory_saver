#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

TorchMemorySaver::TorchMemorySaver()
    : disk_backend_(compute_disk_backup_dir_from_env(), compute_disk_chunk_bytes_from_env()) {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(
    void **ptr,
    CUdevice device,
    size_t raw_size,
    const std::string& tag,
    const bool enable_cpu_backup,
    const bool enable_disk_backup) {
    // Enforce here, not only in the Python layer: an assert is stripped under
    // python -O and bypassed by direct C-API / env-var use.
    SIMPLE_CHECK(!(enable_cpu_backup && enable_disk_backup),
                 "cpu_backup and disk_backup are mutually exclusive");
#if TMS_ROCM_LEGACY_CHUNKED
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on the ROCm 6.x legacy chunked path");
    return ROCmHIPImplementation::rocm_malloc(
        ptr,
        device,
        raw_size,
        tag,
        enable_cpu_backup,
        allocation_metadata_,
        allocator_metadata_mutex_);

#elif defined(USE_XPU)
    // Disk backup spills via cudaMallocHost/cudaMemcpy (csrc/disk_backend.cpp),
    // which have no Level Zero equivalent wired up here; reject rather than
    // silently ignore (the Python setter also raises). cpu_backup still works.
    SIMPLE_CHECK(!enable_disk_backup, "disk backup is not supported on Intel XPU");
    return XPUImplementation::xpu_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
    const size_t allocation_size = CUDAUtils::cu_mem_get_allocation_size(raw_size, device);
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + allocation_size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " allocation_size=" << allocation_size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, allocation_size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, allocation_size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, allocation_size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, allocation_size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{
                raw_size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr,
                enable_disk_backup, DiskBackupSlot{}, allocation_size, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " raw_size=" << raw_size
              << " allocation_size=" << allocation_size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_XPU)
    return XPUImplementation::xpu_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.allocation_size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

    if (metadata.enable_disk_backup) {
        disk_backend_.release(metadata.disk);
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
              << " metadata.allocation_size=" << metadata.allocation_size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

#if defined(USE_XPU)
uint64_t TorchMemorySaver::xpu_committed_bytes(int device_id) {
    return XPUImplementation::xpu_committed_bytes(
        device_id, allocation_metadata_, allocator_metadata_mutex_);
}
#endif

void TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_XPU)
    XPUImplementation::xpu_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

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

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.raw_size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.raw_size, cudaMemcpyDeviceToHost));
        } else if (metadata.enable_disk_backup) {
            disk_backend_.offload(ptr, metadata.raw_size, metadata.disk);
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.allocation_size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_XPU)
    XPUImplementation::xpu_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

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

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(
            &newAllocHandle, metadata.allocation_size, metadata.device));

        CURESULT_CHECK(cuMemMap(
            (CUdeviceptr) ptr, metadata.allocation_size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.allocation_size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.raw_size, cudaMemcpyHostToDevice));

            // TODO may provide a flag to choose whether to free immediately
            // (users may want to lazily free to reduce re-alloc time)
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
        } else if (metadata.enable_disk_backup) {
            disk_backend_.onload(ptr, metadata.raw_size, metadata.disk);
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.raw_size=" << metadata.raw_size
                  << " metadata.allocation_size=" << metadata.allocation_size
                  << " (old)metadata.allocHandle=" << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED || defined(USE_XPU)
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.raw_size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            // Disk-backed allocations have no CPU-resident copy; callers must use the resumed GPU tensor.
            if (metadata.enable_disk_backup) {
                return nullptr;
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
