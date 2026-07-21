#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

#include <algorithm>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

TorchMemorySaver::TorchMemorySaver() {
    const char* dir = std::getenv("TMS_DISK_BACKUP_DIR");
    disk_backup_dir_ = (dir != nullptr) ? std::string(dir) : std::string("/tmp");

    const char* chunk_mb = std::getenv("TMS_DISK_BACKUP_CHUNK_MB");
    size_t mb = (chunk_mb != nullptr) ? std::strtoull(chunk_mb, nullptr, 10) : 256;
    if (mb == 0) mb = 256;
    disk_chunk_bytes_ = mb * 1024ull * 1024ull;
}

void TorchMemorySaver::ensure_disk_staging_() {
    if (disk_staging_ == nullptr) {
        CUDA_ERROR_CHECK(cudaMallocHost(&disk_staging_, disk_chunk_bytes_));
        SIMPLE_CHECK(disk_staging_ != nullptr, "failed to allocate pinned disk staging buffer");
    }
}

void TorchMemorySaver::disk_offload_(void* ptr, AllocationMetadata& metadata) {
    ensure_disk_staging_();

    if (metadata.disk_fd < 0) {
        metadata.disk_path = disk_backup_dir_ + "/tms_" + std::to_string((long)getpid())
            + "_" + std::to_string(disk_backup_counter_++) + ".bin";
        metadata.disk_fd = open(metadata.disk_path.c_str(), O_RDWR | O_CREAT, 0600);
        SIMPLE_CHECK(metadata.disk_fd >= 0, "failed to open disk backup file");
        int fret = posix_fallocate(metadata.disk_fd, 0, (off_t) metadata.size);
        SIMPLE_CHECK(fret == 0, "posix_fallocate on disk backup file failed");
    }

    size_t offset = 0;
    while (offset < metadata.size) {
        size_t n = std::min(disk_chunk_bytes_, metadata.size - offset);
        CUDA_ERROR_CHECK(cudaMemcpy(disk_staging_, (uint8_t*) ptr + offset, n, cudaMemcpyDeviceToHost));
        size_t written = 0;
        while (written < n) {
            ssize_t w = pwrite(metadata.disk_fd, (uint8_t*) disk_staging_ + written, n - written, (off_t)(offset + written));
            SIMPLE_CHECK(w > 0, "pwrite to disk backup file failed");
            written += (size_t) w;
        }
        offset += n;
    }
}

void TorchMemorySaver::disk_reload_(void* ptr, AllocationMetadata& metadata) {
    ensure_disk_staging_();
    SIMPLE_CHECK(metadata.disk_fd >= 0, "disk backup fd invalid on resume (was it ever paused?)");

    size_t offset = 0;
    while (offset < metadata.size) {
        size_t n = std::min(disk_chunk_bytes_, metadata.size - offset);
        size_t got = 0;
        while (got < n) {
            ssize_t r = pread(metadata.disk_fd, (uint8_t*) disk_staging_ + got, n - got, (off_t)(offset + got));
            SIMPLE_CHECK(r > 0, "pread from disk backup file failed");
            got += (size_t) r;
        }
        CUDA_ERROR_CHECK(cudaMemcpy((uint8_t*) ptr + offset, disk_staging_, n, cudaMemcpyHostToDevice));
        offset += n;
    }
    // fd stays open for the next pause.
}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup, const bool enable_disk_backup) {
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
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr,
                               enable_disk_backup, -1, std::string(), allocHandle}
        );
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

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

    if (metadata.enable_disk_backup && metadata.disk_fd >= 0) {
        close(metadata.disk_fd);
        unlink(metadata.disk_path.c_str());
        metadata.disk_fd = -1;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

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

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost));
        } else if (metadata.enable_disk_backup) {
            disk_offload_(ptr, metadata);
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

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice));

            // TODO may provide a flag to choose whether to free immediately
            // (users may want to lazily free to reduce re-alloc time)
            CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
            metadata.cpu_backup = nullptr;
        } else if (metadata.enable_disk_backup) {
            disk_reload_(ptr, metadata);
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
        size_t total_size = metadata.size;
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
