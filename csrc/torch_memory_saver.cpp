#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <optional>
#include <thread>
#include <chrono>

// #define TMS_DEBUG_LOG

// ----------------------------------------------- copied code --------------------------------------------------

// Cannot use pytorch (libc10.so) since LD_PRELOAD happens earlier than `import torch`
// #include <ATen/cuda/Exceptions.h>

// torch Macros.h
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

// ----------------------------------------------- utils --------------------------------------------------

#define SIMPLE_CHECK(COND, MSG) \
  do { \
    if (!(COND)) { \
        std::cerr << "[torch_memory_saver.cpp] " << MSG \
                  << "  at " << __FILE__ << ":" << __LINE__ \
                  << " in function " << __func__ << std::endl; \
        exit(1); \
    } \
  } while (false)

// very naive
// TODO merge with above
#define CUDA_ERROR_CHECK(EXPR) \
  do { \
    cudaError __result = (EXPR); \
    if (__result != cudaSuccess) { \
        std::cerr << "[torch_memory_saver.cpp] cudaError error " \
            << " result=" << __result << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
            << std::endl; \
        exit(1); \
    } \
  } while (false)

// very naive
// TODO merge with above
#define CURESULT_CHECK(EXPR) \
  do { \
    CUresult __result = (EXPR); \
    if (__result != CUDA_SUCCESS) { \
        std::cerr << "[torch_memory_saver.cpp] CUresult error " \
            << " result=" << __result << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
            << std::endl; \
        exit(1); \
    } \
  } while (false)

namespace APIForwarder {
    static void *check_dlsym(void *value) {
        if (nullptr == value) {
            std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror() << std::endl;
            exit(1);
        }
        return value;
    }

    typedef cudaError_t (*CudaMallocFunc)(void **, size_t);

    typedef cudaError_t (*CudaFreeFunc)(void *);

    static CudaMallocFunc real_cudaMalloc = NULL;
    static CudaFreeFunc real_cudaFree = NULL;

    static cudaError_t call_real_cuda_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_cudaMalloc)) {
            real_cudaMalloc = (CudaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, "cudaMalloc"));
        }

        cudaError_t ret = real_cudaMalloc(ptr, size);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] cudaMalloc [MODE NORMAL]"
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

    static cudaError_t call_real_cuda_free(void *ptr) {
        if (C10_UNLIKELY(nullptr == real_cudaFree)) {
            real_cudaFree = (CudaFreeFunc) check_dlsym(dlsym(RTLD_NEXT, "cudaFree"));
        }

        cudaError_t ret = real_cudaFree(ptr);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] cudaFree [MODE NORMAL]"
                  << " ptr=" << ptr << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }
}

namespace CUDAUtils {
    static void cu_mem_create(CUmemGenericAllocationHandle *allocHandle, size_t size, CUdevice device) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        CURESULT_CHECK(cuMemCreate(allocHandle, size, &prop, 0));
    }

    static void cu_mem_set_access(void *ptr, size_t size, CUdevice device) {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CURESULT_CHECK(cuMemSetAccess((CUdeviceptr) ptr, size, &accessDesc, 1));
    }

    static size_t cuda_mem_get_info_free_mem() {
        size_t free_mem = 0;
        size_t total_mem = 0;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        return free_mem;
    }
}

// ----------------------------------------------- primary class --------------------------------------------------

struct _AllocationMetadata {
    size_t size;
    CUdevice device;
    CUmemGenericAllocationHandle allocHandle;
    bool enableCpuBackup;
    // TODO if this is costly, do not put it here, but make a separate map
    void* cpuBackup;
};

enum CopyDirection {
    DEVICE_TO_HOST,
    HOST_TO_DEVICE,
};

class TorchMemorySaver {
public:
    TorchMemorySaver() {}

    cudaError_t malloc(void **ptr, size_t size, bool enableCpuBackup) {
        CUdevice device;
        CURESULT_CHECK(cuCtxGetDevice(&device));

        CUmemGenericAllocationHandle allocHandle;
        CUDAUtils::cu_mem_create(&allocHandle, size, device);

        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
        CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
        CUDAUtils::cu_mem_set_access(*ptr, size, device);

        {
            const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, enableCpuBackup});
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " allocHandle=" << allocHandle
                  << std::endl;
#endif

        return cudaSuccess;
    }

    cudaError_t free(void *ptr) {
        _AllocationMetadata metadata;
        {
            const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
            SIMPLE_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
            metadata = allocation_metadata_[ptr];
            allocation_metadata_.erase(ptr);
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " metadata.allocHandle=" << metadata.allocHandle
                  << std::endl;
#endif

        return cudaSuccess;
    }

    void pause() {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata metadata = it->second;

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                      << metadata.allocHandle
                      << std::endl;
#endif
        }
    }

    void resume() {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            CUmemGenericAllocationHandle newAllocHandle;
            CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

            CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

            CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                      << metadata.allocHandle
                      << " (new)newAllocHandle=" << newAllocHandle
                      << std::endl;
#endif

            metadata.allocHandle = newAllocHandle;
        }
    }

    // TODO optimize later e.g. speedup
    void copy_between_device_host(CopyDirection direction, bool fuse_resume) {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        // TODO refactor
        int64_t free_mem = CUDAUtils::cuda_mem_get_info_free_mem();

        // TODO merge to below
        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata& metadata = it->second;

            if (fuse_resume) {
                // TODO refactor
                while (true) {
                    if (free_mem >= metadata.size + 3 * 1024 * 1024) { break; }

                    int64_t free_mem = CUDAUtils::cuda_mem_get_info_free_mem();
                    if (free_mem >= metadata.size + 3 * 1024 * 1024) { break; }

                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                CUmemGenericAllocationHandle newAllocHandle;
                CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);
                CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
                CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);
                metadata.allocHandle = newAllocHandle;

                free_mem -= metadata.size;
            }
        }

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata& metadata = it->second;

            if (metadata.enableCpuBackup) {
                switch(direction) {
                    case DEVICE_TO_HOST:
                        if (metadata.cpuBackup == nullptr) {
                            CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpuBackup, metadata.size));
                        }
                        CUDA_ERROR_CHECK(cudaMemcpyAsync(metadata.cpuBackup, ptr, metadata.size, cudaMemcpyDeviceToHost));
                        break;

                    case HOST_TO_DEVICE:
                        CUDA_ERROR_CHECK(cudaMemcpyAsync(ptr, metadata.cpuBackup, metadata.size, cudaMemcpyHostToDevice));
                        // TODO free host memory later
                        break;
                }
            }

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.gpu2cpu"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                      << metadata.allocHandle
                      << std::endl;
#endif
        }

        if (direction == DEVICE_TO_HOST) {
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        }
    }

    static TorchMemorySaver &instance() {
        static TorchMemorySaver instance;
        return instance;
    }

private:
    // Similar to torch's CUDACachingAllocator and CUDAPluggableAllocator
    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void *, _AllocationMetadata> allocation_metadata_;
};

namespace RegionManager {
    static thread_local std::optional<bool> is_interesting_region_;

    bool get_init_is_interesting_region_from_env() {
        const char* env = std::getenv("TMS_INIT_IS_INTERESTING_REGION");
        if (env != nullptr) {
            std::string val(env);
            return val == "1" || val == "true" || val == "TRUE" || val == "yes" || val == "YES";
        } else {
            return false;
        }
    }

    bool is_interesting_region() {
        if (!is_interesting_region_.has_value()) {
            is_interesting_region_ = get_init_is_interesting_region_from_env();
        }
        return is_interesting_region_.value();
    }

    void enter() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_enter" << std::endl;
#endif
        SIMPLE_CHECK(is_interesting_region() == false, "Bad is_interesting_region_ state");
        is_interesting_region_ = true;
    }

    void leave() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_leave" << std::endl;
#endif
        SIMPLE_CHECK(is_interesting_region() == true, "Bad is_interesting_region_ state");
        is_interesting_region_ = false;
    }
}

namespace EnableCpuBackupRegionManager {
    static thread_local bool enable_cpu_backup_ = true;
}

// ------------------------------------------------- entrypoints ------------------------------------------------

cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(ptr, size, EnableCpuBackupRegionManager::enable_cpu_backup_);
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    if (RegionManager::is_interesting_region()) {
        return TorchMemorySaver::instance().free(ptr);
    } else {
        return APIForwarder::call_real_cuda_free(ptr);
    }
}

extern "C" {
void tms_region_enter() {
    RegionManager::enter();
}

void tms_region_leave() {
    RegionManager::leave();
}

void tms_pause() {
    TorchMemorySaver::instance().pause();
}

void tms_resume() {
    TorchMemorySaver::instance().resume();
}

//void tms_copy_host_to_device() {
//    TorchMemorySaver::instance().copy_between_device_host(CopyDirection::HOST_TO_DEVICE);
//}

void tms_resume_and_copy_host_to_device() {
    TorchMemorySaver::instance().copy_between_device_host(CopyDirection::HOST_TO_DEVICE, true);
}

void tms_copy_device_to_host() {
    TorchMemorySaver::instance().copy_between_device_host(CopyDirection::DEVICE_TO_HOST, false);
}

void tms_enable_cpu_backup() {
    SIMPLE_CHECK(EnableCpuBackupRegionManager::enable_cpu_backup_ == false, "enable_cpu_backup_ bad");
    EnableCpuBackupRegionManager::enable_cpu_backup_ = true;
}

void tms_disable_cpu_backup() {
    SIMPLE_CHECK(EnableCpuBackupRegionManager::enable_cpu_backup_ == true, "enable_cpu_backup_ bad");
    EnableCpuBackupRegionManager::enable_cpu_backup_ = false;
}

}
