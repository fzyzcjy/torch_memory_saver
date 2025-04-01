#include <sys/types.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>

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
        std::cerr << "[torch_memory_saver.cpp] " << MSG << std::endl; \
        exit(1); \
    } \
  } while (false)

// very naive
#define CURESULT_CHECK(EXPR) \
  do { \
    hipError_t __result = (EXPR); \
    if (__result != hipSuccess) { \
        std::cerr << "[torch_memory_saver.cpp] hipError_t error " \
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

    typedef hipError_t (*CudaMallocFunc)(void **, size_t);

    typedef hipError_t (*CudaFreeFunc)(void *);

    static CudaMallocFunc real_cudaMalloc = NULL;
    static CudaFreeFunc real_cudaFree = NULL;

    static hipError_t call_real_cuda_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_cudaMalloc)) {
            real_cudaMalloc = (CudaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, "hipMalloc"));
        }

        hipError_t ret = real_cudaMalloc(ptr, size);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] hipMalloc [MODE NORMAL]"
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

    static hipError_t call_real_cuda_free(void *ptr) {
        if (C10_UNLIKELY(nullptr == real_cudaFree)) {
            real_cudaFree = (CudaFreeFunc) check_dlsym(dlsym(RTLD_NEXT, "hipFree"));
        }

        hipError_t ret = real_cudaFree(ptr);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] hipFree [MODE NORMAL]"
                  << " ptr=" << ptr << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }
}

namespace CUDAUtils {
    static void cu_mem_create(hipMemGenericAllocationHandle_t *allocHandle, size_t size, hipDevice_t device) {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        CURESULT_CHECK(hipMemCreate(allocHandle, size, &prop, 0));
    }

    static void cu_mem_set_access(void *ptr, size_t size, hipDevice_t device) {
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = device;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        CURESULT_CHECK(hipMemSetAccess((hipDeviceptr_t) ptr, size, &accessDesc, 1));
    }
}

// ----------------------------------------------- primary class --------------------------------------------------

struct _AllocationMetadata {
    size_t size;
    hipDevice_t device;
    hipMemGenericAllocationHandle_t allocHandle;
};

class TorchMemorySaver {
public:
    TorchMemorySaver() {}

    hipError_t malloc(void **ptr, size_t size) {
        hipDevice_t device;
        CURESULT_CHECK(hipCtxGetDevice(&device));

        hipMemGenericAllocationHandle_t allocHandle;
        CUDAUtils::cu_mem_create(&allocHandle, size, device);

        CURESULT_CHECK(hipMemAddressReserve((hipDeviceptr_t *) ptr, size, 0, 0, 0));
        CURESULT_CHECK(hipMemMap((hipDeviceptr_t) * ptr, size, 0, allocHandle, 0));
        CUDAUtils::cu_mem_set_access(*ptr, size, device);

        {
            const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle});
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " allocHandle=" << allocHandle
                  << std::endl;
#endif

        return hipSuccess;
    }

    hipError_t free(void *ptr) {
        _AllocationMetadata metadata;
        {
            const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
            SIMPLE_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
            metadata = allocation_metadata_[ptr];
            allocation_metadata_.erase(ptr);
        }

        CURESULT_CHECK(hipMemUnmap((hipDeviceptr_t) ptr, metadata.size));
        CURESULT_CHECK(hipMemRelease(metadata.allocHandle));
        CURESULT_CHECK(hipMemAddressFree((hipDeviceptr_t) ptr, metadata.size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " metadata.allocHandle=" << metadata.allocHandle
                  << std::endl;
#endif

        return hipSuccess;
    }

    void pause() {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata metadata = it->second;

            CURESULT_CHECK(hipMemUnmap((hipDeviceptr_t) ptr, metadata.size));
            CURESULT_CHECK(hipMemRelease(metadata.allocHandle));

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

            hipMemGenericAllocationHandle_t newAllocHandle;
            CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

            CURESULT_CHECK(hipMemMap((hipDeviceptr_t) ptr, metadata.size, 0, newAllocHandle, 0));

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
    static thread_local bool is_interesting_region_ = false;

    void enter() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_enter" << std::endl;
#endif
        is_interesting_region_ = true;
    }

    void leave() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_leave" << std::endl;
#endif
        is_interesting_region_ = false;
    }

    bool is_interesting_region() {
        return is_interesting_region_;
    }
}

// ------------------------------------------------- entrypoints ------------------------------------------------

hipError_t hipMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(ptr, size);
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

hipError_t hipFree(void *ptr) {
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
}
