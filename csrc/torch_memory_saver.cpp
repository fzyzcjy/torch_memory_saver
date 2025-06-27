#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>
#include <string>

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
}

// ----------------------------------------------- state management --------------------------------------------------

enum class AllocationState {
    ACTIVE,
    PAUSED,
    FREED
};

struct _AllocationMetadata {
    size_t size;
    CUdevice device;
    CUmemGenericAllocationHandle allocHandle;
    std::string tag;
    AllocationState state;
};

// ----------------------------------------------- primary class --------------------------------------------------

class TorchMemorySaver {
public:
    TorchMemorySaver() {}

    cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
        CUdevice device;
        CURESULT_CHECK(cuCtxGetDevice(&device));

        CUmemGenericAllocationHandle allocHandle;
        CUDAUtils::cu_mem_create(&allocHandle, size, device);
        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
        CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
        CUDAUtils::cu_mem_set_access(*ptr, size, device);

        {
            const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag, AllocationState::ACTIVE});
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " allocHandle=" << allocHandle << " tag=" << tag
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

            if (metadata.state == AllocationState::FREED) {
                std::cerr << "[torch_memory_saver.cpp] ERROR: Trying to free already freed pointer" << std::endl;
                exit(1);
            }

            allocation_metadata_.erase(ptr);
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag
                  << std::endl;
#endif

        return cudaSuccess;
    }

    void pause(const std::string& tag) {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        bool found_valid_allocation = false;
        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            if (!tag.empty() && metadata.tag != tag) {
                continue;
            }

            if (metadata.state == AllocationState::PAUSED) {
                std::cerr << "[torch_memory_saver.cpp] ERROR: Trying to pause already paused allocation"
                          << " ptr=" << ptr << " tag=" << metadata.tag << std::endl;
                exit(1);
            }

            if (metadata.state == AllocationState::FREED) {
                std::cerr << "[torch_memory_saver.cpp] ERROR: Trying to pause freed allocation"
                          << " ptr=" << ptr << " tag=" << metadata.tag << std::endl;
                exit(1);
            }

            found_valid_allocation = true;

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
            metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                      << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                      << std::endl;
#endif
        }

        if (!found_valid_allocation) {
            std::string error_msg = "No active allocations found";
            if (!tag.empty()) {
                error_msg += " for tag: " + tag;
            }
            error_msg += ". Cannot pause freed or non-existent allocations.";
            std::cerr << "[torch_memory_saver.cpp] ERROR: " << error_msg << std::endl;
            exit(1);
        }
    }

    void resume(const std::string& tag) {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        bool found_paused_allocation = false;
        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            if (!tag.empty() && metadata.tag != tag) {
                continue;
            }

            if (metadata.state == AllocationState::FREED) {
                std::cerr << "[torch_memory_saver.cpp] ERROR: Trying to resume freed allocation"
                          << " ptr=" << ptr << " tag=" << metadata.tag << std::endl;
                exit(1);
            }

            if (metadata.state != AllocationState::PAUSED) {
                continue;
            }

            found_paused_allocation = true;

            CUmemGenericAllocationHandle newAllocHandle;
            CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

            CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

            CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                      << metadata.allocHandle
                      << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                      << std::endl;
#endif

            metadata.allocHandle = newAllocHandle;
            metadata.state = AllocationState::ACTIVE;
        }

        if (!found_paused_allocation) {
            std::string error_msg = "No paused allocations found";
            if (!tag.empty()) {
                error_msg += " for tag: " + tag;
            }
            error_msg += ". Cannot resume unpaused or freed allocations.";
            std::cerr << "[torch_memory_saver.cpp] ERROR: " << error_msg << std::endl;
            exit(1);
        }
    }

    static TorchMemorySaver &instance() {
        static TorchMemorySaver instance;
        return instance;
    }


private:
    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void *, _AllocationMetadata> allocation_metadata_;
};


// ----------------------------------------------- region manager --------------------------------------------------

namespace RegionManager {
    static thread_local bool is_interesting_region_ = false;
    static thread_local std::string current_tag_ = "default";

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

    void set_current_tag(const std::string& tag) {
        current_tag_ = tag;
    }

    const std::string& get_current_tag() {
        return current_tag_;
    }
}

// ------------------------------------------------- entrypoints ------------------------------------------------

cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(ptr, size, RegionManager::get_current_tag());
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

void tms_set_current_tag(const char* tag) {
    if (tag == nullptr) {
        std::cerr << "[torch_memory_saver.cpp] FATAL: NULL tag passed to tms_set_current_tag" << std::endl;
        exit(1);
    }
    RegionManager::set_current_tag(std::string(tag));
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}
}