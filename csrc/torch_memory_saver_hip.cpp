#include <sys/types.h>

// Define HIP platform before including HIP headers
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <dlfcn.h>
#include <unordered_map>
#include <mutex>
#include <vector>

// #define TMS_DEBUG_LOG

#define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
#define MIN(a, b) (a < b ? a : b)

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
    static void cu_mem_create_and_map(hipDevice_t device, 
                                      size_t size, 
                                      void* d_mem,
                                      std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
                                      std::vector<size_t>& chunk_sizes) {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;

        // Get granularity
        size_t granularity;
        CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                               hipMemAllocationGranularityMinimum));

        // Make sure chunk size is aligned with hardware granularity
        size_t aligned_chunk_size = ((MEMCREATE_CHUNK_SIZE + granularity - 1) / granularity) * granularity;
        size_t num_chunks = (size + aligned_chunk_size - 1) / aligned_chunk_size;

        allocHandles.resize(num_chunks);
        chunk_sizes.resize(num_chunks);

        // Calculate chunk sizes
        for (size_t i = 0; i < num_chunks; ++i) {
            chunk_sizes[i] = MIN(size - i * aligned_chunk_size, aligned_chunk_size);
#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] chunk_sizes[" << i << "] = " << chunk_sizes[i] << std::endl;
#endif
        }

        // Create memory handles for each chunk
        for (size_t i = 0; i < num_chunks; ++i) {
            CURESULT_CHECK(hipMemCreate(&allocHandles[i], chunk_sizes[i], &prop, 0));
#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] allocHandles[" << i << "] = " << allocHandles[i] << std::endl;
#endif
        }

        // Map each chunk
        size_t allocated_size = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
            CURESULT_CHECK(hipMemMap((hipDeviceptr_t)map_addr, chunk_sizes[i], 0, allocHandles[i], 0));
            allocated_size += chunk_sizes[i];
#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] mapped chunk " << i << " at offset " << allocated_size - chunk_sizes[i] << std::endl;
#endif
        }

        // Set access permissions
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = device;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        CURESULT_CHECK(hipMemSetAccess(d_mem, size, &accessDesc, 1));
    }

    static void cu_mem_unmap_and_release(hipDevice_t device,
                                         size_t size,
                                         hipDeviceptr_t d_mem,
                                         const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
                                         const std::vector<size_t>& chunk_sizes) {
        // Unmap each chunk
        size_t allocated_size = 0;
        for (size_t i = 0; i < allocHandles.size(); ++i) {
            void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
            CURESULT_CHECK(hipMemUnmap((hipDeviceptr_t)map_addr, chunk_sizes[i]));
            allocated_size += chunk_sizes[i];
#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] unmapped chunk " << i << " at offset " << allocated_size - chunk_sizes[i] << std::endl;
#endif
        }

        // Release each handle
        for (size_t i = 0; i < allocHandles.size(); ++i) {
            CURESULT_CHECK(hipMemRelease(allocHandles[i]));
#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] released allocHandles[" << i << "]" << std::endl;
#endif
        }
    }

    static size_t cu_mem_get_granularity(hipDevice_t device) {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        
        size_t granularity;
        CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                               hipMemAllocationGranularityMinimum));
        return granularity;
    }
}

// ----------------------------------------------- primary class --------------------------------------------------

struct _AllocationMetadata {
    size_t size;
    size_t aligned_size;
    hipDevice_t device;
    std::vector<hipMemGenericAllocationHandle_t> allocHandles;
    std::vector<size_t> chunk_sizes;
};

class TorchMemorySaver {
public:
    TorchMemorySaver() {}

    hipError_t malloc(void **ptr, size_t size) {
        hipDevice_t device;
        CURESULT_CHECK(hipCtxGetDevice(&device));

        // Get granularity and calculate aligned size
        size_t granularity = CUDAUtils::cu_mem_get_granularity(device);
        size_t aligned_size = (size + granularity - 1) & ~(granularity - 1);

        // Reserve aligned memory address, rocm will check granularity
        CURESULT_CHECK(hipMemAddressReserve((hipDeviceptr_t *)ptr, aligned_size, granularity, 0, 0));

        // Create allocation metadata
        _AllocationMetadata metadata;
        metadata.size = size;
        metadata.aligned_size = aligned_size;
        metadata.device = device;

        // Create and map chunks
        CUDAUtils::cu_mem_create_and_map(device, size, (hipDeviceptr_t)*ptr, 
                                         metadata.allocHandles, metadata.chunk_sizes);

        size_t num_chunks = metadata.allocHandles.size();

        {
            const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(*ptr, std::move(metadata));
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " num_chunks=" << num_chunks
                  << std::endl;
#endif

        return hipSuccess;
    }

    hipError_t free(void *ptr) {
        _AllocationMetadata metadata;
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
                  << " num_chunks=" << metadata.allocHandles.size()
                  << std::endl;
#endif

        return hipSuccess;
    }

    void pause() {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            // Unmap and release chunks (but keep metadata for resume)
            CUDAUtils::cu_mem_unmap_and_release(metadata.device, metadata.size,
                                                (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size 
                      << " num_chunks=" << metadata.allocHandles.size()
                      << std::endl;
#endif
        }
    }

    void resume() {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            // Create new handles and map chunks
            CUDAUtils::cu_mem_create_and_map(metadata.device, metadata.size,
                                             (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size
                      << " num_chunks=" << metadata.allocHandles.size()
                      << std::endl;
#endif
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
