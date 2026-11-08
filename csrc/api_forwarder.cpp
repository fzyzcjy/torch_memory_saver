#include <iostream>
#include "api_forwarder.h"
#include "utils.h"
#include "macro.h"

namespace APIForwarder {
    using CudaMallocFunc = cudaError_t (*)(void**, size_t);
    using CudaFreeFunc = cudaError_t (*)(void*);

#if defined(USE_ROCM)
    static constexpr const char* MALLOC_NAME = "hipMalloc";
    static constexpr const char* FREE_NAME = "hipFree";
#elif defined(USE_MUSA)
    static constexpr const char* MALLOC_NAME = "musaMalloc";
    static constexpr const char* FREE_NAME = "musaFree";
#else
    static constexpr const char* MALLOC_NAME = "cudaMalloc";
    static constexpr const char* FREE_NAME = "cudaFree";
#endif

    static void *check_dlsym(void *value) {
        if (nullptr == value) {
            std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror() << std::endl;
            exit(1);
        }
        return value;
    }

    static CudaMallocFunc real_cuda_malloc_ = NULL;
    static CudaFreeFunc real_cuda_free_ = NULL;

    cudaError_t call_real_cuda_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_cuda_malloc_)) {
            real_cuda_malloc_ = (CudaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, MALLOC_NAME));
        }

        cudaError_t ret = real_cuda_malloc_(ptr, size);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] APIForwarder.call_real_cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

    cudaError_t call_real_cuda_free(void *ptr) {
        if (C10_UNLIKELY(nullptr == real_cuda_free_)) {
            real_cuda_free_ = (CudaFreeFunc) check_dlsym(dlsym(RTLD_NEXT, FREE_NAME));
        }

        cudaError_t ret = real_cuda_free_(ptr);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] APIForwarder.call_real_cuda_free "
                  << " ptr=" << ptr << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }
}
