#include "hardware_musa_support.h"

#include <dlfcn.h>
#include <cstring>

#include "utils.h"

namespace MUSAImplementation {
    static bool caller_belongs_to_library(void* caller, const char* name) {
        Dl_info info;
        if (dladdr(caller, &info) == 0 || info.dli_fname == nullptr) {
            return false;
        }
        return std::strstr(info.dli_fname, name) != nullptr;
    }

    bool caller_requires_plain_malloc(void* caller) {
        return caller_belongs_to_library(caller, "libmccl")
            || caller_belongs_to_library(caller, "libmudnn");
    }

    void zero_region(void* ptr, size_t size, int device) {
        int original_device = -1;
        CUDA_ERROR_CHECK(cudaGetDevice(&original_device));
        CUDA_ERROR_CHECK(cudaSetDevice(device));
        CUDA_ERROR_CHECK(musaMemset(ptr, 0, size));
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        CUDA_ERROR_CHECK(cudaSetDevice(original_device));
    }
}
