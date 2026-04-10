#pragma once
#include <iostream>
#include <vector>
#include "macro.h"

// #define TMS_DEBUG_LOG

// Cannot use pytorch (libc10.so) since LD_PRELOAD happens earlier than `import torch`
// Thus copy from torch Macros.h
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

#define SIMPLE_CHECK(COND, MSG) \
  do { \
    if (!(COND)) { \
        std::cerr << "[torch_memory_saver.cpp] " << MSG \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define CURESULT_CHECK(EXPR) \
  do { \
    CUresult __result = (EXPR); \
    if (__result != CUDA_SUCCESS) { \
        const char* err_str = nullptr; \
        cuGetErrorString(__result, &err_str); \
        std::cerr << "[torch_memory_saver.cpp] CUresult error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define CUDA_ERROR_CHECK(EXPR) \
  do { \
    cudaError_t __result = (EXPR); \
    if (__result != cudaSuccess) { \
        const char* err_str = cudaGetErrorString(__result); \
        std::cerr << "[torch_memory_saver.cpp] cudaError error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)


namespace CUDAUtils {
#if defined(USE_ROCM)

    #if HIP_VERSION < 60304000 // rocm/hip 6.3.4
        #pragma message "You need to implement torch_memory_saver in ROCm/HIP 6.3.4 or lower. We did not support it currently."
    #else
        #if TMS_ROCM_LEGACY_CHUNKED
            #pragma message "Using ROCm/HIP 6.x implementation (chunked allocation workaround)"
        #else
            #pragma message "Using ROCm/HIP >= 7.0 implementation (single allocation, same as CUDA)"
        #endif

        static CUdevice cu_ctx_get_device() {
            int device;
            CUDA_ERROR_CHECK(hipGetDevice(&device));
            return static_cast<CUdevice>(device);
        }

        static CUdevice cu_device_get(int device_ordinal) {
            CUdevice ans;
            CURESULT_CHECK(hipDeviceGet(&ans, device_ordinal));
            return ans;
        }

        static hipMemAllocationProp make_alloc_prop(CUdevice device) {
            hipMemAllocationProp prop = {};
            prop.type = hipMemAllocationTypePinned;
            prop.location.type = hipMemLocationTypeDevice;
            prop.location.id = device;
            prop.allocFlags.compressionType = 0x0;
            return prop;
        }

        static cudaError_t cu_mem_create(CUmemGenericAllocationHandle *alloc_handle, size_t size, CUdevice device) {
            hipMemAllocationProp prop = make_alloc_prop(device);

            hipError_t ret = hipMemCreate(alloc_handle, size, &prop, 0);
            if (ret == hipErrorOutOfMemory) {
                std::cerr << "[torch_memory_saver.cpp] hipMemCreate hipErrorOutOfMemory (may not be an issue e.g. torch allocator will free cache and retry)" << std::endl;
                return hipErrorOutOfMemory;
            }
            CURESULT_CHECK(ret);

            return hipSuccess;
        }

        static void cu_mem_set_access(void *ptr, size_t size, CUdevice device) {
            hipMemAccessDesc accessDesc = {};
            accessDesc.location.type = hipMemLocationTypeDevice;
            accessDesc.location.id = device;
            accessDesc.flags = hipMemAccessFlagsProtReadWrite;
            CURESULT_CHECK(hipMemSetAccess(ptr, size, &accessDesc, 1));
        }

        static size_t cu_mem_get_granularity(CUdevice device) {
            hipMemAllocationProp prop = make_alloc_prop(device);
            size_t granularity = 0;
            CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                                          hipMemAllocationGranularityMinimum));
            return granularity;
        }
    #endif

#elif defined(USE_CUDA)
    // Build the CUmemAllocationProp used by both cu_mem_create and
    // cu_mem_get_granularity. Keeping them in sync is critical: the
    // granularity the driver reports depends on the flags in the prop (in
    // particular gpuDirectRDMACapable), so querying with a different prop can
    // return an alignment that the actual allocation rejects.
    static CUmemAllocationProp make_alloc_prop(CUdevice device) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        int flag = 0;
        CURESULT_CHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
        if (flag) {
            prop.allocFlags.gpuDirectRDMACapable = 1;
        }
        return prop;
    }

    static cudaError_t cu_mem_create(CUmemGenericAllocationHandle *alloc_handle, size_t size, CUdevice device) {
        CUmemAllocationProp prop = make_alloc_prop(device);

        CUresult ret = cuMemCreate(alloc_handle, size, &prop, 0);
        if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
            std::cerr << "[torch_memory_saver.cpp] cuMemCreate CUDA_ERROR_OUT_OF_MEMORY (may not be an issue e.g. torch allocator will free cache and retry)" << std::endl;
            return cudaErrorMemoryAllocation;
        }
        CURESULT_CHECK(ret);

        return cudaSuccess;
    }

    static void cu_mem_set_access(void *ptr, size_t size, CUdevice device) {
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CURESULT_CHECK(cuMemSetAccess((CUdeviceptr) ptr, size, &access_desc, 1));
    }

    static CUdevice cu_ctx_get_device() {
        CUdevice ans;
        CURESULT_CHECK(cuCtxGetDevice(&ans));
        return ans;
    }

    static CUdevice cu_device_get(int device_ordinal) {
        CUdevice ans;
        CURESULT_CHECK(cuDeviceGet(&ans, device_ordinal));
        return ans;
    }

    static size_t cu_mem_get_granularity(CUdevice device) {
        CUmemAllocationProp prop = make_alloc_prop(device);
        size_t granularity = 0;
        CURESULT_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        return granularity;
    }

#else
    #error "USE_PLATFORM is not set"

#endif
}

inline bool get_bool_env_var(const char* name) {
    const char* env_cstr = std::getenv(name);
    if (env_cstr == nullptr) {
        return false;
    }

    std::string env_str(env_cstr);
    if (env_str == "1" || env_str == "true" || env_str == "TRUE" || env_str == "yes" || env_str == "YES") {
        return true;
    }
    if (env_str == "0" || env_str == "false" || env_str == "FALSE" || env_str == "no" || env_str == "NO") {
        return false;
    }

    std::cerr << "[torch_memory_saver.cpp] Unsupported environment varialbe value "
              << " name=" << name << " value=" << env_str
              << std::endl;
    exit(1);
}
