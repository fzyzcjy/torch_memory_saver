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
        // After rocm-7.0, we can use the same way to implement torch_memory_saver as CUDA side. --> Need to verify
        #pragma message "Using ROCm/HIP >= 6.4.2 implementation"
        // hipMemCreate currently has issue in rocm-6.3.4. After it is fixed in rocm-7.0, we can use the same way to implement torch_memory_saver as CUDA side.
        // Current, we based on the chuck-wise method to implement it.
        static void cu_mem_create_and_map(hipDevice_t device, 
                                          size_t aligned_size, 
                                          void* d_mem,
                                          std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
                                          std::vector<size_t>& chunk_sizes) {

            hipMemAllocationProp prop = {};
            prop.type = hipMemAllocationTypePinned;
            prop.location.type = hipMemLocationTypeDevice;
            prop.location.id = device;

            // // Get granularity
            // size_t granularity;
            // CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
            //             hipMemAllocationGranularityMinimum));

            // // Make sure chunk size is aligned with hardware granularity
            // size_t aligned_chunk_size = ((MEMCREATE_CHUNK_SIZE + granularity - 1) / granularity) * granularity;
            // size_t num_chunks = (size + aligned_chunk_size - 1) / aligned_chunk_size;
            
            // Get granularity, Make sure chunk size is aligned with hardware granularity
            // size == aligned_size  
            size_t num_chunks = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE;

            allocHandles.resize(num_chunks);
            chunk_sizes.resize(num_chunks);

            // Calculate chunk sizes
            for (size_t i = 0; i < num_chunks; ++i) {
                // chunk_sizes[i] = MIN(size - i * aligned_chunk_size, aligned_chunk_size);
                chunk_sizes[i] = MIN(aligned_size - i * MEMCREATE_CHUNK_SIZE, MEMCREATE_CHUNK_SIZE);
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
            CURESULT_CHECK(hipMemSetAccess(d_mem, aligned_size, &accessDesc, 1));
        }


        static void cu_mem_unmap_and_release(hipDevice_t device,
                                            size_t aligned_size,
                                            hipDeviceptr_t d_mem,
                                            const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
                                            const std::vector<size_t>& chunk_sizes) {
          
            // Unmap each chunk
            size_t deallocated_size = 0;
            for (size_t i = 0; i < allocHandles.size(); ++i) {
                void* map_addr = (void*)((uintptr_t)d_mem + deallocated_size);
                CURESULT_CHECK(hipMemUnmap((hipDeviceptr_t)map_addr, chunk_sizes[i]));
                deallocated_size += chunk_sizes[i];
#ifdef TMS_DEBUG_LOG
                std::cout << "[torch_memory_saver.cpp] unmapped chunk " << i << " at offset " << deallocated_size - chunk_sizes[i] << std::endl;
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
            CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
            return granularity;
        }

        static CUdevice cu_ctx_get_device() {
            CUdevice ans;
            CURESULT_CHECK(hipCtxGetDevice(&ans));
            return ans;
        }

        static CUdevice cu_device_get(int device_ordinal) {
            CUdevice ans;
            CURESULT_CHECK(hipDeviceGet(&ans, device_ordinal));
            return ans;
        }
    #endif

#elif defined(USE_CUDA)
    static void cu_mem_create(CUmemGenericAllocationHandle *alloc_handle, size_t size, CUdevice device) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        int flag = 0;
        CURESULT_CHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
        if (flag) {  // support GPUDirect RDMA if possible
            prop.allocFlags.gpuDirectRDMACapable = 1;
        }

        CURESULT_CHECK(cuMemCreate(alloc_handle, size, &prop, 0));
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
