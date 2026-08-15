#pragma once

#if defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cassert>
/*
 * ROCm API Mapping References:
 * - CUDA Driver API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Driver_API_functions_supported_by_HIP.html
 * - CUDA Runtime API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html
 */
// --- Error Handling Types and Constants ---
#define CUresult hipError_t
#define cudaError_t hipError_t
#define CUDA_SUCCESS hipSuccess
#define cudaSuccess hipSuccess
// --- Error Reporting Functions ---
#define cuGetErrorString hipDrvGetErrorString
#define cudaGetErrorString hipGetErrorString
// --- Memory Management Functions ---
#define CUdeviceptr hipDeviceptr_t
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define cuMemAddressReserve hipMemAddressReserve
#define cuMemAddressFree hipMemAddressFree
#define cuMemMap hipMemMap
#define cuMemUnmap hipMemUnmap
#define cuMemRelease hipMemRelease
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipFreeHost
#define cudaMemcpy hipMemcpy
#define cudaMemGetInfo hipMemGetInfo
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
// --- Memory Copy Direction Constants ---
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
// --- Device and Stream Types ---
#define CUdevice hipDevice_t
#define cudaStream_t hipStream_t
// --- Error codes ---
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
// --- Memory Allocation Handle ---
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
// --- Chunk size for memory creation operations (2 MB) ---
#define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
// --- Utility Macros ---
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// --- ROCm Version Feature Flags ---
// ROCm 6.x has hipMemCreate bug, requires chunked allocation workaround
// ROCm 7.0+ has fixed the bug, can use non-chunked allocation like CUDA
#if HIP_VERSION < 70000000
    #define TMS_ROCM_LEGACY_CHUNKED 1
#else
    #define TMS_ROCM_LEGACY_CHUNKED 0
#endif

#elif defined(USE_CUDA)
#include <cuda_runtime_api.h>
#include <cuda.h>

#define TMS_ROCM_LEGACY_CHUNKED 0

#elif defined(USE_XPU)
#include <cstddef>

typedef int CUresult;
typedef int cudaError_t;
typedef int CUdevice;        // device ordinal
typedef void *cudaStream_t;  // unused on XPU

#define CUDA_SUCCESS 0
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidDevicePointer 17

inline const char *cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidDevicePointer: return "invalid device pointer";
        default: return "unknown XPU error";
    }
}

typedef int cudaMemcpyKind;
#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyHostToDevice 1

inline cudaError_t cudaMallocHost(void **ptr, size_t size) {
    (void)size;
    *ptr = nullptr;
    return cudaErrorMemoryAllocation;
}

inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind) {
    (void)dst; (void)src; (void)size; (void)kind;
    return cudaErrorMemoryAllocation;
}

#define TMS_ROCM_LEGACY_CHUNKED 0

#else
#error "USE_PLATFORM is not set"
#endif
