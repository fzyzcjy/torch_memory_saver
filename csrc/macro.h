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
// Intel XPU (Level Zero) backend compatibility layer.
//
// This backend implements pause/resume via Level Zero VMM (not CUDA/HIP) in
// hardware_xpu_support.cpp. To reuse the platform-agnostic orchestration in
// core.cpp -- which returns cudaError_t and takes CUdevice (real CUDA types on
// CUDA, hipError_t via #define on ROCm) -- we typedef those to plain int here.
// hardware_xpu_support.cpp uses real Level Zero APIs and translates ze_result_t
// to these codes at the boundary; users only ever see the Python API.
#include <cstddef>

typedef int CUresult;
typedef int cudaError_t;     // Just an error code (0=success, 2=OOM, 17=bad ptr)
typedef int CUdevice;        // Device ordinal (0, 1, 2, ...)
typedef void *cudaStream_t;  // Unused on XPU (pluggable allocator is global)

#define CUDA_SUCCESS 0
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidDevicePointer 17

// cudaGetErrorString for the shared CUDA_ERROR_CHECK macro (utils.h); XPU has no
// CUDA/HIP runtime, so stringify just the cudaError_t codes the shared code returns.
inline const char *cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidDevicePointer: return "invalid device pointer";
        default: return "unknown XPU error";
    }
}

#define TMS_ROCM_LEGACY_CHUNKED 0

#else
#error "USE_PLATFORM is not set"
#endif
