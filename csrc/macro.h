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
// Intel XPU (Level Zero) backend. There is no CUDA/HIP runtime here, so we map
// the handful of CUDA spellings used by the shared (device-agnostic) code onto
// plain types. The actual device work lives in hardware_xpu_support.cpp and is
// expressed directly in Level Zero / SYCL, not through these aliases.
#include <cstddef>

typedef int CUresult;
typedef int cudaError_t;
typedef int CUdevice;        // a device ordinal on XPU
typedef void *cudaStream_t;  // unused on XPU (pluggable allocator passes it through)

#define CUDA_SUCCESS 0
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidDevicePointer 17

// The shared CUDA_ERROR_CHECK macro (utils.h) formats errors via
// cudaGetErrorString. The ROCm branch above provides this by mapping it to
// hipGetErrorString; XPU has no CUDA/HIP runtime, so we provide the equivalent
// here. The XPU backend (hardware_xpu_support.cpp) reports detailed Level Zero
// result codes itself; these labels only stringify the small set of
// cudaError_t values the shared code returns.
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
