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
// This backend does NOT use CUDA/HIP. Instead, it implements pause/resume via
// Level Zero VMM (zeVirtualMemMap/Unmap, zePhysicalMemCreate/Destroy) in
// hardware_xpu_support.cpp. However, to share the orchestration code in core.cpp
// (which is platform-agnostic), we map the CUDA-style type names onto plain C
// types. This avoids duplicating the entire allocator logic for each platform.
//
// Why use cudaError_t instead of ze_result_t?
//   - The shared code (core.h, core.cpp) defines a platform-agnostic interface
//     using cudaError_t return values and CUdevice for device IDs.
//   - On CUDA: these map to actual CUDA runtime types.
//   - On ROCm: #define cudaError_t hipError_t (HIP's equivalent).
//   - On XPU: we typedef them to plain int (just error codes: 0=success, etc.)
//   - This allows one implementation of TorchMemorySaver to work across all
//     platforms, while the actual GPU work uses native APIs in each backend.
//
// The XPU implementation in hardware_xpu_support.cpp uses real Level Zero APIs
// (ze_result_t, etc.) and translates them to these compatibility codes at the
// boundary. Users never see these types—they call the Python API.
#include <cstddef>

typedef int CUresult;
typedef int cudaError_t;     // Just an error code (0=success, 2=OOM, 17=bad ptr)
typedef int CUdevice;        // Device ordinal (0, 1, 2, ...)
typedef void *cudaStream_t;  // Unused on XPU (pluggable allocator is global)

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
