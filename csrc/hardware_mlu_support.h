#pragma once
#include <cn_api.h>
#include <cnrt.h>

#define CUresult CNresult
#define cudaError_t cnrtRet_t
#define CUDA_SUCCESS CN_SUCCESS
#define cudaSuccess cnrtSuccess
#define cuGetErrorString cnGetErrorString
#define cudaGetErrorString cnrtGetErrorStr
#define CUdeviceptr CNaddr
#define CUdevice CNdev
#define CUmemGenericAllocationHandle CNmemGenericAllocationHandle
#define cudaStream_t cnrtQueue_t
#define cuMemAddressReserve cnMemAddressReserve
#define cuMemAddressFree cnMemAddressFree
#define cuMemMap cnMemMap
#define cuMemUnmap cnMemUnmap
#define cuMemRelease cnMemRelease
#define cudaMalloc cnrtMalloc
#define cudaFree cnrtFree
#define cudaMallocHost cnrtHostMalloc
#define cudaFreeHost cnrtFreeHost
#define cudaMemGetInfo cnrtMemGetInfo
#define cudaDeviceSynchronize cnrtSyncDevice
#define cudaGetDevice cnrtGetDevice
#define cudaSetDevice cnrtSetDevice
#define cudaMemcpyDeviceToHost cnrtMemcpyDevToHost
#define cudaMemcpyHostToDevice cnrtMemcpyHostToDev
#define cudaErrorMemoryAllocation cnrtErrorNoMem
#define TMS_ROCM_LEGACY_CHUNKED 0

inline cnrtRet_t cudaMemcpy(void *dst, const void *src, size_t size, cnrtMemTransDir_t kind) {
    return cnrtMemcpy(dst, const_cast<void *>(src), size, kind);
}
