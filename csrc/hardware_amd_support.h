#pragma once
#include "macro.h"
#include "utils.h"
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

#if defined(USE_ROCM)

// Device utility functions for ROCm
namespace DeviceUtils {
    // Get global device ID from local device ID
    int get_global_device_id(hipDevice_t local_device_id);
}

// High-level ROCm implementation functions
namespace ROCmHIPImplementation {
    // Malloc implementation for ROCm
    cudaError_t rocm_malloc(
        void **ptr, 
        CUdevice device, 
        size_t size, 
        const std::string& tag, 
        bool enable_cpu_backup,
        size_t& aligned_size,
        std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        std::vector<size_t>& chunk_sizes
    );
    
    // Free implementation for ROCm
    cudaError_t rocm_free(
        void *ptr,
        size_t size,
        size_t aligned_size,
        CUdevice device,
        const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        const std::vector<size_t>& chunk_sizes
    );
    
    // Pause implementation for ROCm
    void rocm_pause(
        void* ptr,
        size_t size,
        size_t aligned_size,
        CUdevice device,
        bool enable_cpu_backup,
        void*& cpu_backup,
        const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        const std::vector<size_t>& chunk_sizes
    );
    
    // Resume implementation for ROCm
    void rocm_resume(
        void* ptr,
        size_t size,
        size_t aligned_size,
        CUdevice device,
        bool enable_cpu_backup,
        void* cpu_backup,
        std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        std::vector<size_t>& chunk_sizes
    );
}

#endif // USE_ROCM