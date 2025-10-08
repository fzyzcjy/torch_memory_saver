#pragma once
#include "macro.h"
#include "utils.h"
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <mutex>

#if defined(USE_ROCM)

// Forward declaration
enum class AllocationState;
struct AllocationMetadata;

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
        std::unordered_map<void*, AllocationMetadata>& allocation_metadata,
        std::mutex& allocator_metadata_mutex
    );
    
    // Free implementation for ROCm
    cudaError_t rocm_free(
        void *ptr,
        std::unordered_map<void*, AllocationMetadata>& allocation_metadata,
        std::mutex& allocator_metadata_mutex
    );
    
    // Pause implementation for ROCm
    void rocm_pause(
        const std::string& tag,
        std::unordered_map<void*, AllocationMetadata>& allocation_metadata,
        std::mutex& allocator_metadata_mutex
    );
    
    // Resume implementation for ROCm
    void rocm_resume(
        const std::string& tag,
        std::unordered_map<void*, AllocationMetadata>& allocation_metadata,
        std::mutex& allocator_metadata_mutex
    );
}

#endif // USE_ROCM