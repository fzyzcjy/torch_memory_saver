#include "hardware_amd_support.h"

#if defined(USE_ROCM)

#include <iostream>

#if HIP_VERSION < 60304000
    #pragma message "You need to implement torch_memory_saver in ROCm/HIP 6.3.4 or lower. We did not support it currently."
#else
    #pragma message "Using ROCm/HIP >= 6.4.2 implementation"

namespace DeviceUtils {
    int get_global_device_id(hipDevice_t local_device_id) {
        // Check for HIP_VISIBLE_DEVICES environment variable
        const char* hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
        
        if (hip_visible && strlen(hip_visible) > 0) {
            std::string devices_str(hip_visible);
            std::stringstream ss(devices_str);
            std::string device_str;
            std::vector<int> device_list;
            
            // Parse comma-separated device list
            while (std::getline(ss, device_str, ',')) {
                if (!device_str.empty()) {
                    device_list.push_back(std::atoi(device_str.c_str()));
                }
            }
            
            if (local_device_id < device_list.size()) {
                int global_device_id = device_list[local_device_id];
#ifdef TMS_DEBUG_LOG
                std::cout << "[torch_memory_saver.cpp] HIP_VISIBLE_DEVICES=" << hip_visible 
                        << " local_device_id=" << local_device_id 
                        << " -> global_device_id=" << global_device_id << std::endl;
#endif
                return global_device_id;
            }
        }
        
        // Fallback: return local device ID as-is
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] No HIP_VISIBLE_DEVICES, using local_device_id=" << local_device_id << std::endl;
#endif
        return local_device_id;
    }
}

// Internal helper functions
namespace {
    void cu_mem_create_and_map(
        hipDevice_t device, 
        size_t aligned_size, 
        void* d_mem,
        std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        std::vector<size_t>& chunk_sizes
    ) {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;

        size_t num_chunks = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE;

        allocHandles.resize(num_chunks);
        chunk_sizes.resize(num_chunks);

        // Calculate chunk sizes
        for (size_t i = 0; i < num_chunks; ++i) {
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

    void cu_mem_unmap_and_release(
        hipDevice_t device,
        size_t aligned_size,
        hipDeviceptr_t d_mem,
        const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        const std::vector<size_t>& chunk_sizes
    ) {
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
}

namespace ROCmHIPImplementation {
    cudaError_t rocm_malloc(
        void **ptr, 
        CUdevice device, 
        size_t size, 
        const std::string& tag, 
        bool enable_cpu_backup,
        size_t& aligned_size,
        std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        std::vector<size_t>& chunk_sizes
    ) {
        // Get device
        CURESULT_CHECK(hipCtxGetDevice(&device));

        // Calculate aligned size
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        prop.allocFlags.compressionType = 0x0;

        size_t granularity;
        CURESULT_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                                hipMemAllocationGranularityMinimum));
        aligned_size = ((size + granularity - 1) / granularity) * granularity;
        aligned_size = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE * MEMCREATE_CHUNK_SIZE;

        assert(MEMCREATE_CHUNK_SIZE % granularity == 0);
        assert(aligned_size % MEMCREATE_CHUNK_SIZE == 0);
        assert(aligned_size % granularity == 0);

        // Get global device ID and determine NUMA node
        int global_device_id = DeviceUtils::get_global_device_id(device);
        uint64_t node_id = 0;
        if (global_device_id > 3) {
            node_id = 1;
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " size=" << size
                  << " granularity=" << granularity
                  << " aligned_size=" << aligned_size
                  << " node_id=" << node_id
                  << " device=" << device
                  << " global_device_id=" << global_device_id
                  << std::endl;
#endif

        // Reserve aligned memory address
        hipDeviceptr_t d_mem;
        CURESULT_CHECK(hipMemAddressReserve(&d_mem, aligned_size, granularity, 0, node_id));
        *ptr = (void*)d_mem;

        // Create and map chunks
        cu_mem_create_and_map(device, aligned_size, (hipDeviceptr_t)*ptr, 
                             allocHandles, chunk_sizes);

#ifdef TMS_DEBUG_LOG
        size_t num_chunks = allocHandles.size();
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
                  << " aligned_size=" << aligned_size
                  << " num_chunks=" << num_chunks
                  << std::endl;
#endif

        return cudaSuccess;
    }

    cudaError_t rocm_free(
        void *ptr,
        size_t size,
        size_t aligned_size,
        CUdevice device,
        const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        const std::vector<size_t>& chunk_sizes
    ) {
        // Unmap and release chunks
        cu_mem_unmap_and_release(device, size, (hipDeviceptr_t)ptr, allocHandles, chunk_sizes);

        // Free the reserved address using stored aligned_size
        CURESULT_CHECK(hipMemAddressFree((hipDeviceptr_t)ptr, aligned_size));

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
                  << " ptr=" << ptr << " size=" << size
                  << " aligned_size=" << aligned_size
                  << " num_chunks=" << allocHandles.size()
                  << std::endl;
#endif

        return cudaSuccess;
    }

    void rocm_pause(
        void* ptr,
        size_t size,
        size_t aligned_size,
        CUdevice device,
        bool enable_cpu_backup,
        void*& cpu_backup,
        const std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        const std::vector<size_t>& chunk_sizes
    ) {
        // Copy data to CPU backup if enabled
        if (enable_cpu_backup) {
            if (nullptr == cpu_backup) {
                CUDA_ERROR_CHECK(hipMallocHost(&cpu_backup, aligned_size));
            }
            SIMPLE_CHECK(cpu_backup != nullptr, "cpu_backup should not be nullptr");
            CUDA_ERROR_CHECK(cudaMemcpy(cpu_backup, ptr, aligned_size, hipMemcpyDeviceToHost));
        }

        // Unmap and release chunks (but keep metadata for resume)
        cu_mem_unmap_and_release(device, aligned_size, (hipDeviceptr_t)ptr, allocHandles, chunk_sizes);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                << " ptr=" << ptr << " size=" << size 
                << " aligned_size=" << aligned_size
                << " num_chunks=" << allocHandles.size()
                << std::endl;
#endif
    }

    void rocm_resume(
        void* ptr,
        size_t size,
        size_t aligned_size,
        CUdevice device,
        bool enable_cpu_backup,
        void* cpu_backup,
        std::vector<hipMemGenericAllocationHandle_t>& allocHandles,
        std::vector<size_t>& chunk_sizes
    ) {
        // Create new handles and map chunks
        cu_mem_create_and_map(device, aligned_size, (hipDeviceptr_t)ptr, allocHandles, chunk_sizes);

        // Restore from CPU backup if enabled
        if (enable_cpu_backup) {
            SIMPLE_CHECK(cpu_backup != nullptr, "cpu_backup should not be nullptr");
            CUDA_ERROR_CHECK(cudaMemcpy(ptr, cpu_backup, aligned_size, hipMemcpyHostToDevice));
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                << " ptr=" << ptr << " size=" << size
                << " aligned_size=" << aligned_size
                << " num_chunks=" << allocHandles.size()
                << std::endl;
#endif
    }
}

#endif // HIP_VERSION

#endif // USE_ROCM