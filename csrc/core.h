#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include "utils.h"

enum class AllocationState {
    ACTIVE,  // Memory is mapped and accessible
    PAUSED    // Memory is unmapped and inaccessible
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    CUmemGenericAllocationHandle allocHandle;
    std::string tag;
    AllocationState state;  // Track whether this allocation is paused or resumed
    bool enable_cpu_backup;
    void* cpu_backup;
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup);
    cudaError_t free(void* ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
};
