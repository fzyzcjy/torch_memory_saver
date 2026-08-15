#pragma once

#include "utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if !defined(USE_XPU)
#include <cerrno>
#include <sys/mman.h>
#endif

enum class CpuBackupKind : uint8_t {
    PINNED = 0,
    MMAP = 1,
};

constexpr CpuBackupKind kDefaultCpuBackupKind = CpuBackupKind::PINNED;

struct CpuBackupSlot {
    CpuBackupKind kind = kDefaultCpuBackupKind;
    void* data = nullptr;
    size_t size = 0;
};

#if defined(USE_XPU)

// XPU owns host shadows with malloc/free in hardware_xpu_support.cpp and must
// not call these CUDA/HIP helpers (kind is unused on that path).
inline void cpu_backup_release(CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backup_release is not supported on XPU");
}

inline void cpu_backup_offload(void*, size_t, CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backup_offload is not supported on XPU");
}

inline void cpu_backup_onload(void*, size_t, const CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backup_onload is not supported on XPU");
}

#else

inline void cpu_backup_release(CpuBackupSlot& slot) {
    if (slot.data == nullptr) {
        return;
    }
    switch (slot.kind) {
        case CpuBackupKind::MMAP: {
            if (munmap(slot.data, slot.size) != 0) {
                const int err = errno;
                SIMPLE_CHECK(false,
                             "munmap cpu_backup failed errno=" << err << " " << strerror(err));
            }
            break;
        }
        case CpuBackupKind::PINNED:
            CUDA_ERROR_CHECK(cudaFreeHost(slot.data));
            break;
        default:
            SIMPLE_CHECK(false, "unknown cpu_backup_kind=" << static_cast<int>(slot.kind));
    }
    slot.data = nullptr;
    slot.size = 0;
}

inline void cpu_backup_offload(void* gpu_ptr, size_t size, CpuBackupSlot& slot) {
    if (slot.data == nullptr) {
        switch (slot.kind) {
            case CpuBackupKind::MMAP: {
                void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) {
                    const int err = errno;
                    SIMPLE_CHECK(false,
                                 "mmap cpu_backup failed errno=" << err << " " << strerror(err));
                }
                slot.data = p;
                break;
            }
            case CpuBackupKind::PINNED:
                CUDA_ERROR_CHECK(cudaMallocHost(&slot.data, size));
                SIMPLE_CHECK(slot.data != nullptr, "cudaMallocHost cpu_backup returned nullptr");
                break;
            default:
                SIMPLE_CHECK(false, "unknown cpu_backup_kind=" << static_cast<int>(slot.kind));
        }
        slot.size = size;
    }
    SIMPLE_CHECK(slot.data != nullptr && slot.size == size, "cpu_backup slot size mismatch");
    // TODO may use cudaMemcpyAsync if needed
    CUDA_ERROR_CHECK(cudaMemcpy(slot.data, gpu_ptr, size, cudaMemcpyDeviceToHost));
}

inline void cpu_backup_onload(void* gpu_ptr, size_t size, const CpuBackupSlot& slot) {
    SIMPLE_CHECK(slot.data != nullptr, "cpu_backup missing on resume");
    SIMPLE_CHECK(slot.size == size, "cpu_backup slot size mismatch");
    // TODO may use cudaMemcpyAsync if needed
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ptr, slot.data, size, cudaMemcpyHostToDevice));
    // Pageable H2D may return after staging while DMA is still in flight.
    if (slot.kind == CpuBackupKind::MMAP) {
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
}

#endif

inline CpuBackupKind cpu_backup_kind_from_str(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return kDefaultCpuBackupKind;
    }
    std::string s(value);
    if (s == "mmap") {
#if defined(USE_ROCM) || defined(USE_XPU)
        SIMPLE_CHECK(false, "cpu_backup_backend=mmap is not supported on this platform");
#else
        return CpuBackupKind::MMAP;
#endif
    }
    if (s == "pinned") {
        return CpuBackupKind::PINNED;
    }
    SIMPLE_CHECK(false, "cpu_backup_backend must be mmap or pinned value=" << s);
    return kDefaultCpuBackupKind;
}

inline const char* cpu_backup_kind_to_str(CpuBackupKind kind) {
    switch (kind) {
        case CpuBackupKind::MMAP:
            return "mmap";
        case CpuBackupKind::PINNED:
            return "pinned";
        default:
            SIMPLE_CHECK(false, "unknown cpu_backup_kind=" << static_cast<int>(kind));
            return "";
    }
}
