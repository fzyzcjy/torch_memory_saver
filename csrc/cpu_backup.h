#pragma once

#include "utils.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(USE_CUDA)
#include <iterator>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#endif

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

namespace cpu_backuper {

#if defined(USE_XPU)

// XPU owns host shadows with malloc/free in hardware_xpu_support.cpp and must
// not call these CUDA/HIP helpers (kind is unused on that path).
inline void allocate(size_t, CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backuper::allocate is not supported on XPU");
}

inline void release(CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backuper::release is not supported on XPU");
}

inline void offload(void*, size_t, CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backuper::offload is not supported on XPU");
}

inline void onload(void*, size_t, CUdevice, const CpuBackupSlot&) {
    SIMPLE_CHECK(false, "cpu_backuper::onload is not supported on XPU");
}

#else

inline void raw_free(void* p, CpuBackupKind kind, size_t size) {
    switch (kind) {
        case CpuBackupKind::MMAP:
            if (munmap(p, size) != 0) {
                const int err = errno;
                SIMPLE_CHECK(false,
                             "munmap cpu_backup failed errno=" << err << " " << strerror(err));
            }
            break;
        case CpuBackupKind::PINNED:
            CUDA_ERROR_CHECK(cudaFreeHost(p));
            break;
        default:
            SIMPLE_CHECK(false, "unknown cpu_backup_kind=" << static_cast<int>(kind));
    }
}

#if defined(USE_CUDA)
// Released host backup buffers are pooled for reuse: recreating them dominates
// pause()/resume() otherwise (issue #111). cudaMallocHost page-locks every
// page (~1.1 s for 4 GiB on H20) and a fresh anonymous mmap pays first-touch
// faults on every cycle, while the DMA copy itself takes ~80 ms. The pool is
// keyed by (kind, size) and holds at most the process's peak backup footprint,
// which the host already had to accommodate while paused.
// TMS_CPU_BACKUP_POOL_MAX_BYTES opts in and caps the pooled bytes: 0
// (default) keeps the legacy free-on-release behavior, -1 pools without a
// cap, any positive value is a byte cap. A value set via the C/Python API
// takes precedence over the env var.
struct CpuBackupSlotPool {
    std::mutex mu;
    std::map<std::pair<CpuBackupKind, size_t>, std::vector<void*>> buckets;
    size_t total_bytes = 0;
    int64_t max_bytes = 0;

    CpuBackupSlotPool() {
        const char* env = std::getenv("TMS_CPU_BACKUP_POOL_MAX_BYTES");
        if (env != nullptr && *env != '\0') {
            char* end = nullptr;
            const long long parsed = std::strtoll(env, &end, 10);
            SIMPLE_CHECK(end != env && *end == '\0',
                         "TMS_CPU_BACKUP_POOL_MAX_BYTES must be an integer value=" << env);
            max_bytes = parsed < 0 ? -1 : parsed;
        }
    }
};

inline CpuBackupSlotPool& slot_pool() {
    static CpuBackupSlotPool pool;
    return pool;
}

inline bool pool_take(size_t size, CpuBackupSlot& slot) {
    CpuBackupSlotPool& pool = slot_pool();
    const std::lock_guard<std::mutex> lock(pool.mu);
    if (pool.max_bytes == 0) {
        return false;
    }
    auto it = pool.buckets.find({slot.kind, size});
    if (it == pool.buckets.end()) {
        return false;
    }
    slot.data = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) {
        pool.buckets.erase(it);
    }
    pool.total_bytes -= size;
    slot.size = size;
    return true;
}

// Returns true when the slot was pooled (slot cleared); false when the caller
// must really free it.
inline bool pool_offer(CpuBackupSlot& slot) {
    CpuBackupSlotPool& pool = slot_pool();
    const std::lock_guard<std::mutex> lock(pool.mu);
    if (pool.max_bytes == 0 ||
        (pool.max_bytes > 0 && pool.total_bytes + slot.size > static_cast<size_t>(pool.max_bytes))) {
        return false;
    }
    pool.buckets[{slot.kind, slot.size}].push_back(slot.data);
    pool.total_bytes += slot.size;
    slot.data = nullptr;
    slot.size = 0;
    return true;
}

inline void set_pool_max_bytes(int64_t value) {
    CpuBackupSlotPool& pool = slot_pool();
    const std::lock_guard<std::mutex> lock(pool.mu);
    pool.max_bytes = value < 0 ? -1 : value;
    if (pool.max_bytes < 0) {
        return;
    }
    // Evict (really free) from the tail of the ordered buckets until within
    // the new cap.
    while (pool.total_bytes > static_cast<size_t>(value) && !pool.buckets.empty()) {
        auto it = std::prev(pool.buckets.end());
        void* p = it->second.back();
        const CpuBackupKind kind = it->first.first;
        const size_t size = it->first.second;
        it->second.pop_back();
        if (it->second.empty()) {
            pool.buckets.erase(it);
        }
        pool.total_bytes -= size;
        raw_free(p, kind, size);
    }
}

inline size_t pool_total_bytes() {
    CpuBackupSlotPool& pool = slot_pool();
    const std::lock_guard<std::mutex> lock(pool.mu);
    return pool.total_bytes;
}

inline int64_t pool_max_bytes() {
    CpuBackupSlotPool& pool = slot_pool();
    const std::lock_guard<std::mutex> lock(pool.mu);
    return pool.max_bytes;
}
#endif  // USE_CUDA

inline void allocate(size_t size, CpuBackupSlot& slot) {
#if defined(USE_CUDA)
    if (pool_take(size, slot)) {
        return;
    }
#endif
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

inline void release(CpuBackupSlot& slot) {
    if (slot.data == nullptr) {
        return;
    }
#if defined(USE_CUDA)
    if (pool_offer(slot)) {
        return;
    }
#endif
    raw_free(slot.data, slot.kind, slot.size);
    slot.data = nullptr;
    slot.size = 0;
}

inline void offload(void* gpu_ptr, size_t size, CpuBackupSlot& slot) {
    if (slot.data == nullptr) {
        allocate(size, slot);
    }
    SIMPLE_CHECK(slot.data != nullptr && slot.size == size, "cpu_backup slot size mismatch");
    // TODO may use cudaMemcpyAsync if needed
    CUDA_ERROR_CHECK(cudaMemcpy(slot.data, gpu_ptr, size, cudaMemcpyDeviceToHost));
}

inline void onload(void* gpu_ptr, size_t size, CUdevice device, const CpuBackupSlot& slot) {
    SIMPLE_CHECK(slot.data != nullptr, "cpu_backup missing on resume");
    SIMPLE_CHECK(slot.size == size, "cpu_backup slot size mismatch");
    int original_device = -1;
    if (slot.kind == CpuBackupKind::MMAP) {
        CUDA_ERROR_CHECK(cudaGetDevice(&original_device));
        CUDA_ERROR_CHECK(cudaSetDevice(static_cast<int>(device)));
    }
    // TODO may use cudaMemcpyAsync if needed
    CUDA_ERROR_CHECK(cudaMemcpy(gpu_ptr, slot.data, size, cudaMemcpyHostToDevice));
    // Pageable H2D may return after staging while DMA is still in flight.
    if (slot.kind == CpuBackupKind::MMAP) {
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        CUDA_ERROR_CHECK(cudaSetDevice(original_device));
    }
}

#endif

}

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
