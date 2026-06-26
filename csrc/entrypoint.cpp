#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <optional>
#include "macro.h"

// ----------------------------------------------- threadlocal configs --------------------------------------------------

class ThreadLocalConfig {
public:
    std::string current_tag_ = "default";

    bool is_interesting_region() {
        if (!is_interesting_region_.has_value()) {
            is_interesting_region_ = get_bool_env_var("TMS_INIT_ENABLE");
        }
        return is_interesting_region_.value();
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() {
        if (!enable_cpu_backup_.has_value()) {
            enable_cpu_backup_ = get_bool_env_var("TMS_INIT_ENABLE_CPU_BACKUP");
        }
        return enable_cpu_backup_.value();
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

    bool enable_disk_backup() {
        if (!enable_disk_backup_.has_value()) {
            enable_disk_backup_ = get_bool_env_var("TMS_INIT_ENABLE_DISK_BACKUP");
        }
        return enable_disk_backup_.value();
    }

    void set_enable_disk_backup(bool value) {
        enable_disk_backup_ = value;
    }

private:
    std::optional<bool> is_interesting_region_;
    std::optional<bool> enable_cpu_backup_;
    std::optional<bool> enable_disk_backup_;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_,
            thread_local_config.enable_cpu_backup(), thread_local_config.enable_disk_backup());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    return TorchMemorySaver::instance().free(ptr);
}
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_malloc "
              << " size=" << size << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    // The torch MemPool allocator is pool-scoped: this hook only fires for
    // allocations made inside a region() (i.e. within use_mem_pool), so we are
    // always in an interesting region here.
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    void *ptr;
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_,
        thread_local_config.enable_cpu_backup(), thread_local_config.enable_disk_backup()));
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
#if !defined(USE_XPU)
    // Upstream CUDA/ROCm behavior: assert we are in an interesting region.
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
#endif
    // XPU is torch-mode only with a pool-scoped allocator, so this hook fires
    // exactly for the region() tensors we allocated. Those are freed when they
    // go out of scope, which can happen AFTER the region context has exited
    // (is_interesting_region() back to false) -- hence no assert on XPU. The
    // pointer is always VMM-managed here, so free() handles it directly.
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().free(ptr));
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.set_interesting_region(is_interesting_region);
}

bool tms_get_interesting_region() {
    return thread_local_config.is_interesting_region();
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

const char* tms_get_current_tag() {
    return thread_local_config.current_tag_.c_str();
}

bool tms_get_enable_cpu_backup() {
    return thread_local_config.enable_cpu_backup();
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.set_enable_cpu_backup(enable_cpu_backup);
}

bool tms_get_enable_disk_backup() {
    return thread_local_config.enable_disk_backup();
}

void tms_set_enable_disk_backup(bool enable_disk_backup) {
    thread_local_config.set_enable_disk_backup(enable_disk_backup);
}

void tms_set_disk_backup_dir(const char* dir) {
    SIMPLE_CHECK(dir != nullptr, "disk backup dir should not be null");
    TorchMemorySaver::instance().set_disk_backup_dir(std::string(dir));
}

void set_memory_margin_bytes(uint64_t value) {
    TorchMemorySaver::instance().set_memory_margin_bytes(value);
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}

uint8_t* tms_get_cpu_backup_pointer(const uint8_t* gpu_ptr, uint64_t size) {
    return TorchMemorySaver::instance().get_cpu_backup_pointer(gpu_ptr, size);
}

#if defined(USE_XPU)
// Pre-warm per-device SYCL contexts before the pluggable allocator is
// registered (creating a sycl::context inside an allocator callback can
// deadlock the SYCL runtime). Called from Python init.
void tms_xpu_prewarm_devices(int n_devices) {
    XPUImplementation::xpu_prewarm_devices(n_devices);
}

// Authoritative free-device-memory reading via sysman. torch.xpu's allocator
// accounting does NOT reflect physical pages released by zeVirtualMemUnmap.
uint64_t tms_xpu_device_free_bytes(int device_id) {
    return XPUImplementation::xpu_device_free_bytes(device_id);
}

// Committed physical bytes the saver holds on an XPU device. Driver-independent
// (sysman free-bytes is frozen/deprecated on newer drivers); used by tests to
// verify real physical commit/release across pause/resume.
uint64_t tms_xpu_committed_bytes(int device_id) {
    return TorchMemorySaver::instance().xpu_committed_bytes(device_id);
}
#endif
}
