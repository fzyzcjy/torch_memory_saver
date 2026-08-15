#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <optional>
#include "macro.h"
#include "cpu_backup.h"

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

    CpuBackupKind cpu_backup_kind() const {
        return cpu_backup_kind_;
    }

    void set_cpu_backup_kind(CpuBackupKind value) {
        cpu_backup_kind_ = value;
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
    CpuBackupKind cpu_backup_kind_ = cpu_backup_kind_from_str(
        std::getenv("TMS_INIT_CPU_BACKUP_BACKEND"));
    std::optional<bool> enable_disk_backup_;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        const bool enable_cpu_backup = thread_local_config.enable_cpu_backup();
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_,
            enable_cpu_backup, thread_local_config.cpu_backup_kind(),
            thread_local_config.enable_disk_backup());
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
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    void *ptr;
    const bool enable_cpu_backup = thread_local_config.enable_cpu_backup();
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_,
        enable_cpu_backup, thread_local_config.cpu_backup_kind(),
        thread_local_config.enable_disk_backup()));
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
#if !defined(USE_XPU)
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
#endif
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

const char* tms_get_cpu_backup_backend() {
    return cpu_backup_kind_to_str(thread_local_config.cpu_backup_kind());
}

void tms_set_cpu_backup_backend(const char* backend) {
    thread_local_config.set_cpu_backup_kind(cpu_backup_kind_from_str(backend));
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

void tms_set_retain_cpu_backup(bool retain_cpu_backup) {
#ifndef USE_CUDA
    SIMPLE_CHECK(!retain_cpu_backup, "retained CPU backup policy is CUDA-only");
#else
    TorchMemorySaver::instance().set_retain_cpu_backup(retain_cpu_backup);
#endif
}

bool tms_get_retain_cpu_backup() {
#ifndef USE_CUDA
    return false;
#else
    return TorchMemorySaver::instance().get_retain_cpu_backup();
#endif
}

int tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    return static_cast<int>(TorchMemorySaver::instance().pause(tag_str));
}

int tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    return static_cast<int>(TorchMemorySaver::instance().resume(tag_str));
}

uint8_t* tms_get_cpu_backup_pointer(const uint8_t* gpu_ptr, uint64_t size) {
    return TorchMemorySaver::instance().get_cpu_backup_pointer(gpu_ptr, size);
}

#if defined(USE_XPU)
void tms_xpu_prewarm_devices(int n_devices) {
    XPUImplementation::xpu_prewarm_devices(n_devices);
}

uint64_t tms_xpu_device_free_bytes(int device_id) {
    return XPUImplementation::xpu_device_free_bytes(device_id);
}

uint64_t tms_xpu_committed_bytes(int device_id) {
    return TorchMemorySaver::instance().xpu_committed_bytes(device_id);
}

uint64_t tms_xpu_leaked_bytes(int device_id) {
    return TorchMemorySaver::instance().xpu_leaked_bytes(device_id);
}

uint64_t tms_xpu_tracked_bytes(int device_id) {
    return TorchMemorySaver::instance().xpu_tracked_bytes(device_id);
}

uint32_t tms_xpu_affected_devices(const char* tag, int* out_device_ids, uint32_t capacity) {
    return TorchMemorySaver::instance().xpu_affected_devices(tag, out_device_ids, capacity);
}

// Test-only: returns the error instead of aborting, so fault injection can observe it.
int tms_xpu_free(const void* ptr) {
    return static_cast<int>(TorchMemorySaver::instance().free(const_cast<void*>(ptr)));
}
#endif
}
