#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <stdexcept>
#include <string>

// ----------------------------------------------- threadlocal configs --------------------------------------------------

struct ThreadLocalConfig {
    bool is_interesting_region_ = false;
    std::string current_tag_ = "default";
    bool enable_cpu_backup_ = false;
};
static thread_local ThreadLocalConfig thread_local_config;

static std::string last_error_message;
static std::mutex error_mutex;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region_) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup_);
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    if (thread_local_config.is_interesting_region_) {
        return TorchMemorySaver::instance().free(ptr);
    } else {
        return APIForwarder::call_real_cuda_free(ptr);
    }
}
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] tms_torch_malloc "
              << " size=" << size << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region_, "only support interesting region");
    void *ptr;
    TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup_);
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region_, "only support interesting region");
    TorchMemorySaver::instance().free(ptr);
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.is_interesting_region_ = is_interesting_region;
}

bool tms_get_interesting_region() {
    return thread_local_config.is_interesting_region_;
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.enable_cpu_backup_ = enable_cpu_backup;
}

int tms_pause(const char* tag) {
    try {
        std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
        TorchMemorySaver::instance().pause(tag_str);
        return 0;  // Success
    } catch (const std::exception& e) {
        const std::lock_guard<std::mutex> lock(error_mutex);
        last_error_message = e.what();
        return -1;  // Error
    }
}

int tms_resume(const char* tag) {
    try {
        std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
        TorchMemorySaver::instance().resume(tag_str);
        return 0;  // Success
    } catch (const std::exception& e) {
        const std::lock_guard<std::mutex> lock(error_mutex);
        last_error_message = e.what();
        return -1;  // Error
    }
}

const char* tms_get_last_error() {
    const std::lock_guard<std::mutex> lock(error_mutex);
    return last_error_message.c_str();
}
}