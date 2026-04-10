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

    size_t chunk_size() {
        return chunk_size_;
    }

    void set_chunk_size(size_t value) {
        chunk_size_ = value;
    }

private:
    std::optional<bool> is_interesting_region_;
    std::optional<bool> enable_cpu_backup_;
    size_t chunk_size_ = 0;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup(), thread_local_config.chunk_size());
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
    CUDA_ERROR_CHECK(TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup(), thread_local_config.chunk_size()));
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] entrypoint::tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
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

void set_memory_margin_bytes(uint64_t value) {
    TorchMemorySaver::instance().set_memory_margin_bytes(value);
}

size_t tms_get_chunk_size() {
    return thread_local_config.chunk_size();
}

void tms_set_chunk_size(size_t chunk_size) {
    thread_local_config.set_chunk_size(chunk_size);
}

size_t tms_get_num_chunks_for_tag(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    return TorchMemorySaver::instance().get_num_chunks(tag_str);
}

void tms_get_chunk_states(const char* tag, uint8_t* out_active, size_t num_chunks) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().get_chunk_states(tag_str, out_active, num_chunks);
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}

void tms_pause_chunks(const char* tag, const size_t* chunk_indices, size_t num_indices) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause_chunks(tag_str, chunk_indices, num_indices);
}

void tms_resume_chunks(const char* tag, const size_t* chunk_indices, size_t num_indices) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume_chunks(tag_str, chunk_indices, num_indices);
}

uint8_t* tms_get_cpu_backup_pointer(const uint8_t* gpu_ptr, uint64_t size) {
    return TorchMemorySaver::instance().get_cpu_backup_pointer(gpu_ptr, size);
}
}
