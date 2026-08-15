#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct DiskBackupSlot {
    std::string path;
};

// Spills paused GPU allocations to per-allocation files, streaming through one
// shared staging buffer. dir_ must be a real disk mount, not tmpfs.
class DiskBackend {
public:
    DiskBackend(std::string dir, size_t chunk_bytes);

    void set_dir(const std::string& dir) { dir_ = dir; }

    void offload(void* gpu_ptr, size_t size, DiskBackupSlot& slot);
    void onload(void* gpu_ptr, size_t size, DiskBackupSlot& slot);
    void release(DiskBackupSlot& slot);

private:
    void ensure_staging_buf_();
    void warn_once_if_not_real_disk_();
    int create_slot_file_(DiskBackupSlot& slot, size_t size);
    int open_slot_file_(const DiskBackupSlot& slot);

    std::string dir_;
    size_t chunk_bytes_;
    std::vector<uint8_t> staging_buf_;
    bool checked_fs_ = false;
};

std::string compute_disk_backup_dir_from_env();
size_t compute_disk_chunk_bytes_from_env();
