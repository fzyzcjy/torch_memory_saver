#include "disk_backend.h"
#include "macro.h"
#include "utils.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/vfs.h>

#ifndef TMPFS_MAGIC
#define TMPFS_MAGIC 0x01021994
#endif
#ifndef RAMFS_MAGIC
#define RAMFS_MAGIC 0x858458f6
#endif

namespace {

void pwrite_all(int fd, const void* buf, size_t n, off_t offset) {
    size_t done = 0;
    while (done < n) {
        ssize_t w = pwrite(fd, (const uint8_t*) buf + done, n - done, offset + (off_t) done);
        if (w < 0) {
            if (errno == EINTR) continue;
            SIMPLE_CHECK(false, (std::string("pwrite to disk backup file failed: ") + strerror(errno)).c_str());
        }
        SIMPLE_CHECK(w > 0, "pwrite to disk backup file returned 0");
        done += (size_t) w;
    }
}

void pread_all(int fd, void* buf, size_t n, off_t offset) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pread(fd, (uint8_t*) buf + done, n - done, offset + (off_t) done);
        if (r < 0) {
            if (errno == EINTR) continue;
            SIMPLE_CHECK(false, (std::string("pread from disk backup file failed: ") + strerror(errno)).c_str());
        }
        SIMPLE_CHECK(r > 0, "pread from disk backup file returned 0 (unexpected EOF)");
        done += (size_t) r;
    }
}

void stream_chunked(int fd, void* gpu_ptr, size_t size, void* staging, size_t chunk, bool to_disk) {
    size_t offset = 0;
    while (offset < size) {
        size_t n = std::min(chunk, size - offset);
        if (to_disk) {
            CUDA_ERROR_CHECK(cudaMemcpy(staging, (uint8_t*) gpu_ptr + offset, n, cudaMemcpyDeviceToHost));
            pwrite_all(fd, staging, n, (off_t) offset);
        } else {
            pread_all(fd, staging, n, (off_t) offset);
            CUDA_ERROR_CHECK(cudaMemcpy((uint8_t*) gpu_ptr + offset, staging, n, cudaMemcpyHostToDevice));
        }
        offset += n;
    }
}

// Drop this file's pages from the page cache (no effect on tmpfs).
void drop_page_cache(int fd, size_t size) {
    fsync(fd);
    posix_fadvise(fd, 0, (off_t) size, POSIX_FADV_DONTNEED);
}

}  // namespace

std::string compute_disk_backup_dir_from_env() {
    const char* dir = std::getenv("TMS_DISK_BACKUP_DIR");
    return (dir != nullptr) ? std::string(dir) : std::string("/tmp");
}

size_t compute_disk_chunk_bytes_from_env() {
    const char* chunk_mb = std::getenv("TMS_DISK_BACKUP_CHUNK_MB");
    size_t mb = (chunk_mb != nullptr) ? std::strtoull(chunk_mb, nullptr, 10) : 256;
    if (mb == 0) mb = 256;
    return mb * 1024ull * 1024ull;
}

DiskBackend::DiskBackend(std::string dir, size_t chunk_bytes)
    : dir_(std::move(dir)), chunk_bytes_(chunk_bytes) {}

void DiskBackend::ensure_staging_buf_() {
    if (staging_buf_ == nullptr) {
        CUDA_ERROR_CHECK(cudaMallocHost(&staging_buf_, chunk_bytes_));
        SIMPLE_CHECK(staging_buf_ != nullptr, "failed to allocate pinned disk staging buffer");
    }
}

void DiskBackend::warn_once_if_not_real_disk_() {
    if (checked_fs_) return;
    checked_fs_ = true;
    struct statfs sfs;
    if (statfs(dir_.c_str(), &sfs) == 0 &&
        (sfs.f_type == TMPFS_MAGIC || sfs.f_type == RAMFS_MAGIC)) {
        std::cerr << "[torch_memory_saver.cpp] WARNING: disk backup dir '" << dir_
                  << "' is tmpfs/ramfs; backups live in RAM and do not bound host memory. "
                  << "Point TMS_DISK_BACKUP_DIR at a real disk mount." << std::endl;
    }
}

int DiskBackend::create_slot_file_(DiskBackupSlot& slot, size_t size) {
    warn_once_if_not_real_disk_();
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);

    // mkostemp: atomic unique O_RDWR|O_CREAT|O_EXCL create, mode 0600.
    std::string tmpl = dir_ + "/tms_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = mkostemp(buf.data(), O_CLOEXEC);
    SIMPLE_CHECK(fd >= 0, (std::string("failed to create disk backup file under ") + dir_
                           + ": " + strerror(errno)).c_str());
    slot.path.assign(buf.data());

    int fret = posix_fallocate(fd, 0, (off_t) size);
    if (fret != 0) {
        if (fret == EOPNOTSUPP || fret == ENOTSUP || fret == EINVAL) {
            SIMPLE_CHECK(ftruncate(fd, (off_t) size) == 0,
                (std::string("ftruncate fallback failed on ") + slot.path + ": " + strerror(errno)).c_str());
        } else {
            SIMPLE_CHECK(false, (std::string("posix_fallocate failed on ") + slot.path + ": " + strerror(fret)).c_str());
        }
    }
    return fd;
}

int DiskBackend::open_slot_file_(const DiskBackupSlot& slot) {
    int fd = open(slot.path.c_str(), O_RDWR | O_CLOEXEC);
    SIMPLE_CHECK(fd >= 0, (std::string("failed to reopen disk backup file ") + slot.path + ": " + strerror(errno)).c_str());
    return fd;
}

void DiskBackend::offload(void* gpu_ptr, size_t size, DiskBackupSlot& slot) {
    ensure_staging_buf_();
    int fd = slot.path.empty() ? create_slot_file_(slot, size) : open_slot_file_(slot);
    stream_chunked(fd, gpu_ptr, size, staging_buf_, chunk_bytes_, /*to_disk=*/true);
    drop_page_cache(fd, size);
    close(fd);
}

void DiskBackend::onload(void* gpu_ptr, size_t size, DiskBackupSlot& slot) {
    ensure_staging_buf_();
    SIMPLE_CHECK(!slot.path.empty(), "disk backup missing on resume (was it ever paused?)");
    int fd = open_slot_file_(slot);
    stream_chunked(fd, gpu_ptr, size, staging_buf_, chunk_bytes_, /*to_disk=*/false);
    drop_page_cache(fd, size);
    close(fd);
}

void DiskBackend::release(DiskBackupSlot& slot) {
    if (!slot.path.empty()) {
        unlink(slot.path.c_str());
        slot.path.clear();
    }
}
