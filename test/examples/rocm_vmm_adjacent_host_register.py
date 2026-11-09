"""A hipHostRegister'ed buffer right after a VMM allocation must not break pause, resume, or free.

RCCL registers small host buffers at runtime, and the kernel can place one at the first free page after
an allocation. HIP's VMM range check counts such a buffer when it starts exactly at the range end and
rejects the range.
"""
import ctypes
import mmap
import sys


_SIZE = 4 << 20
_SCAN_LIMIT = 2 << 20
_MAP_FIXED_NOREPLACE = 0x100000
_MEMCPY_HOST_TO_DEVICE = 1
_MEMCPY_DEVICE_TO_HOST = 2


def _bind(api, name, argtypes, restype=ctypes.c_int):
    fn = getattr(api, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def _assert_round_trip(memcpy, ptr, expected):
    host_write = ctypes.c_ubyte(expected)
    host_read = ctypes.c_ubyte()
    assert memcpy(ptr, ctypes.byref(host_write), 1, _MEMCPY_HOST_TO_DEVICE) == 0
    assert memcpy(ctypes.byref(host_read), ptr, 1, _MEMCPY_DEVICE_TO_HOST) == 0
    assert host_read.value == expected


def _map_first_free_page(host_mmap, start):
    for addr in range(start, start + _SCAN_LIMIT, mmap.PAGESIZE):
        page = host_mmap(
            addr,
            mmap.PAGESIZE,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
        if page == addr:
            return page
    raise AssertionError(f"no free page within {_SCAN_LIMIT} bytes after {hex(start)}")


def run(hook_mode: str):
    assert hook_mode == "preload"

    api = ctypes.CDLL(None)
    hip_malloc = _bind(api, "hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
    hip_memcpy = _bind(
        api, "hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    )
    hip_free = _bind(api, "hipFree", [ctypes.c_void_p])
    hip_set_device = _bind(api, "hipSetDevice", [ctypes.c_int])
    hip_host_register = _bind(
        api, "hipHostRegister", [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    )
    hip_host_unregister = _bind(api, "hipHostUnregister", [ctypes.c_void_p])
    host_mmap = _bind(
        api,
        "mmap",
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long],
        ctypes.c_void_p,
    )
    host_munmap = _bind(api, "munmap", [ctypes.c_void_p, ctypes.c_size_t])
    tms_pause = _bind(api, "tms_pause", [ctypes.c_char_p], None)
    tms_resume = _bind(api, "tms_resume", [ctypes.c_char_p], None)

    assert hip_set_device(0) == 0
    ptr = ctypes.c_void_p()
    assert hip_malloc(ctypes.byref(ptr), _SIZE) == 0
    _assert_round_trip(hip_memcpy, ptr, 0xA5)

    host = _map_first_free_page(host_mmap, ptr.value + _SIZE)
    assert hip_host_register(host, mmap.PAGESIZE, 0) == 0

    tms_pause(None)
    tms_resume(None)
    _assert_round_trip(hip_memcpy, ptr, 0x5A)
    assert hip_free(ptr) == 0

    assert hip_host_unregister(host) == 0
    assert host_munmap(host, mmap.PAGESIZE) == 0


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
