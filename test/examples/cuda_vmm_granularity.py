import ctypes
import sys


_UNALIGNED_SIZE = 6912
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _bind(api, name, argtypes, restype=ctypes.c_int):
    fn = getattr(api, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def _assert_round_trip(cuda_memcpy, ptr, expected):
    host_write = ctypes.c_ubyte(expected)
    host_read = ctypes.c_ubyte()
    assert (
        cuda_memcpy(
            ptr,
            ctypes.byref(host_write),
            ctypes.sizeof(host_write),
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        == 0
    )
    assert (
        cuda_memcpy(
            ctypes.byref(host_read),
            ptr,
            ctypes.sizeof(host_read),
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        == 0
    )
    assert host_read.value == expected


def run(hook_mode: str):
    assert hook_mode == "preload"

    api = ctypes.CDLL(None)
    cuda_malloc = _bind(
        api, "cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    )
    cuda_memcpy = _bind(
        api,
        "cudaMemcpy",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    )
    cuda_free = _bind(api, "cudaFree", [ctypes.c_void_p])
    cuda_set_device = _bind(api, "cudaSetDevice", [ctypes.c_int])
    tms_pause = _bind(api, "tms_pause", [ctypes.c_char_p], None)
    tms_resume = _bind(api, "tms_resume", [ctypes.c_char_p], None)

    assert cuda_set_device(0) == 0
    ptr = ctypes.c_void_p()
    assert cuda_malloc(ctypes.byref(ptr), _UNALIGNED_SIZE) == 0
    assert ptr.value is not None
    _assert_round_trip(cuda_memcpy, ptr, 0xA5)
    tms_pause(None)
    tms_resume(None)
    _assert_round_trip(cuda_memcpy, ptr, 0x5A)
    assert cuda_free(ptr) == 0


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
