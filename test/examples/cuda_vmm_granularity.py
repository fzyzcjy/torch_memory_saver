import ctypes
import sys
from ctypes.util import find_library

import torch

from torch_memory_saver.testing_utils import get_device


_UNALIGNED_SIZE = 6912
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
    assert (
        memcpy(
            ptr,
            ctypes.byref(host_write),
            ctypes.sizeof(host_write),
            _MEMCPY_HOST_TO_DEVICE,
        )
        == 0
    )
    assert (
        memcpy(
            ctypes.byref(host_read),
            ptr,
            ctypes.sizeof(host_read),
            _MEMCPY_DEVICE_TO_HOST,
        )
        == 0
    )
    assert host_read.value == expected


def run(hook_mode: str):
    assert hook_mode == "preload"

    device_type = get_device()
    prefix = "hip" if torch.version.hip else "musa" if device_type == "musa" else "cuda"
    api = ctypes.CDLL(None)
    # torch loads libmusart without global visibility; memcpy/set_device are not
    # exported by the TMS preload hook, so bind the runtime directly on MUSA.
    runtime_api = (
        ctypes.CDLL(find_library("musart") or "libmusart.so")
        if prefix == "musa"
        else api
    )
    gpu_malloc = _bind(
        api, f"{prefix}Malloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    )
    gpu_memcpy = _bind(
        runtime_api,
        f"{prefix}Memcpy",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    )
    gpu_free = _bind(api, f"{prefix}Free", [ctypes.c_void_p])
    gpu_set_device = _bind(runtime_api, f"{prefix}SetDevice", [ctypes.c_int])
    tms_pause = _bind(api, "tms_pause", [ctypes.c_char_p], None)
    tms_resume = _bind(api, "tms_resume", [ctypes.c_char_p], None)

    assert gpu_set_device(0) == 0
    ptr = ctypes.c_void_p()
    assert gpu_malloc(ctypes.byref(ptr), _UNALIGNED_SIZE) == 0
    assert ptr.value is not None
    _assert_round_trip(gpu_memcpy, ptr, 0xA5)
    tms_pause(None)
    tms_resume(None)
    _assert_round_trip(gpu_memcpy, ptr, 0x5A)
    assert gpu_free(ptr) == 0

    ptr_free = ctypes.c_void_p()
    ptr_keep = ctypes.c_void_p()
    assert gpu_malloc(ctypes.byref(ptr_free), _UNALIGNED_SIZE) == 0
    assert gpu_malloc(ctypes.byref(ptr_keep), _UNALIGNED_SIZE) == 0
    _assert_round_trip(gpu_memcpy, ptr_keep, 0xA5)
    tms_pause(None)
    assert gpu_free(ptr_free) == 0
    tms_resume(None)
    _assert_round_trip(gpu_memcpy, ptr_keep, 0x5A)
    assert gpu_free(ptr_keep) == 0


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
