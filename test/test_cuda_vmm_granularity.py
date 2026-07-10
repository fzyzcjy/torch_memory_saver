import ctypes
import os
import subprocess
import sys

import pytest
import torch

from torch_memory_saver import configure_subprocess


_UNALIGNED_SIZE = 6912

_CHILD = f"""
import ctypes

size = {_UNALIGNED_SIZE}
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaSetDevice.argtypes = [ctypes.c_int]
cudart.cudaSetDevice.restype = ctypes.c_int
cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
cudart.cudaMemset.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

assert cudart.cudaSetDevice(0) == 0

# Resolve the symbols interposed by torch-memory-saver's official preload hook.
process = ctypes.CDLL(None)
cuda_malloc = process.cudaMalloc
cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda_malloc.restype = ctypes.c_int
cuda_free = process.cudaFree
cuda_free.argtypes = [ctypes.c_void_p]
cuda_free.restype = ctypes.c_int
tms_pause = process.tms_pause
tms_pause.argtypes = [ctypes.c_char_p]
tms_pause.restype = None
tms_resume = process.tms_resume
tms_resume.argtypes = [ctypes.c_char_p]
tms_resume.restype = None

ptr = ctypes.c_void_p()
assert cuda_malloc(ctypes.byref(ptr), size) == 0
assert ptr.value is not None
assert cudart.cudaMemset(ptr, 0xA5, size) == 0
assert cudart.cudaDeviceSynchronize() == 0

tms_pause(None)
tms_resume(None)

host = (ctypes.c_ubyte * size)()
cuda_memcpy_device_to_host = 2
assert cudart.cudaMemcpy(
    ctypes.cast(host, ctypes.c_void_p),
    ptr,
    size,
    cuda_memcpy_device_to_host,
) == 0
assert cudart.cudaDeviceSynchronize() == 0
assert all(value == 0xA5 for value in host)
assert cuda_free(ptr) == 0
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA VMM test requires a CUDA GPU",
)
def test_preload_hook_handles_unaligned_vmm_allocation():
    env = os.environ.copy()
    env["TMS_INIT_ENABLE"] = "1"
    env["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    with configure_subprocess():
        env["LD_PRELOAD"] = os.environ["LD_PRELOAD"]
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD],
            env=env,
            capture_output=True,
            text=True,
        )

    assert completed.returncode == 0, (
        f"child exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
