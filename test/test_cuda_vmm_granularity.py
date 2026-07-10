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
api = ctypes.CDLL(None)

def bind(name, argtypes, restype=ctypes.c_int):
    fn = getattr(api, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn

cuda_malloc = bind("cudaMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
cuda_free = bind("cudaFree", [ctypes.c_void_p])
cuda_set_device = bind("cudaSetDevice", [ctypes.c_int])
tms_pause = bind("tms_pause", [ctypes.c_char_p], None)
tms_resume = bind("tms_resume", [ctypes.c_char_p], None)

assert cuda_set_device(0) == 0
ptr = ctypes.c_void_p()
assert cuda_malloc(ctypes.byref(ptr), size) == 0
assert ptr.value is not None
tms_pause(None)
tms_resume(None)
assert cuda_free(ptr) == 0
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA VMM test requires a CUDA GPU",
)
def test_unaligned_cuda_malloc_survives_pause_resume():
    env = os.environ.copy()
    env["TMS_INIT_ENABLE"] = "1"

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
