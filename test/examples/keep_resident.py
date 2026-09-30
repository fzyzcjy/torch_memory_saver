"""CUDA allocation residency survives both tagged and global pause/resume."""

import ctypes
import sys

import torch

from torch_memory_saver import torch_memory_saver as tms


def buffer_id(tensor):
    driver = ctypes.CDLL("libcuda.so.1")
    query = driver.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int
    result = ctypes.c_uint64()
    assert query(ctypes.byref(result), 7, tensor.data_ptr()) == 0
    return result.value


def run(hook_mode):
    tms.hook_mode = hook_mode
    size = 64 * 1024**2
    with tms.region(tag="mixed", enable_cpu_backup=True):
        resident = torch.full((size,), 17, dtype=torch.uint8, device="cuda")
        ordinary = torch.full_like(resident, 29)
    # An interior view must protect the containing allocator allocation.
    protected_bytes = tms.keep_resident(resident[1024:2048])
    assert protected_bytes >= size
    assert tms.keep_resident(resident) == protected_bytes
    assert tms.keep_resident(resident[:0]) == 0
    unmanaged = torch.empty(1, device="cuda")
    assert tms.keep_resident(unmanaged) == 0
    resident_id = buffer_id(resident)

    for cycle, tag in enumerate(("mixed", None, "mixed")):
        ordinary_id = buffer_id(ordinary)
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        tms.pause(tag)
        free_paused = torch.cuda.mem_get_info()[0]
        assert free_paused - free_before >= size
        assert buffer_id(resident) == resident_id
        assert torch.all(resident == 17 + cycle).item()
        resident.add_(1)
        torch.cuda.synchronize()
        tms.resume(tag)
        assert buffer_id(resident) == resident_id
        assert buffer_id(ordinary) != ordinary_id
        assert torch.all(resident == 18 + cycle).item()
        assert torch.all(ordinary == 29 + cycle).item()
        ordinary.add_(1)
    print(f"keep_resident passed: hook_mode={hook_mode}, resident_bytes={protected_bytes}")
    check_allocator_free(hook_mode)


def check_allocator_free(hook_mode):
    # Exercise actual allocator free, not del tensor (the caching pool can
    # retain the block). A recycled allocation must not inherit residency.
    lib = tms._impl._binary_wrapper.cdll
    with tms.region(tag="raw"):
        if hook_mode == "preload":
            lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.cudaFree.argtypes = [ctypes.c_void_p]
            ptr = ctypes.c_void_p()
            assert lib.cudaMalloc(ctypes.byref(ptr), 2 * 1024**2) == 0
        else:
            lib.tms_torch_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
            lib.tms_torch_malloc.restype = ctypes.c_void_p
            lib.tms_torch_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
            ptr = ctypes.c_void_p(lib.tms_torch_malloc(2 * 1024**2, 0, None))
        assert lib.tms_keep_resident(ptr) == 2 * 1024**2
        tms.pause("raw")
        tms.resume("raw")
        if hook_mode == "preload":
            assert lib.cudaFree(ptr) == 0
        else:
            lib.tms_torch_free(ptr, 2 * 1024**2, 0, None)
        assert lib.tms_keep_resident(ptr) == 0


if __name__ == "__main__":
    tms.hook_mode = sys.argv[1]
    with tms.region(tag="paused"):
        tensor = torch.empty(1024, device="cuda")
    tms.pause("paused")
    tms.keep_resident(tensor)
    raise AssertionError("keep_resident accepted a paused allocation")
