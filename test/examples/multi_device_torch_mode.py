"""Multi-device pause/resume in hook_mode="torch".

A torch MemPool binds to the device that is current when it is created, and
use_mem_pool only redirects allocations on that device. So each pauseable
allocation must be made with its target device current (the realistic pattern:
one process pins one device per TP rank).

This protects the fix that keys _mem_pools by device: without it, the pool
created for device 0 is reused for device 1, the device-1 allocation bypasses
the custom allocator, and pause() fails to free it.

Unlike multi_device.py (which allocates on a non-current device inside one
region -- only the global-malloc preload hook can capture that), this pins each
device at alloc time and therefore works in hook_mode="torch".

Run directly:
  python test/examples/multi_device_torch_mode.py torch
"""

import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory

_ALLOC = 100_000_000  # ~100 MB uint8
_FREED = 80_000_000   # expected drop/restore, with slack


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    if torch.cuda.device_count() < 2:
        print(f"skip: need >=2 CUDA devices, have {torch.cuda.device_count()}")
        return

    def used(d):
        torch.cuda.synchronize(d)
        return get_and_print_gpu_memory(f"dev{d}", gpu_id=d)

    # Pin each device current at alloc time so its MemPool binds to it.
    torch.cuda.set_device(0)
    with torch_memory_saver.region(tag="t"):
        a = torch.full((_ALLOC,), 1, dtype=torch.uint8, device="cuda:0")
    torch.cuda.set_device(1)
    with torch_memory_saver.region(tag="t"):
        b = torch.full((_ALLOC,), 1, dtype=torch.uint8, device="cuda:1")
    assert a.device == torch.device("cuda:0"), a.device
    assert b.device == torch.device("cuda:1"), b.device
    alloc0, alloc1 = used(0), used(1)

    torch_memory_saver.pause("t")
    pause0, pause1 = used(0), used(1)

    torch_memory_saver.resume("t")
    resume0, resume1 = used(0), used(1)

    # Both devices must free on pause and re-commit on resume. The device-1
    # assertions are what fail if the pool is not keyed by device.
    assert (alloc0 - pause0) >= _FREED, "device 0 not freed on pause"
    assert (alloc1 - pause1) >= _FREED, "device 1 not freed on pause"
    assert (resume0 - pause0) >= _FREED, "device 0 not re-committed on resume"
    assert (resume1 - pause1) >= _FREED, "device 1 not re-committed on resume"
    print("multi_device_torch_mode OK")


if __name__ == "__main__":
    run(hook_mode=sys.argv[1] if len(sys.argv) > 1 else "torch")
