"""Multi-device pause/resume in hook_mode="torch".

A torch MemPool is bound to the device that is current when it is created, so a
pauseable allocation must be made with its target device current (this is also
the realistic pattern: one process pins one device per TP rank). Verifies that
pause frees memory on BOTH devices and resume re-commits on both.

Unlike ``multi_device.py`` (which allocates on a non-current device inside a
single region -- only the preload/global-malloc hook can capture that), this
pins each device at alloc time, so it works with the pool-scoped torch-mode
allocator on any platform.

Run directly:
  python test/examples/multi_device_torch_mode.py torch
"""

import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory, get_device

_ALLOC_BYTES = 256 * 1024 * 1024 * 4  # 256M float32 = 1 GiB
_FREED_GIB = 0.8  # expected drop/restore, with slack for alignment/noise


def run(hook_mode: str):
    assert hook_mode == "torch", "multi-device torch-mode test requires hook_mode='torch'"

    device = get_device()  # "cuda" or "xpu"
    mod = torch.get_device_module()  # torch.cuda / torch.xpu
    if mod.device_count() < 2:
        print(f"skip: need >=2 {device} devices, have {mod.device_count()}")
        return

    torch_memory_saver.hook_mode = hook_mode

    def used_gib(d):
        mod.synchronize(d)
        return get_and_print_gpu_memory(f"dev{d}", gpu_id=d) / 1024**3

    d0, d1 = 0, 1

    # Pin each device current at alloc time so its MemPool binds to it.
    mod.set_device(d0)
    with torch_memory_saver.region(tag="t"):
        a = torch.full((_ALLOC_BYTES // 4,), 1.0, dtype=torch.float32, device=f"{device}:{d0}")
    mod.set_device(d1)
    with torch_memory_saver.region(tag="t"):
        b = torch.full((_ALLOC_BYTES // 4,), 1.0, dtype=torch.float32, device=f"{device}:{d1}")
    alloc0, alloc1 = used_gib(d0), used_gib(d1)

    torch_memory_saver.pause("t")
    pause0, pause1 = used_gib(d0), used_gib(d1)

    torch_memory_saver.resume("t")
    a.fill_(2.0)
    b.fill_(2.0)
    resume0, resume1 = used_gib(d0), used_gib(d1)

    assert (alloc0 - pause0) > _FREED_GIB, "device 0 not freed on pause"
    assert (alloc1 - pause1) > _FREED_GIB, "device 1 not freed on pause"
    assert (resume0 - pause0) > _FREED_GIB, "device 0 not re-committed on resume"
    assert (resume1 - pause1) > _FREED_GIB, "device 1 not re-committed on resume"
    assert float(a[0]) == 2.0, "device 0 tensor unusable after resume"
    assert float(b[0]) == 2.0, "device 1 tensor unusable after resume"
    print("multi_device_torch_mode OK")


if __name__ == "__main__":
    run(hook_mode=sys.argv[1] if len(sys.argv) > 1 else "torch")
