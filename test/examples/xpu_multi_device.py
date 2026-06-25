"""XPU multi-device pause/resume (item: multi-GPU coverage on Intel XPU).

A torch MemPool is bound to the device that is current when it is created, so to
place a pauseable allocation on each device that device must be current at alloc
time (this is also the realistic pattern: one process pins one device per TP
rank). Verifies, via the sysman free-bytes query, that pause frees memory on
BOTH devices and resume re-commits on both.

Run directly:
  python test/examples/xpu_multi_device.py torch
"""

import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_device

_GIB = 1024**3


def run(hook_mode: str):
    assert get_device() == "xpu", "xpu_multi_device is XPU-only"
    assert hook_mode == "torch", "XPU only supports hook_mode='torch'"

    n = torch.xpu.device_count()
    assert n >= 2, f"need >=2 XPU devices, have {n}"

    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()
    cdll = torch_memory_saver._impl._binary_wrapper.cdll

    def free_gib(d):
        torch.xpu.synchronize()
        return cdll.tms_xpu_device_free_bytes(d) / _GIB

    d0, d1 = 0, 1

    torch.xpu.set_device(d0)
    with torch_memory_saver.region(tag="t"):
        a = torch.full((256 * 1024 * 1024,), 1, dtype=torch.float32, device=f"xpu:{d0}")
    torch.xpu.set_device(d1)
    with torch_memory_saver.region(tag="t"):
        b = torch.full((256 * 1024 * 1024,), 1, dtype=torch.float32, device=f"xpu:{d1}")
    torch.xpu.synchronize()
    alloc0, alloc1 = free_gib(d0), free_gib(d1)
    print(f"after alloc:  xpu:{d0}={alloc0:.2f} xpu:{d1}={alloc1:.2f} GiB")

    torch_memory_saver.pause("t")
    pause0, pause1 = free_gib(d0), free_gib(d1)
    print(f"after pause:  xpu:{d0}={pause0:.2f} xpu:{d1}={pause1:.2f} GiB")

    torch_memory_saver.resume("t")
    a.fill_(2.0)
    b.fill_(2.0)
    torch.xpu.synchronize()
    resume0, resume1 = free_gib(d0), free_gib(d1)
    print(f"after resume: xpu:{d0}={resume0:.2f} xpu:{d1}={resume1:.2f} GiB")

    assert (pause0 - alloc0) > 0.8, "device 0 not freed on pause"
    assert (pause1 - alloc1) > 0.8, "device 1 not freed on pause"
    assert (pause0 - resume0) > 0.8, "device 0 not re-committed on resume"
    assert (pause1 - resume1) > 0.8, "device 1 not re-committed on resume"
    assert float(a[0]) == 2.0, "device 0 tensor unusable after resume"
    assert float(b[0]) == 2.0, "device 1 tensor unusable after resume"
    print("xpu_multi_device OK")


if __name__ == "__main__":
    run(hook_mode=sys.argv[1] if len(sys.argv) > 1 else "torch")
