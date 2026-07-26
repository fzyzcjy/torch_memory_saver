"""Not to be used by end users, but only for tests of the package itself."""

import torch

from torch_memory_saver.utils import is_xpu


def get_device() -> str:
    """Device string for tests: 'xpu' on Intel GPUs, else 'cuda'."""
    return "xpu" if is_xpu() else "cuda"


def empty_cache():
    if is_xpu():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def _xpu_used_memory(gpu_id=0):
    """Used device memory on XPU, for delta-based pause/resume test assertions.

    Prefers the saver's own committed physical bytes (tms_xpu_committed_bytes) --
    driver-independent, drops to 0 on pause() -- since torch/sysman free-byte
    telemetry is frozen on newer drivers. Falls back to sysman-derived used bytes
    on older builds; the two return different quantities (saver-committed vs
    whole-device), fine for deltas but not a reliable absolute usage figure.
    """
    try:
        from torch_memory_saver import torch_memory_saver as _saver

        # Idempotent; required because committed-bytes reads saver state.
        _saver._ensure_initialized()
        cdll = _saver._impl._binary_wrapper.cdll
        if hasattr(cdll, "tms_xpu_committed_bytes"):
            return int(cdll.tms_xpu_committed_bytes(gpu_id))
        if hasattr(cdll, "tms_xpu_device_free_bytes"):
            free = int(cdll.tms_xpu_device_free_bytes(gpu_id))
            _, total = torch.xpu.mem_get_info(gpu_id)
            return total - free
    except Exception:
        pass
    free, total = torch.xpu.mem_get_info(gpu_id)
    return total - free


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message."""
    if is_xpu():
        mem = _xpu_used_memory(gpu_id)
    elif torch.version.hip:
        # ROCm: amd-smi (device_memory_used) has delays, use mem_get_info for real-time tracking
        # see https://github.com/ROCm/amdsmi/issues/175 for details
        free, total = torch.cuda.mem_get_info(gpu_id)
        mem = total - free
    else:
        mem = torch.cuda.device_memory_used(gpu_id)
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem
