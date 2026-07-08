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
    """Used device memory on XPU, for pause/resume test assertions.

    torch.xpu.memory_allocated reflects the allocator's bookkeeping and does NOT
    drop when the memory saver releases physical pages via zeVirtualMemUnmap. On
    newer Intel GPU drivers, device free-byte telemetry (sysman
    zesMemoryGetState().free and torch.xpu.mem_get_info().free) is
    frozen/deprecated and never moves.

    Preferred metric: the saver's own committed physical bytes
    (tms_xpu_committed_bytes), which is driver-independent, drops to 0 on pause()
    and is restored on resume(). Falls back to the sysman-derived used bytes only
    when the committed-bytes symbol is unavailable (older builds).

    NOTE: the two paths return DIFFERENT quantities -- saver-committed bytes vs
    whole-device used bytes. This is fine for the delta-based pause/resume
    assertions here, but the value is not a reliable absolute device-usage figure.
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
