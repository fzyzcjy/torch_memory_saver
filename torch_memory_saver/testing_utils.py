"""Not to be used by end users, but only for tests of the package itself."""

import torch


def is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_device() -> str:
    """Device string for tests: 'xpu' on Intel GPUs, else 'cuda'."""
    return "xpu" if is_xpu() else "cuda"


def empty_cache():
    if is_xpu():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def _xpu_used_memory(gpu_id=0):
    """Real used device memory on XPU.

    torch.xpu.memory_allocated reflects the allocator's bookkeeping and does NOT
    drop when the memory saver releases physical pages via zeVirtualMemUnmap.
    The native lib exposes a sysman-backed free-bytes query; use it when the
    saver has been initialized, else fall back to torch.xpu.mem_get_info.
    """
    try:
        from torch_memory_saver import torch_memory_saver as _saver

        impl = getattr(_saver, "_impl", None)
        cdll = getattr(getattr(impl, "_binary_wrapper", None), "cdll", None)
        if cdll is not None and hasattr(cdll, "tms_xpu_device_free_bytes"):
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
