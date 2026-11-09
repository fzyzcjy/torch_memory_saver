"""Not to be used by end users, but only for tests of the package itself."""

import torch

from torch_memory_saver.utils import is_musa, is_xpu


def get_device_module():
    """Device module for tests: torch.xpu on Intel GPUs, torch.musa on MUSA, else torch.cuda."""
    if is_xpu():
        return torch.xpu
    if is_musa():
        return torch.musa
    return torch.cuda


def get_device() -> str:
    """Device string for tests: 'xpu' on Intel GPUs, 'musa' on MUSA, else 'cuda'."""
    if is_xpu():
        return "xpu"
    if is_musa():
        return "musa"
    return "cuda"


def get_graph_class():
    """Graph capture class for tests (MUSA exposes MUSAGraph instead of CUDAGraph)."""
    if is_musa():
        return torch.musa.MUSAGraph
    return torch.cuda.CUDAGraph


def empty_cache():
    get_device_module().empty_cache()


def _xpu_used_memory(gpu_id=0):
    """Used device memory on XPU, for delta-based pause/resume test assertions."""
    try:
        from torch_memory_saver import torch_memory_saver as _saver

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


def _musa_used_memory(gpu_id=0):
    free, total = torch.musa.mem_get_info(gpu_id)
    return total - free


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message."""
    if is_xpu():
        mem = _xpu_used_memory(gpu_id)
    elif is_musa():
        mem = _musa_used_memory(gpu_id)
    elif torch.version.hip:
        # ROCm: amd-smi (device_memory_used) has delays, use mem_get_info for real-time tracking
        # see https://github.com/ROCm/amdsmi/issues/175 for details
        free, total = torch.cuda.mem_get_info(gpu_id)
        mem = total - free
    else:
        mem = torch.cuda.device_memory_used(gpu_id)
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem
