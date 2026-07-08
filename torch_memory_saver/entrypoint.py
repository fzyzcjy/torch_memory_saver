import atexit
import ctypes

import numpy as np
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
import torch

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase, HookMode
from .utils import is_xpu

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"


class TorchMemorySaver:
    def __init__(self):
        self._impl_ctor_kwargs = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(self, tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
               enable_disk_backup: bool = False):
        """Context manager for memory saving with optional tag.

        enable_disk_backup spills paused memory to files instead of a pinned CPU
        buffer; mutually exclusive with enable_cpu_backup. The target directory is
        process-global (TMS_DISK_BACKUP_DIR or set_disk_backup_dir()), not
        region-scoped, and must be a real disk mount (not tmpfs).
        """
        self._ensure_initialized()
        assert not (enable_cpu_backup and enable_disk_backup), \
            "enable_cpu_backup and enable_disk_backup are mutually exclusive"
        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup,
                               enable_disk_backup=enable_disk_backup):
            yield

    @contextmanager
    def cuda_graph(
            self,
            cuda_graph, pool=None, stream=None, capture_error_mode='global',
            tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
    ):
        """Similar to `torch.cuda.graph`, but ensures memory in it to be pauseable."""
        self._ensure_initialized()
        with self._impl.cuda_graph(
                cuda_graph=cuda_graph,
                pool=pool, stream=stream, capture_error_mode=capture_error_mode,
                tag=tag, enable_cpu_backup=enable_cpu_backup,
        ):
            yield

    @contextmanager
    def disable(self):
        self._ensure_initialized()
        with self._impl.disable():
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        self._ensure_initialized()
        self._impl.resume(tag=tag)

    # for compatibility
    @property
    def enabled(self):
        return True

    @property
    def hook_mode(self):
        raise AttributeError

    @hook_mode.setter
    def hook_mode(self, hook_mode: HookMode):
        assert self._impl_ctor_kwargs is not None, "Cannot configure after initialization"
        self._impl_ctor_kwargs["hook_mode"] = hook_mode

    @property
    def memory_margin_bytes(self):
        raise NotImplementedError("Only setter is supported")

    @memory_margin_bytes.setter
    def memory_margin_bytes(self, value: int):
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.set_memory_margin_bytes(value)

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        self._ensure_initialized()
        return self._impl.get_cpu_backup(x, zero_copy=zero_copy)

    def set_disk_backup_dir(self, path: str):
        """Set the directory for disk backup files (created if needed)."""
        self._ensure_initialized()
        os.makedirs(path, exist_ok=True)
        self._impl._binary_wrapper.cdll.tms_set_disk_backup_dir(path.encode("utf-8"))

    def _ensure_initialized(self):
        if self._impl is not None:
            return
        self._impl = _TorchMemorySaverImpl(**self._impl_ctor_kwargs)
        del self._impl_ctor_kwargs


class _TorchMemorySaverImpl:
    def __init__(self, hook_mode: HookMode = "preload"):
        self._is_xpu = is_xpu()
        if self._is_xpu:
            assert hook_mode == "torch", (
                "XPU only supports hook_mode='torch' (in-process pluggable "
                "allocator); preload/LD_PRELOAD is CUDA-only."
            )
        self._hook_mode = hook_mode
        self._hook_util = HookUtilBase.create(hook_mode=hook_mode)
        self._binary_wrapper = BinaryWrapper(path_binary=self._hook_util.get_path_binary())

        # torch.cuda / torch.xpu share the MemPool / use_mem_pool / current_device
        # / synchronize API; resolve the right module once instead of branching
        # per platform.
        self._device_module = torch.get_device_module()

        if self._is_xpu:
            # Prewarm per-device SYCL contexts before registering the allocator.
            if hasattr(self._binary_wrapper.cdll, "tms_xpu_prewarm_devices"):
                self._binary_wrapper.cdll.tms_xpu_prewarm_devices(
                    self._device_module.device_count()
                )

        self._mem_pools = defaultdict(
            lambda: self._device_module.MemPool(allocator=self._hook_util.get_allocator())
        )
        self._use_mem_pool = self._device_module.use_mem_pool

        _sanity_checks()
        if torch.version.hip or self._is_xpu:
            # HIP/SYCL runtime static dtors may run before MemPool's at exit
            # (destruction-order fiasco); free pools via atexit while it's alive.
            atexit.register(self._mem_pools.clear)

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool, enable_disk_backup: bool):
        if self._is_xpu and enable_disk_backup:
            # Unsupported on XPU: disk spill goes through cudaMallocHost/cudaMemcpy
            # (csrc/disk_backend.cpp) with no Level Zero path wired up. The C-ABI
            # malloc also rejects it; raise here for a clean error rather than a
            # process-fatal abort. Use enable_cpu_backup instead.
            raise NotImplementedError(
                "TorchMemorySaver disk backup (enable_disk_backup=True) is not "
                "supported on Intel XPU; use enable_cpu_backup=True instead."
            )
        # For hook_mode=preload, we need this b/c https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # (For hook_mode=torch we may not need it, but currently our primary usage is hook_mode=preload, thus we do this for simplicity)
        #
        # A MemPool is bound to the current device at creation; key by device so
        # multi-device processes (e.g. TP ranks) don't reuse another's pool.
        key = (tag, enable_cpu_backup, enable_disk_backup, self._current_device())
        mem_pool = self._mem_pools[key]
        with self._use_mem_pool(mem_pool):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup,
                                          enable_disk_backup=enable_disk_backup):
                yield

    def _current_device(self) -> int:
        return self._device_module.current_device()

    @contextmanager
    def cuda_graph(self, cuda_graph, pool, stream, capture_error_mode, tag: str, enable_cpu_backup: bool):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        with torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def _with_region_config(self, tag: str, enable_cpu_backup: bool, enable_disk_backup: bool = False):
        cdll = self._binary_wrapper.cdll
        orig_tag = cdll.tms_get_current_tag().decode("utf-8")
        orig_interesting_region = cdll.tms_get_interesting_region()
        orig_enable_cpu_backup = cdll.tms_get_enable_cpu_backup()
        orig_enable_disk_backup = cdll.tms_get_enable_disk_backup()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True,
                                        enable_cpu_backup=enable_cpu_backup,
                                        enable_disk_backup=enable_disk_backup)
        try:
            yield
        finally:
            assert cdll.tms_get_interesting_region()
            assert cdll.tms_get_enable_cpu_backup() == enable_cpu_backup
            assert cdll.tms_get_enable_disk_backup() == enable_disk_backup
            assert cdll.tms_get_current_tag().decode("utf-8") == tag
            self._binary_wrapper.set_config(
                tag=orig_tag,
                interesting_region=orig_interesting_region,
                enable_cpu_backup=orig_enable_cpu_backup,
                enable_disk_backup=orig_enable_disk_backup,
            )

    @contextmanager
    def disable(self, dispose_mem_pool_after_use: bool = True):
        assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
        assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        try:
            if self._is_xpu:
                # XPU torch-mode: interesting_region=False already routes every
                # allocation to the passthrough path, so no separate pool is
                # needed (and nesting torch.xpu.use_mem_pool can crash on XPU).
                yield
            else:
                # We can either reuse the pool or delete it immediately, and we implement the latter currently since Slime uses it.
                # About why we need a pool: https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
                pool = torch.cuda.MemPool()
                with torch.cuda.use_mem_pool(pool):
                    yield
                del pool
        finally:
            self._binary_wrapper.cdll.tms_set_interesting_region(True)

    def pause(self, tag: Optional[str]):
        if self._is_xpu:
            # Must not unmap physical pages while kernels are still touching
            # them: zeVirtualMemUnmap on in-flight pages hangs the device, so
            # drain the device first. We sync the CURRENT device only; like the
            # CUDA backend, cross-device synchronization is the caller's job.
            # Call pause()/resume() with the region's device current -- the
            # natural one-process-per-rank (one device) deployment already does.
            self._device_module.synchronize()
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes)
        if self._is_xpu:
            # After remapping, settle the current device's stream so the driver
            # finishes any paging/TLB work before user kernels touch the remapped
            # addresses. Current device only (see pause()).
            self._device_module.synchronize()

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        assert x.is_cuda or x.is_xpu, f"{x.device=}"
        assert x.is_contiguous(), f"{x.shape=} {x.stride()=} {x.dtype=}"

        nbytes = x.nbytes
        gpu_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_uint8))
        cpu_ptr = self._binary_wrapper.cdll.tms_get_cpu_backup_pointer(gpu_ptr, nbytes)
        if not cpu_ptr:
            return None

        np_untyped = np.ctypeslib.as_array(cpu_ptr, shape=(nbytes,))
        assert np_untyped.dtype == np.uint8, f"{np_untyped.dtype=} {np_untyped.shape=}"

        ans_untyped = torch.from_numpy(np_untyped)
        ans = ans_untyped.view(x.dtype).view(x.shape)

        # For simplicity and safety
        if not zero_copy:
            ans = ans.clone()

        assert ans.device == torch.device("cpu"), f"{ans.device=}"
        assert ans.dtype == x.dtype, f"{ans.dtype=} {x.dtype=}"
        assert ans.shape == x.shape, f"{ans.shape=} {x.shape=}"
        assert ans.stride() == x.stride(), f"{ans.stride()=} {x.stride()=}"
        return ans

def _sanity_checks():
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError(
            "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet."
        )
