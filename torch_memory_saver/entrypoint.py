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
        if self._impl._is_xpu:
            # Unsupported on XPU: the OOM-margin guard needs a device-wide free-bytes
            # reading the Intel stack reports frozen (mem_get_info/sysman never move as
            # memory is consumed). Honoring it would be a lie, so reject not ignore.
            raise NotImplementedError(
                "TorchMemorySaver.memory_margin_bytes is not supported on Intel "
                "XPU: the OOM-margin guard needs a device-wide free-bytes "
                "reading, which the Intel GPU driver does not provide reliably "
                "(free-bytes telemetry is frozen). Manage device memory headroom "
                "outside torch_memory_saver instead."
            )
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

        self._device_module = torch.get_device_module()

        # Devices with pauseable allocations. pause()/resume() unmap+remap across
        # ALL of them, so on XPU each must be drained first (see pause()). Recorded
        # everywhere but consulted only on XPU; on CUDA/ROCm the caller syncs.
        self._region_devices: set[int] = set()

        if self._is_xpu:
            # Prewarm per-device SYCL contexts before registering the allocator.
            if hasattr(self._binary_wrapper.cdll, "tms_xpu_prewarm_devices"):
                self._binary_wrapper.cdll.tms_xpu_prewarm_devices(
                    self._device_module.device_count()
                )

        self._mem_pools = defaultdict(
            lambda: self._device_module.MemPool(allocator=self._hook_util.get_allocator())
        )

        _sanity_checks()
        if torch.version.hip or self._is_xpu:
            # HIP/SYCL runtime static dtors may run before MemPool's at exit
            # (destruction-order fiasco); free pools via atexit while runtime alive.
            atexit.register(self._mem_pools.clear)

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool, enable_disk_backup: bool):
        if self._is_xpu and enable_disk_backup:
            # Unsupported on XPU: disk spill uses cudaMallocHost/cudaMemcpy with no
            # Level Zero path. Raise a clean error (the C-ABI malloc also rejects it,
            # process-fatally) rather than aborting; use enable_cpu_backup instead.
            raise NotImplementedError(
                "TorchMemorySaver disk backup (enable_disk_backup=True) is not "
                "supported on Intel XPU; use enable_cpu_backup=True instead."
            )
        # A MemPool is bound to the current device at creation; key by device so
        # multi-device processes (e.g. TP ranks) don't reuse another's pool.
        device = self._current_device()
        # Record device so pause()/resume() can drain it -- backend unmaps all
        # devices but syncs only recorded ones.
        self._region_devices.add(device)
        key = (tag, enable_cpu_backup, enable_disk_backup, device)
        mem_pool = self._mem_pools[key]
        with self._device_module.use_mem_pool(mem_pool):
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
        if self._is_xpu:
            # Unsupported on XPU: pool-scoped allocs in the disabled body still route
            # to tms_torch_malloc and exit(1); reject rather than kill the process.
            raise NotImplementedError(
                "TorchMemorySaver.disable() is not supported on Intel XPU. "
                "Allocate outside a region() instead of using disable()."
            )

        assert dispose_mem_pool_after_use, "Only dispose_mem_pool_after_use=true is supported now"
        assert self._binary_wrapper.cdll.tms_get_interesting_region(), "disable() should be called only when tms is active"

        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        try:
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
            # Unmapping pages a kernel still touches hangs the device (DEVICE_LOST);
            # drain every device the backend will unmap for `tag` (_affected_devices).
            self._sync_affected_devices(tag)
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_pause(tag_bytes)
        if self._is_xpu and ret != 0:
            # Non-zero: an allocation was not fully released (left recoverable, see
            # xpu_pause); raise. Retained bytes are visible via tms_xpu_leaked_bytes.
            raise RuntimeError(
                f"torch_memory_saver.pause(tag={tag!r}) partially failed "
                f"(code {ret}); some allocations could not be released and are "
                "retained. Retry after resolving the failure."
            )

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_resume(tag_bytes)
        if self._is_xpu:
            # Non-zero: some allocation couldn't be re-created/mapped/restored; the
            # backend rolled it back to PAUSED (data preserved), so raise. Idempotent.
            if ret != 0:
                raise RuntimeError(
                    f"torch_memory_saver.resume(tag={tag!r}) partially failed "
                    f"(code {ret}); some allocations remain paused. Retry after "
                    "resolving the failure (e.g. free device memory)."
                )
            # Settle each re-mapped device's stream so the driver finishes paging/TLB
            # work before user kernels touch the remapped addresses.
            self._sync_affected_devices(tag)

    def _sync_affected_devices(self, tag: Optional[str]):
        # Drain every device the backend will unmap/remap for `tag` (a missed one
        # risks a DEVICE_LOST hang); synchronize(device) preserves the current device.
        for device in self._affected_devices(tag):
            self._device_module.synchronize(device)

    def _affected_devices(self, tag: Optional[str]) -> list[int]:
        """Device ids the backend will unmap/remap for `tag` (authoritative).

        Queries tms_xpu_affected_devices (reads the live allocation map under the
        allocator lock); falls back to the _region_devices mirror only if the
        symbol is absent (non-XPU or older .so).
        """
        cdll = self._binary_wrapper.cdll
        if not hasattr(cdll, "tms_xpu_affected_devices"):
            return list(self._region_devices)
        tag_bytes = tag.encode("utf-8") if tag else None
        # First call with no buffer learns the count, then size the buffer to it.
        # Count can only shrink between calls (no allocs during pause/resume), so
        # one retry suffices; loop guards against any interleaving.
        capacity = 0
        while True:
            buf = (ctypes.c_int * capacity)() if capacity else None
            count = int(cdll.tms_xpu_affected_devices(tag_bytes, buf, capacity))
            if count <= capacity:
                return [int(buf[i]) for i in range(count)] if capacity else []
            capacity = count

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
