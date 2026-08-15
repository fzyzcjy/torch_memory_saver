import atexit
import ctypes

import numpy as np
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Literal, Optional
import torch

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase, HookMode
from .utils import is_xpu

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"
CpuBackupBackend = Literal["mmap", "pinned"]


class TorchMemorySaver:
    def __init__(self):
        self._impl_ctor_kwargs = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(
        self,
        tag: str = _TAG_DEFAULT,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
        cpu_backup_backend: Optional[CpuBackupBackend] = None,
    ):
        """Context manager for memory saving with optional tag.

        enable_disk_backup spills paused memory to files instead of a CPU
        buffer; mutually exclusive with enable_cpu_backup. The target directory is
        process-global (TMS_DISK_BACKUP_DIR or set_disk_backup_dir()), not
        region-scoped, and must be a real disk mount (not tmpfs).

        cpu_backup_backend selects the host shadow when enable_cpu_backup is True:
        "pinned" (default; cudaMallocHost/hipHostMalloc) or "mmap" (reclaimable
        RSS on CUDA). ROCm/XPU are pinned-only.
        """
        self._ensure_initialized()
        assert not (enable_cpu_backup and enable_disk_backup), \
            "enable_cpu_backup and enable_disk_backup are mutually exclusive"
        with self._impl.region(
            tag=tag,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
            cpu_backup_backend=cpu_backup_backend,
        ):
            yield

    @contextmanager
    def cuda_graph(
            self,
            cuda_graph, pool=None, stream=None, capture_error_mode='global',
            tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False,
            cpu_backup_backend: Optional[CpuBackupBackend] = None,
    ):
        """Similar to `torch.cuda.graph`, but ensures memory in it to be pauseable."""
        self._ensure_initialized()
        with self._impl.cuda_graph(
                cuda_graph=cuda_graph,
                pool=pool, stream=stream, capture_error_mode=capture_error_mode,
                tag=tag, enable_cpu_backup=enable_cpu_backup,
                cpu_backup_backend=cpu_backup_backend,
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
            raise NotImplementedError(
                "TorchMemorySaver.memory_margin_bytes is not supported on Intel "
                "XPU: the OOM-margin guard needs a device-wide free-bytes "
                "reading, which the Intel GPU driver does not provide reliably "
                "(free-bytes telemetry is frozen). Manage device memory headroom "
                "outside torch_memory_saver instead."
            )
        self._impl._binary_wrapper.cdll.set_memory_margin_bytes(value)

    @property
    def retain_cpu_backup(self) -> bool:
        self._ensure_initialized()
        return self._impl._binary_wrapper.cdll.tms_get_retain_cpu_backup()

    @retain_cpu_backup.setter
    def retain_cpu_backup(self, value: bool):
        self._ensure_initialized()
        self._impl._binary_wrapper.cdll.tms_set_retain_cpu_backup(value)

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
    def region(
        self,
        tag: str,
        enable_cpu_backup: bool,
        enable_disk_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ):
        if self._is_xpu and enable_disk_backup:
            raise NotImplementedError(
                "TorchMemorySaver disk backup (enable_disk_backup=True) is not "
                "supported on Intel XPU; use enable_cpu_backup=True instead."
            )
        # See https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # Key by device too: a MemPool is bound to its creation device, so a
        # multi-device process must not reuse one device's pool on another.
        # Include backend so torch-mode pool reuse cannot stick the first kind.
        device = self._current_device()
        resolved_cpu_backup_backend = ""
        if enable_cpu_backup:
            resolved_cpu_backup_backend = (
                cpu_backup_backend
                if cpu_backup_backend is not None
                else self._binary_wrapper.cdll.tms_get_cpu_backup_backend().decode("utf-8")
            )
        key = (
            tag,
            enable_cpu_backup,
            enable_disk_backup,
            resolved_cpu_backup_backend,
            device,
        )
        mem_pool = self._mem_pools[key]
        with self._device_module.use_mem_pool(mem_pool):
            with self._with_region_config(
                tag=tag,
                enable_cpu_backup=enable_cpu_backup,
                enable_disk_backup=enable_disk_backup,
                cpu_backup_backend=cpu_backup_backend,
            ):
                yield
    def _current_device(self) -> int:
        return self._device_module.current_device()

    @contextmanager
    def cuda_graph(
        self,
        cuda_graph,
        pool,
        stream,
        capture_error_mode,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
    ):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        with torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(
                tag=tag,
                enable_cpu_backup=enable_cpu_backup,
                cpu_backup_backend=cpu_backup_backend,
            ):
                yield

    @contextmanager
    def _with_region_config(
        self,
        tag: str,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[CpuBackupBackend],
        enable_disk_backup: bool = False,
    ):
        cdll = self._binary_wrapper.cdll
        orig_tag = cdll.tms_get_current_tag().decode("utf-8")
        orig_interesting_region = cdll.tms_get_interesting_region()
        orig_enable_cpu_backup = cdll.tms_get_enable_cpu_backup()
        orig_cpu_backup_backend = cdll.tms_get_cpu_backup_backend().decode("utf-8")
        orig_enable_disk_backup = cdll.tms_get_enable_disk_backup()
        expected_cpu_backup_backend = cpu_backup_backend or orig_cpu_backup_backend

        self._binary_wrapper.set_config(
            tag=tag,
            interesting_region=True,
            enable_cpu_backup=enable_cpu_backup,
            cpu_backup_backend=cpu_backup_backend,
            enable_disk_backup=enable_disk_backup,
        )
        try:
            yield
        finally:
            assert cdll.tms_get_interesting_region()
            assert cdll.tms_get_enable_cpu_backup() == enable_cpu_backup
            assert (
                cdll.tms_get_cpu_backup_backend().decode("utf-8")
                == expected_cpu_backup_backend
            )
            assert cdll.tms_get_enable_disk_backup() == enable_disk_backup
            assert cdll.tms_get_current_tag().decode("utf-8") == tag
            self._binary_wrapper.set_config(
                tag=orig_tag,
                interesting_region=orig_interesting_region,
                enable_cpu_backup=orig_enable_cpu_backup,
                cpu_backup_backend=orig_cpu_backup_backend,
                enable_disk_backup=orig_enable_disk_backup,
            )

    @contextmanager
    def disable(self, dispose_mem_pool_after_use: bool = True):
        if self._is_xpu:
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
            self._xpu_sync_affected_devices(tag)
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_pause(tag_bytes)
        if self._is_xpu and ret != 0:
            raise RuntimeError(
                f"torch_memory_saver.pause(tag={tag!r}) partially failed "
                f"(code {ret}); some allocations could not be released and are "
                "retained. Retry after resolving the failure."
            )

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        ret = self._binary_wrapper.cdll.tms_resume(tag_bytes)
        if self._is_xpu:
            if ret != 0:
                raise RuntimeError(
                    f"torch_memory_saver.resume(tag={tag!r}) partially failed "
                    f"(code {ret}); some allocations remain paused. Retry after "
                    "resolving the failure (e.g. free device memory)."
                )
            self._xpu_sync_affected_devices(tag)

    def _xpu_sync_affected_devices(self, tag: Optional[str]):
        for device in self._xpu_affected_devices(tag):
            self._device_module.synchronize(device)

    def _xpu_affected_devices(self, tag: Optional[str]) -> list[int]:
        """Device ids the XPU backend will unmap/remap for `tag` (authoritative).

        XPU only (callers guard on _is_xpu). Queries tms_xpu_affected_devices,
        which reads the live allocation map under the allocator lock -- exactly
        what xpu_pause/xpu_resume iterate.
        """
        cdll = self._binary_wrapper.cdll
        tag_bytes = tag.encode("utf-8") if tag else None
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
