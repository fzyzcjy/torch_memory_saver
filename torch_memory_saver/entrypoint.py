import ctypes
import logging
import os
from contextlib import contextmanager
from typing import Optional
import torch

from .binary_wrapper import BinaryWrapper
from .hooks.base import HookUtilBase, HookMode

logger = logging.getLogger(__name__)

_TAG_DEFAULT = "default"


class TorchMemorySaver:
    def __init__(self):
        self._impl_ctor_kwargs = {}
        self._impl: Optional[_TorchMemorySaverImpl] = None

    @contextmanager
    def region(self, tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False):
        """Context manager for memory saving with optional tag"""
        self._ensure_initialized()
        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup):
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
        with self._impl.disable():
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        self._impl.pause(tag=tag)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
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

    def _ensure_initialized(self):
        if self._impl is not None:
            return
        self._impl = _TorchMemorySaverImpl(**self._impl_ctor_kwargs)
        del self._impl_ctor_kwargs


class _TorchMemorySaverImpl:
    def __init__(self, hook_mode: HookMode = "preload"):
        self._hook_mode = hook_mode
        self._hook_util = HookUtilBase.create(hook_mode=hook_mode)
        self._binary_wrapper = BinaryWrapper(path_binary=self._hook_util.get_path_binary())
        self._mem_pool = torch.cuda.MemPool(allocator=self._hook_util.get_allocator())
        _sanity_checks()

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        with torch.cuda.use_mem_pool(self._mem_pool):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def cuda_graph(self, cuda_graph, pool, stream, capture_error_mode, tag: str, enable_cpu_backup: bool):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        with torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def _with_region_config(self, tag: str, enable_cpu_backup: bool):
        assert not self._binary_wrapper.cdll.tms_get_interesting_region()
        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup)
        try:
            yield
        finally:
            assert self._binary_wrapper.cdll.tms_get_interesting_region()
            self._binary_wrapper.set_config(tag=_TAG_DEFAULT, interesting_region=False, enable_cpu_backup=False)

    @contextmanager
    def disable(self):
        old_is_interesting_region = self._binary_wrapper.cdll.tms_get_interesting_region()
        self._binary_wrapper.cdll.tms_set_interesting_region(False)
        try:
            yield
        finally:
            self._binary_wrapper.cdll.tms_set_interesting_region(old_is_interesting_region)

    def pause(self, tag: Optional[str]):
        self._binary_wrapper.pause(tag)

    def resume(self, tag: Optional[str]):
        self._binary_wrapper.resume(tag)


def _sanity_checks():
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError(
            "TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet."
        )
