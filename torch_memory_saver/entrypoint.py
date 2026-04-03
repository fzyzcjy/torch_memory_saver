import atexit
import ctypes

import numpy as np
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Sequence
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
    def region(self, tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False, num_chunks: int = 1):
        """Context manager for memory saving with optional tag.

        Args:
            tag: Tag for selective pause/resume.
            enable_cpu_backup: Whether to backup data to CPU on pause.
            num_chunks: Number of independently pauseable chunks. When > 1,
                allocations inside this region use chunked virtual memory:
                one contiguous virtual address range backed by N independent
                physical allocations. Each chunk can be paused/resumed
                individually via pause_chunks/resume_chunks.
        """
        self._ensure_initialized()
        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup, num_chunks=num_chunks):
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

    def pause_chunks(self, tag: str, chunk_indices: Sequence[int]):
        """Pause specific chunks of a chunked allocation.

        Args:
            tag: Tag identifying the chunked allocation.
            chunk_indices: Indices of chunks to pause (0-based).
        """
        self._ensure_initialized()
        self._impl.pause_chunks(tag=tag, chunk_indices=chunk_indices)

    def resume_chunks(self, tag: str, chunk_indices: Sequence[int]):
        """Resume specific chunks of a chunked allocation.

        Args:
            tag: Tag identifying the chunked allocation.
            chunk_indices: Indices of chunks to resume (0-based).
        """
        self._ensure_initialized()
        self._impl.resume_chunks(tag=tag, chunk_indices=chunk_indices)

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
        self._mem_pools = defaultdict(lambda: torch.cuda.MemPool(allocator=self._hook_util.get_allocator()))
        _sanity_checks()
        if torch.version.hip:
            # Unlike CUDA where cuMem* are Driver API calls, HIP puts everything in user-space libraries
            # whose C++ static destructors may run before MemPool's destructor during process exit ("static 
            # destruction order fiasco"). By clearing _mem_pools in an atexit handler, we ensure MemPool 
            # destruction (and thus HIP API calls) happens while the HIP/HSA runtime is still fully alive.
            atexit.register(self._mem_pools.clear)

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool, num_chunks: int = 1):
        # For hook_mode=preload, we need this b/c https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # (For hook_mode=torch we may not need it, but currently our primary usage is hook_mode=preload, thus we do this for simplicity)
        mem_pool = self._mem_pools[(tag, enable_cpu_backup)]
        with torch.cuda.use_mem_pool(mem_pool):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup, num_chunks=num_chunks):
                yield

    @contextmanager
    def cuda_graph(self, cuda_graph, pool, stream, capture_error_mode, tag: str, enable_cpu_backup: bool):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        with torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def _with_region_config(self, tag: str, enable_cpu_backup: bool, num_chunks: int = 1):
        cdll = self._binary_wrapper.cdll
        orig_tag = cdll.tms_get_current_tag().decode("utf-8")
        orig_interesting_region = cdll.tms_get_interesting_region()
        orig_enable_cpu_backup = cdll.tms_get_enable_cpu_backup()
        orig_num_chunks = cdll.tms_get_num_chunks()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup, num_chunks=num_chunks)
        try:
            yield
        finally:
            assert cdll.tms_get_interesting_region()
            assert cdll.tms_get_enable_cpu_backup() == enable_cpu_backup
            assert cdll.tms_get_current_tag().decode("utf-8") == tag
            self._binary_wrapper.set_config(
                tag=orig_tag,
                interesting_region=orig_interesting_region,
                enable_cpu_backup=orig_enable_cpu_backup,
                num_chunks=orig_num_chunks,
            )

    @contextmanager
    def disable(self, dispose_mem_pool_after_use: bool = True):
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
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str]):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes)

    def pause_chunks(self, tag: str, chunk_indices: Sequence[int]):
        tag_bytes = tag.encode("utf-8")
        arr = (ctypes.c_size_t * len(chunk_indices))(*chunk_indices)
        self._binary_wrapper.cdll.tms_pause_chunks(tag_bytes, arr, len(chunk_indices))

    def resume_chunks(self, tag: str, chunk_indices: Sequence[int]):
        tag_bytes = tag.encode("utf-8")
        arr = (ctypes.c_size_t * len(chunk_indices))(*chunk_indices)
        self._binary_wrapper.cdll.tms_resume_chunks(tag_bytes, arr, len(chunk_indices))

    def get_cpu_backup(self, x: torch.Tensor, zero_copy: bool = False):
        assert x.is_cuda, f"{x.device=}"
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
