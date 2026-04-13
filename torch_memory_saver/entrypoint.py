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
    def region(self, tag: str = _TAG_DEFAULT, enable_cpu_backup: bool = False, chunk_size: int = 0):
        """Context manager for memory saving with optional tag.

        Args:
            tag: Tag for selective pause/resume.
            enable_cpu_backup: Whether to backup data to CPU on pause.
            chunk_size: Size in bytes of each independently pauseable chunk.
                When > 0, allocations inside this region use chunked virtual
                memory: one contiguous virtual address range backed by N
                independent physical allocations (N = size / chunk_size).

                Requirements enforced at allocation time:

                  * chunk_size must be a multiple of the device allocation
                    granularity (typically 2 MiB on current GPUs); the
                    library fails loudly otherwise rather than silently
                    rounding up.
                  * The total allocation size must be exactly divisible by
                    chunk_size.

                **Important:** chunk_size applies to *every* cudaMalloc call
                inside this region, including internal segment allocations
                made by PyTorch's caching allocator. Only create tensors
                whose byte size is an exact multiple of chunk_size inside a
                chunked region. Small or oddly-sized allocations will fail
                the divisibility check and abort the process. Put any
                scratch or auxiliary tensors in a separate non-chunked
                region or outside the chunked region entirely.

                Each distinct (tag, enable_cpu_backup, chunk_size) tuple
                gets its own ``torch.cuda.MemPool``. Switching chunk_size
                across calls creates additional long-lived pools.

                Chunk-level APIs (pause_chunks/resume_chunks/get_num_chunks/
                get_chunk_states/get_chunk_layout) require the tag to
                uniquely identify a single allocation. Per-chunk and
                whole-allocation pause/resume compose freely: calling
                either form on a chunk already in the target state is a
                no-op.

                Not supported on ROCm < 7.0 (TMS_ROCM_LEGACY_CHUNKED build).
        """
        self._ensure_initialized()
        with self._impl.region(tag=tag, enable_cpu_backup=enable_cpu_backup, chunk_size=chunk_size):
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

    def pause(self, tag: Optional[str] = None, save_chunk_state: bool = False):
        """Pause memory for specific tag or all memory if tag is None.

        Args:
            tag: Tag to filter which allocations to pause, or None for all.
            save_chunk_state: If True, snapshot each chunked allocation's
                per-chunk active/paused state before pausing everything.
                The snapshot is consumed by a later
                ``resume(restore_chunk_state=True)`` call.
        """
        self._ensure_initialized()
        self._impl.pause(tag=tag, save_chunk_state=save_chunk_state)

    def resume(self, tag: Optional[str] = None, restore_chunk_state: bool = False):
        """Resume memory for specific tag or all memory if tag is None.

        Args:
            tag: Tag to filter which allocations to resume, or None for all.
            restore_chunk_state: If True and a snapshot was saved by a prior
                ``pause(save_chunk_state=True)`` call, only resume the chunks
                that were active at the time of the snapshot. Chunks that were
                already paused before the save-pause remain paused.
        """
        self._ensure_initialized()
        self._impl.resume(tag=tag, restore_chunk_state=restore_chunk_state)

    def pause_chunks(self, tag: str, chunk_indices: Sequence[int]):
        """Pause specific chunks of a chunked allocation.

        The tag must uniquely identify a single chunked allocation. Indices
        pointing at chunks that are already paused are silently skipped, so
        this composes with pause()/resume().
        """
        self._ensure_initialized()
        self._impl.pause_chunks(tag=tag, chunk_indices=chunk_indices)

    def resume_chunks(self, tag: str, chunk_indices: Sequence[int]):
        """Resume specific chunks of a chunked allocation.

        The tag must uniquely identify a single chunked allocation. Indices
        pointing at chunks that are already active are silently skipped.
        """
        self._ensure_initialized()
        self._impl.resume_chunks(tag=tag, chunk_indices=chunk_indices)

    def get_num_chunks(self, tag: str) -> int:
        """Return the number of chunks for the allocation with the given tag.

        The tag must be non-empty and must uniquely identify a single
        allocation. Returns 0 if no allocation matches, 1 for non-chunked
        allocations.
        """
        self._ensure_initialized()
        return self._impl.get_num_chunks(tag=tag)

    def get_chunk_states(self, tag: str) -> list[bool]:
        """Return a list of booleans, one per chunk: True = active, False = paused.

        The tag must uniquely identify a single allocation. Returns an empty
        list if no allocation matches. For non-chunked allocations, returns a
        single-element list.
        """
        self._ensure_initialized()
        return self._impl.get_chunk_states(tag=tag)

    def get_chunk_layout(self, tag: str) -> list[tuple[int, int]]:
        """Return a list of (offset, size) tuples, one per chunk.

        The tag must uniquely identify a single allocation. Returns an empty
        list if no allocation matches. For non-chunked allocations, returns
        ``[(0, allocation_size)]``.
        """
        self._ensure_initialized()
        return self._impl.get_chunk_layout(tag=tag)

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
    def region(self, tag: str, enable_cpu_backup: bool, chunk_size: int = 0):
        # For hook_mode=preload, we need this b/c https://github.com/fzyzcjy/torch_memory_saver/pull/20#issuecomment-3047099047
        # (For hook_mode=torch we may not need it, but currently our primary usage is hook_mode=preload, thus we do this for simplicity)
        # chunk_size is part of the key: a cached block allocated with one
        # chunk_size cannot be reused under a different chunk_size without
        # the per-chunk metadata going out of sync with the physical layout.
        mem_pool = self._mem_pools[(tag, enable_cpu_backup, chunk_size)]
        with torch.cuda.use_mem_pool(mem_pool):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup, chunk_size=chunk_size):
                yield

    @contextmanager
    def cuda_graph(self, cuda_graph, pool, stream, capture_error_mode, tag: str, enable_cpu_backup: bool):
        assert self._hook_mode == "preload", "Only hook_mode=preload supports pauseable CUDA Graph currently"
        with torch.cuda.graph(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
            with self._with_region_config(tag=tag, enable_cpu_backup=enable_cpu_backup):
                yield

    @contextmanager
    def _with_region_config(self, tag: str, enable_cpu_backup: bool, chunk_size: int = 0):
        cdll = self._binary_wrapper.cdll
        orig_tag = cdll.tms_get_current_tag().decode("utf-8")
        orig_interesting_region = cdll.tms_get_interesting_region()
        orig_enable_cpu_backup = cdll.tms_get_enable_cpu_backup()
        orig_chunk_size = cdll.tms_get_chunk_size()

        self._binary_wrapper.set_config(tag=tag, interesting_region=True, enable_cpu_backup=enable_cpu_backup, chunk_size=chunk_size)
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
                chunk_size=orig_chunk_size,
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

    def pause(self, tag: Optional[str], save_chunk_state: bool = False):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_pause(tag_bytes, save_chunk_state)

    def resume(self, tag: Optional[str], restore_chunk_state: bool = False):
        tag_bytes = tag.encode("utf-8") if tag else None
        self._binary_wrapper.cdll.tms_resume(tag_bytes, restore_chunk_state)

    def pause_chunks(self, tag: str, chunk_indices: Sequence[int]):
        tag_bytes = tag.encode("utf-8")
        arr = (ctypes.c_size_t * len(chunk_indices))(*chunk_indices)
        self._binary_wrapper.cdll.tms_pause_chunks(tag_bytes, arr, len(chunk_indices))

    def resume_chunks(self, tag: str, chunk_indices: Sequence[int]):
        tag_bytes = tag.encode("utf-8")
        arr = (ctypes.c_size_t * len(chunk_indices))(*chunk_indices)
        self._binary_wrapper.cdll.tms_resume_chunks(tag_bytes, arr, len(chunk_indices))

    def get_num_chunks(self, tag: str) -> int:
        tag_bytes = tag.encode("utf-8")
        return self._binary_wrapper.cdll.tms_get_num_chunks_for_tag(tag_bytes)

    def get_chunk_states(self, tag: str) -> list[bool]:
        tag_bytes = tag.encode("utf-8")
        num_chunks = self._binary_wrapper.cdll.tms_get_num_chunks_for_tag(tag_bytes)
        if num_chunks == 0:
            return []
        buf = (ctypes.c_uint8 * num_chunks)()
        found = self._binary_wrapper.cdll.tms_get_chunk_states(tag_bytes, buf, num_chunks)
        if not found:
            return []
        return [bool(b) for b in buf]

    def get_chunk_layout(self, tag: str) -> list[tuple[int, int]]:
        tag_bytes = tag.encode("utf-8")
        num_chunks = self._binary_wrapper.cdll.tms_get_num_chunks_for_tag(tag_bytes)
        if num_chunks == 0:
            return []
        offsets = (ctypes.c_size_t * num_chunks)()
        sizes = (ctypes.c_size_t * num_chunks)()
        self._binary_wrapper.cdll.tms_get_chunk_layout(tag_bytes, offsets, sizes, num_chunks)
        return [(offsets[i], sizes[i]) for i in range(num_chunks)]

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
