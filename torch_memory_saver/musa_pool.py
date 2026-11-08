"""MUSA MemPool wrapper for torch_musa < 2.9.1.post2-rc2.

Before torch_musa 2.9.1.post2-rc2, ``MemPool`` has no destructor that calls
``releasePool``. A dropped pool therefore keeps its cached blocks in
``graph_pools``, and ``empty_cache()`` cannot free them. This wrapper keeps the
underlying pool alive, records the devices it was used on, and releases it
explicitly after the scope, followed by ``empty_cache()``.

torch_musa 2.9.1.post2-rc2+ releases the pool in ``MemPool``'s destructor, so
this wrapper should be deleted (and the device module's ``MemPool`` used
directly) once the minimum supported torch_musa version reaches that release.
"""

import re
from contextlib import contextmanager

import torch


# First torch_musa release where MemPool::~MemPool calls releasePool.
_POOL_RELEASE_FIXED = (2, 9, 1, 2, 2)


def _musa_version_key(version: str):
    """Return (major, minor, patch, post, rc) for a torch_musa version string."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return None
    post_match = re.search(r"\.post(\d+)", version)
    post = int(post_match.group(1)) if post_match is not None else 0
    # A plain release sorts after its release candidates; 99 means "no rc".
    rc_match = re.search(r"rc(\d+)", version, re.IGNORECASE)
    rc = int(rc_match.group(1)) if rc_match is not None else 99
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), post, rc)


def _needs_explicit_release() -> bool:
    musa = getattr(torch, "musa", None)
    version = getattr(musa, "__version__", "") if musa is not None else ""
    key = _musa_version_key(version)
    if key is None:
        return True
    return key < _POOL_RELEASE_FIXED


_NEEDS_EXPLICIT_RELEASE = _needs_explicit_release()


class MusaMemPool:
    """Thin wrapper around ``torch.musa.MemPool`` with an explicit release."""

    def __init__(self, allocator=None):
        self._pool = (
            torch.musa.MemPool()
            if allocator is None
            else torch.musa.MemPool(allocator)
        )
        self._devices = set()
        self._released = False

    @property
    def id(self):
        return self._pool.id

    @property
    def allocator(self):
        return self._pool.allocator

    @contextmanager
    def use(self):
        assert self._pool is not None, "MusaMemPool already released"
        self._devices.add(torch.musa.current_device())
        # torch_musa raises "captures_underway.empty() INTERNAL ASSERT FAILED"
        # in empty_cache() while a MemPool is active (use_mem_pool records a
        # capture). Callers such as slime call empty_cache() inside disable(),
        # so neutralize it for the scope; an uncaught error would kill the
        # worker. release() calls the real empty_cache() after this scope.
        saved = []
        for module_name in ("musa", "cuda"):
            module = getattr(torch, module_name, None)
            if module is not None and hasattr(module, "empty_cache"):
                saved.append((module, module.empty_cache))
                module.empty_cache = lambda *a, **k: None
        try:
            with torch.musa.use_mem_pool(self._pool):
                yield
        finally:
            for module, fn in saved:
                module.empty_cache = fn

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if _NEEDS_EXPLICIT_RELEASE:
            release_pool = getattr(
                getattr(torch.musa, "_MUSAC", None), "_musa_releasePool", None
            )
            if release_pool is not None:
                for device in self._devices:
                    release_pool(device, self.id)
        self._pool = None
        try:
            torch.musa.empty_cache()
        except RuntimeError as error:
            # Another mempool capture may still be active; the released pool is
            # already freeable, so a later empty_cache() will reclaim it.
            if "captures_underway.empty()" not in str(error):
                raise
