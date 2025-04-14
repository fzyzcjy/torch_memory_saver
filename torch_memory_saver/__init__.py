import ctypes
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._mem_pool = None
        self._id = _global_info.next_id()
        assert self._id == 1, 'Only support one single instance yet (multi-instance will be implemented later)'

    @contextmanager
    def region(self):
        if _global_info.binary_info.enabled:
            self._ensure_mem_pool()
            with torch.cuda.use_mem_pool(self._mem_pool):
                _global_info.binary_info.cdll.tms_region_enter()
                try:
                    yield
                finally:
                    _global_info.binary_info.cdll.tms_region_leave()
        else:
            yield

    def pause(self):
        if _global_info.binary_info.enabled:
            _global_info.binary_info.cdll.tms_pause()

    def resume(self):
        if _global_info.binary_info.enabled:
            _global_info.binary_info.cdll.tms_resume()

    @property
    def enabled(self):
        return _global_info.binary_info.enabled

    def _ensure_mem_pool(self):
        if self._mem_pool is None:
            self._mem_pool = torch.cuda.MemPool()


@dataclass
class _BinaryInfo:
    cdll: Optional[ctypes.CDLL]

    @property
    def enabled(self):
        return self.cdll is not None

    @staticmethod
    def compute():
        env_ld_preload = os.environ.get('LD_PRELOAD', '')
        if 'torch_memory_saver' in env_ld_preload:
            paths = env_ld_preload.split(':')
            lib_path = next((path for path in paths if 'torch_memory_saver' in path), None)
            if lib_path:
                return _BinaryInfo(cdll=ctypes.CDLL(lib_path))
        else:
            logger.warning(
                f'TorchMemorySaver is disabled for the current process because invalid LD_PRELOAD="{env_ld_preload}" (process_id={os.getpid()})')
            return _BinaryInfo(cdll=None)


class _GlobalInfo:
    def __init__(self):
        self._binary_info: Optional[_BinaryInfo] = None
        self._last_id = 0

    @property
    def binary_info(self):
        if self._binary_info is None:
            self._binary_info = _BinaryInfo.compute()
        return self._binary_info

    def next_id(self):
        self._last_id += 1
        return self._last_id


_global_info = _GlobalInfo()