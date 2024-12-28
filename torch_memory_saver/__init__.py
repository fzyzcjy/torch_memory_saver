import ctypes
import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._mem_pool = torch.cuda.MemPool()
        self._cdll = _global_info.cdll
        self._id = _global_info.next_id()
        logger.debug('setup cdll=%s', self._cdll)
        assert self._id == 1, 'Only support one single instance yet (multi-instance will be implemented later)'

    @contextmanager
    def region(self):
        with torch.cuda.use_mem_pool(self._mem_pool):
            self._cdll.tms_region_enter()
            try:
                yield
            finally:
                self._cdll.tms_region_leave()

    def pause(self):
        self._cdll.tms_pause()

    def resume(self):
        self._cdll.tms_resume()


class _GlobalInfo:
    def __init__(self):
        self._cdll: Optional[ctypes.CDLL] = None
        self._last_id = 0

    @property
    def cdll(self):
        if self._cdll is None:
            self._cdll = _compute_cdll()
        return self._cdll

    def next_id(self):
        self._last_id += 1
        return self._last_id


_global_info = _GlobalInfo()


def _compute_cdll():
    env_ld_preload = os.environ.get('LD_PRELOAD', '')
    assert 'torch_memory_saver' in env_ld_preload, f'Please specify correct LD_PRELOAD (currently: {env_ld_preload})'
    return ctypes.CDLL(env_ld_preload)
