import ctypes
import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self, enable_use_mem_pool=True):
        self.enable_use_mem_pool = enable_use_mem_pool
        self._mem_pool = None
        self._id = _global_info.next_id()
        assert self._id == 1, 'Only support one single instance yet (multi-instance will be implemented later)'

    @contextmanager
    def region(self):
        if _global_info.binary_info.enabled:
            if self.enable_use_mem_pool:
                self._ensure_mem_pool()
                ctx = torch.cuda.use_mem_pool(self._mem_pool)
            else:
                print('HACK: enable_use_mem_pool=False')
                ctx = nullcontext()
            with ctx:
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
            return _BinaryInfo(cdll=ctypes.CDLL(env_ld_preload))
        else:
            print(
                f'TorchMemorySaver is disabled for the current process because invalid LD_PRELOAD. '
                f'You can use configure_subprocess() utility, '
                f'or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. '
                f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
            )
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


def get_binary_path():
    dir_package = Path(__file__).parent
    candidates = [
        p
        for d in [dir_package, dir_package.parent]
        for p in d.glob('torch_memory_saver_cpp.*.so')
    ]
    assert len(candidates) == 1, f'{candidates=}'
    return candidates[0]


@contextmanager
def configure_subprocess():
    with change_env('LD_PRELOAD', str(get_binary_path())):
        yield


@contextmanager
def change_env(key: str, value: str):
    old_value = os.environ.get(key, '')
    os.environ[key] = value
    logger.debug(f'change_env set key={key} value={value}')
    try:
        yield
    finally:
        assert os.environ[key] == value
        os.environ[key] = old_value
        logger.debug(f'change_env restore key={key} value={old_value}')
