import ctypes
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchMemorySaver:
    def __init__(self):
        self._mem_pool = None

    @contextmanager
    def region(self, tag: str = "default"):
        """Context manager for memory saving with optional tag"""
        if _global_info.binary_info.enabled:
            self._ensure_mem_pool()
            with torch.cuda.use_mem_pool(self._mem_pool):
                _global_info.binary_info.cdll.tms_set_current_tag(tag.encode('utf-8'))
                _global_info.binary_info.cdll.tms_region_enter()
                try:
                    yield
                finally:
                    _global_info.binary_info.cdll.tms_region_leave()
        else:
            yield

    def pause(self, tag: Optional[str] = None):
        """Pause memory for specific tag or all memory if tag is None"""
        if _global_info.binary_info.enabled:
            tag_bytes = tag.encode('utf-8') if tag else None
            _global_info.binary_info.cdll.tms_pause(tag_bytes)

    def resume(self, tag: Optional[str] = None):
        """Resume memory for specific tag or all memory if tag is None"""
        if _global_info.binary_info.enabled:
            tag_bytes = tag.encode('utf-8') if tag else None
            _global_info.binary_info.cdll.tms_resume(tag_bytes)

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
    def _setup_function_signatures(cdll):
        """Define function signatures for the C library"""
        cdll.tms_region_enter.argtypes = []
        cdll.tms_region_leave.argtypes = []
        cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
        cdll.tms_pause.argtypes = [ctypes.c_char_p]
        cdll.tms_resume.argtypes = [ctypes.c_char_p]

    @staticmethod
    def compute():
        env_ld_preload = os.environ.get('LD_PRELOAD', '')
        if 'torch_memory_saver' in env_ld_preload:
            try:
                cdll = ctypes.CDLL(env_ld_preload)
                _BinaryInfo._setup_function_signatures(cdll)
                return _BinaryInfo(cdll=cdll)
            except OSError as e:
                logger.error(f'Failed to load CDLL from {env_ld_preload}: {e}')
                return _BinaryInfo(cdll=None)
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

    @property
    def binary_info(self):
        if self._binary_info is None:
            self._binary_info = _BinaryInfo.compute()
        return self._binary_info


_global_info = _GlobalInfo()

# Global singleton instance
torch_memory_saver = TorchMemorySaver()

def get_binary_path():
    dir_package = Path(__file__).parent
    candidates = [
        p
        for d in [dir_package, dir_package.parent]
        for p in d.glob('torch_memory_saver_cpp.*.so')
    ]
    assert len(candidates) == 1, f'Expected exactly one torch_memory_saver_cpp library, found: {candidates}'
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
