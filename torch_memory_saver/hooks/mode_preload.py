import logging
import os
from contextlib import contextmanager
from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package, change_env

logger = logging.getLogger(__name__)


class HookUtilModePreload(HookUtilBase):
    def get_path_binary(self):
        env_ld_preload = os.environ.get("LD_PRELOAD", "")
        
        ld_preload_paths = env_ld_preload.split(":")
        torch_memory_saver_path = None
        for path in ld_preload_paths:
            if "torch_memory_saver" in path:
                torch_memory_saver_path = path
                break
        
        assert torch_memory_saver_path is not None, (
            f"TorchMemorySaver observes invalid LD_PRELOAD. "
            f"You can use configure_subprocess() utility, "
            f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
            f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
        )
        return torch_memory_saver_path


@contextmanager
def configure_subprocess():
    """Configure environment variables for subprocesses. Only needed for hook_mode=preload."""
    with change_env("LD_PRELOAD", str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))):
        yield
