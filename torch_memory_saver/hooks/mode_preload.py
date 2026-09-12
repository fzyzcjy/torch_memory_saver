import logging
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package, change_env

logger = logging.getLogger(__name__)


class HookUtilModePreload(HookUtilBase):
    def get_path_binary(self):
        env_ld_preload = os.environ.get("LD_PRELOAD", "")
        
        interest_paths = [p for p in env_ld_preload.split(":") if "torch_memory_saver" in p]
        assert len(interest_paths) == 1, (
            f"TorchMemorySaver observes invalid LD_PRELOAD. "
            f"You can use configure_subprocess() utility, "
            f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
            f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
        )
        return interest_paths[0]


@contextmanager
def configure_subprocess():
    """Configure environment variables for subprocesses. Only needed for hook_mode=preload."""
    lib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))

    current_preload = os.environ.get("LD_PRELOAD", "")

    new_preload = f"{lib_path}:{current_preload}" if current_preload else lib_path

    # the hook's $ORIGIN RUNPATH only covers co-located pip installs, so also expose the parent's libcudart dir
    cudart_dir = _mapped_cudart_dir()
    current_lib = os.environ.get("LD_LIBRARY_PATH", "")
    new_lib = ":".join(x for x in (cudart_dir, current_lib) if x)

    with change_env("LD_PRELOAD", new_preload):
        with change_env("LD_LIBRARY_PATH", new_lib) if cudart_dir else nullcontext():
            yield


def _mapped_cudart_dir(maps_path="/proc/self/maps"):
    try:
        with open(maps_path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue

        _address, _permissions, _offset, _device, _inode, pathname = fields
        path = Path(pathname)
        if path.name.startswith("libcudart.so."):
            return str(path.parent)
    return None
