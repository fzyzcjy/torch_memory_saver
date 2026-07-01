from contextlib import contextmanager

from .entrypoint import TorchMemorySaver, register_hook_mode
from .hooks import mode_preload as _mode_preload

# Global singleton
torch_memory_saver = TorchMemorySaver()


@contextmanager
def configure_subprocess():
    if not torch_memory_saver._uses_preload():
        yield
        return

    with _mode_preload.configure_subprocess():
        yield
