import ctypes
import logging

logger = logging.getLogger(__name__)


class BinaryWrapper:
    def __init__(self, path_binary: str):
        try:
            self.cdll = ctypes.CDLL(path_binary)
        except OSError as e:
            logger.error(f"Failed to load CDLL from {path_binary}: {e}")
            raise

        _setup_function_signatures(self.cdll)

    def set_config(self, *, tag: str, interesting_region: bool, enable_cpu_backup: bool):
        self.cdll.tms_set_current_tag(tag.encode("utf-8"))
        self.cdll.tms_set_interesting_region(interesting_region)
        self.cdll.tms_set_enable_cpu_backup(enable_cpu_backup)


def _setup_function_signatures(cdll):
    """Define function signatures for the C library"""
    cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    cdll.tms_get_interesting_region.restype = ctypes.c_bool
    cdll.tms_set_enable_cpu_backup.argtypes = [ctypes.c_bool]
    cdll.tms_get_enable_cpu_backup.restype = ctypes.c_bool
    cdll.tms_pause.argtypes = [ctypes.c_char_p]
    cdll.tms_resume.argtypes = [ctypes.c_char_p]
    cdll.set_memory_margin_bytes.argtypes = [ctypes.c_uint64]
    cdll.tms_get_cpu_backup_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
    cdll.tms_get_cpu_backup_pointer.restype = ctypes.POINTER(ctypes.c_uint8)
