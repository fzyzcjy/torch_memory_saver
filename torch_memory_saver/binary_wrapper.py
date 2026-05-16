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

    def set_config(self, *, tag: str, interesting_region: bool, enable_cpu_backup: bool, chunk_size: int = 0):
        self.cdll.tms_set_current_tag(tag.encode("utf-8"))
        self.cdll.tms_set_interesting_region(interesting_region)
        self.cdll.tms_set_enable_cpu_backup(enable_cpu_backup)
        self.cdll.tms_set_chunk_size(chunk_size)


def _setup_function_signatures(cdll):
    """Define function signatures for the C library"""
    cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    cdll.tms_get_current_tag.restype = ctypes.c_char_p
    cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    cdll.tms_get_interesting_region.restype = ctypes.c_bool
    cdll.tms_set_enable_cpu_backup.argtypes = [ctypes.c_bool]
    cdll.tms_get_enable_cpu_backup.restype = ctypes.c_bool
    cdll.tms_set_chunk_size.argtypes = [ctypes.c_size_t]
    cdll.tms_get_chunk_size.restype = ctypes.c_size_t
    cdll.tms_get_num_chunks_for_tag.argtypes = [ctypes.c_char_p]
    cdll.tms_get_num_chunks_for_tag.restype = ctypes.c_size_t
    cdll.tms_get_chunk_states.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    cdll.tms_pause.argtypes = [ctypes.c_char_p]
    cdll.tms_resume.argtypes = [ctypes.c_char_p]
    cdll.tms_pause_chunks.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    cdll.tms_resume_chunks.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    cdll.set_memory_margin_bytes.argtypes = [ctypes.c_uint64]
    cdll.tms_get_cpu_backup_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
    cdll.tms_get_cpu_backup_pointer.restype = ctypes.POINTER(ctypes.c_uint8)
