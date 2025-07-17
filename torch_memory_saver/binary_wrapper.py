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

    def resume(self, tag: str = None) -> bool:
        """
        Resume memory allocations for the given tag.
        Returns True on success, raises RuntimeError on failure.
        """
        tag_bytes = tag.encode("utf-8") if tag else None
        result = self.cdll.tms_resume(tag_bytes)
        if result != 0:
            error_msg = self.cdll.tms_get_last_error()
            if error_msg:
                error_str = error_msg.decode("utf-8")
            else:
                error_str = "Unknown error during resume operation"
            raise RuntimeError(error_str)
        return True

    def pause(self, tag: str = None) -> bool:
        """
        Pause memory allocations for the given tag.
        Returns True on success, raises RuntimeError on failure.
        """
        tag_bytes = tag.encode("utf-8") if tag else None
        result = self.cdll.tms_pause(tag_bytes)
        if result != 0:
            error_msg = self.cdll.tms_get_last_error()
            if error_msg:
                error_str = error_msg.decode("utf-8")
            else:
                error_str = "Unknown error during pause operation"
            raise RuntimeError(error_str)
        return True


def _setup_function_signatures(cdll):
    """Define function signatures for the C library"""
    cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    cdll.tms_get_interesting_region.restype = ctypes.c_bool
    cdll.tms_set_enable_cpu_backup.argtypes = [ctypes.c_bool]
    cdll.tms_pause.argtypes = [ctypes.c_char_p]
    cdll.tms_pause.restype = ctypes.c_int
    cdll.tms_resume.argtypes = [ctypes.c_char_p]
    cdll.tms_resume.restype = ctypes.c_int
    cdll.tms_get_last_error.restype = ctypes.c_char_p
