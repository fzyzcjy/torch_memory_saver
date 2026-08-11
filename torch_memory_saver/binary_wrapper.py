import ctypes
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BinaryWrapper:
    def __init__(self, path_binary: str):
        try:
            self.cdll = ctypes.CDLL(path_binary)
        except OSError as e:
            logger.error(f"Failed to load CDLL from {path_binary}: {e}")
            raise

        _setup_function_signatures(self.cdll)

    def set_config(
        self,
        *,
        tag: str,
        interesting_region: bool,
        enable_cpu_backup: bool,
        cpu_backup_backend: Optional[str] = None,
        enable_disk_backup: bool = False,
    ):
        self.cdll.tms_set_current_tag(tag.encode("utf-8"))
        self.cdll.tms_set_interesting_region(interesting_region)
        self.cdll.tms_set_enable_cpu_backup(enable_cpu_backup)
        self.cdll.tms_set_cpu_backup_backend((cpu_backup_backend or "").encode("utf-8"))
        self.cdll.tms_set_enable_disk_backup(enable_disk_backup)


def _setup_function_signatures(cdll):
    """Define function signatures for the C library"""
    cdll.tms_set_current_tag.argtypes = [ctypes.c_char_p]
    cdll.tms_get_current_tag.restype = ctypes.c_char_p
    cdll.tms_set_interesting_region.argtypes = [ctypes.c_bool]
    cdll.tms_get_interesting_region.restype = ctypes.c_bool
    cdll.tms_set_enable_cpu_backup.argtypes = [ctypes.c_bool]
    cdll.tms_get_enable_cpu_backup.restype = ctypes.c_bool
    cdll.tms_set_cpu_backup_backend.argtypes = [ctypes.c_char_p]
    cdll.tms_get_cpu_backup_backend.restype = ctypes.c_char_p
    cdll.tms_set_enable_disk_backup.argtypes = [ctypes.c_bool]
    cdll.tms_get_enable_disk_backup.restype = ctypes.c_bool
    cdll.tms_set_disk_backup_dir.argtypes = [ctypes.c_char_p]
    cdll.tms_pause.argtypes = [ctypes.c_char_p]
    cdll.tms_pause.restype = ctypes.c_int
    cdll.tms_resume.argtypes = [ctypes.c_char_p]
    cdll.tms_resume.restype = ctypes.c_int
    cdll.set_memory_margin_bytes.argtypes = [ctypes.c_uint64]
    cdll.tms_set_retain_cpu_backup.argtypes = [ctypes.c_bool]
    cdll.tms_get_retain_cpu_backup.restype = ctypes.c_bool
    cdll.tms_get_cpu_backup_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
    cdll.tms_get_cpu_backup_pointer.restype = ctypes.POINTER(ctypes.c_uint8)

    # XPU-only symbols.
    if hasattr(cdll, "tms_xpu_prewarm_devices"):
        cdll.tms_xpu_prewarm_devices.argtypes = [ctypes.c_int]
        cdll.tms_xpu_prewarm_devices.restype = None
    if hasattr(cdll, "tms_xpu_device_free_bytes"):
        cdll.tms_xpu_device_free_bytes.argtypes = [ctypes.c_int]
        cdll.tms_xpu_device_free_bytes.restype = ctypes.c_uint64
    if hasattr(cdll, "tms_xpu_committed_bytes"):
        cdll.tms_xpu_committed_bytes.argtypes = [ctypes.c_int]
        cdll.tms_xpu_committed_bytes.restype = ctypes.c_uint64
    if hasattr(cdll, "tms_xpu_leaked_bytes"):
        cdll.tms_xpu_leaked_bytes.argtypes = [ctypes.c_int]
        cdll.tms_xpu_leaked_bytes.restype = ctypes.c_uint64
    if hasattr(cdll, "tms_xpu_tracked_bytes"):
        cdll.tms_xpu_tracked_bytes.argtypes = [ctypes.c_int]
        cdll.tms_xpu_tracked_bytes.restype = ctypes.c_uint64
    if hasattr(cdll, "tms_xpu_free"):
        cdll.tms_xpu_free.argtypes = [ctypes.c_void_p]
        cdll.tms_xpu_free.restype = ctypes.c_int
    if hasattr(cdll, "tms_xpu_affected_devices"):
        cdll.tms_xpu_affected_devices.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint32,
        ]
        cdll.tms_xpu_affected_devices.restype = ctypes.c_uint32
