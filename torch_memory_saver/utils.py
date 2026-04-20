import ctypes
import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_CUDA_MAJORS = (13, 12)


def _detect_cuda_major() -> int:
    """Pick which libcudart major the hosting process will use.

    Priority: torch's declared CUDA (set at torch-wheel build time), then probe
    libcudart.so.<major> for each major in _SUPPORTED_CUDA_MAJORS (highest first).
    Raises RuntimeError if neither resolves.
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch not importable; falling back to libcudart probe for CUDA detection")
    else:
        cuda_str = getattr(torch.version, "cuda", None)
        if cuda_str:
            try:
                return int(cuda_str.split(".", 1)[0])
            except ValueError:
                logger.warning(
                    "torch.version.cuda=%r is not a parseable CUDA version; "
                    "falling back to libcudart probe", cuda_str,
                )
        else:
            logger.info("torch.version.cuda is unset (CPU-only torch build); probing libcudart")

    for major in _SUPPORTED_CUDA_MAJORS:
        try:
            ctypes.CDLL(f"libcudart.so.{major}")
            return major
        except OSError:
            continue

    raise RuntimeError(
        f"torch_memory_saver: could not detect CUDA runtime. Tried torch.version.cuda "
        f"and libcudart.so.{{{','.join(map(str, _SUPPORTED_CUDA_MAJORS))}}}."
    )


def get_binary_path_from_package(stem: str):
    """Return the path to the .so for `stem`, picking the variant built against
    the detected CUDA major.

    The wheel ships multiple suffixed builds (e.g. `<stem>_cu12.abi3.so`,
    `<stem>_cu13.abi3.so`); this resolves to whichever matches the runtime CUDA.

    Raises:
        RuntimeError: if no CUDA runtime can be detected, or if zero or
            multiple .so files match the expected pattern.
    """
    major = _detect_cuda_major()
    dir_package = Path(__file__).parent
    pattern = f"{stem}_cu{major}.*.so"
    candidates = [p for d in (dir_package, dir_package.parent) for p in d.glob(pattern)]
    if len(candidates) != 1:
        raise RuntimeError(
            f"torch_memory_saver: expected exactly one .so matching {pattern!r} "
            f"(detected CUDA major={major}), found {len(candidates)}: {candidates}. "
            f"This usually means the installed wheel does not match your CUDA runtime."
        )
    return candidates[0]


# private utils, not to be used by end users
@contextmanager
def change_env(key: str, value: str):
    old_value = os.environ.get(key, "")
    os.environ[key] = value
    logger.debug(f"change_env set key={key} value={value}")
    try:
        yield
    finally:
        assert os.environ[key] == value
        os.environ[key] = old_value
        logger.debug(f"change_env restore key={key} value={old_value}")
