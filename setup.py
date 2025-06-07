
import logging
import os
import shutil
from pathlib import Path
import platform
import setuptools
from setuptools import setup

logger = logging.getLogger(__name__)


# copy & modify from torch/utils/cpp_extension.py
def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            cuda_home = '/usr/local/cuda'
    return cuda_home
    
def _get_platform_architecture():
    host_arch = platform.machine()
    if host_arch == "aarch64":
        try:
            uname_output = subprocess.check_output(["uname", "-a"], encoding="utf-8")
            if "tegra" in uname_output:
                return f"tegra-{host_arch}"
        except Exception as e:
            print(f"[warn] Failed to run uname: {e}")

    return host_arch

cuda_home = Path(_find_cuda_home())
arch = platform.machine()
SYSTEM_ARCH_TYPE = os.environ.get("SYSTEM_ARCH", _get_platform_architecture())

if arch == 'aarch64':
    if SYSTEM_ARCH_TYPE == "tegra-aarch64":
        target_dir = 'targets/aarch64-linux'
    else:
        target_dir = 'targets/sbsa-linux'
else:
    target_dir = 'targets/x86_64-linux'

include_dirs = [
    str(cuda_home.resolve() / target_dir / 'include'),
]

library_dirs = [
    str(cuda_home.resolve() / 'lib64'),
    str(cuda_home.resolve() / 'lib64/stubs'),
]

setup(
    name='torch_memory_saver',
    version='0.0.7',
    ext_modules=[setuptools.Extension(
        'torch_memory_saver_cpp',
        ['csrc/torch_memory_saver.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['cuda'],
        define_macros=[('Py_LIMITED_API', '0x03090000')],
        py_limited_api=True,
    )],
    python_requires=">=3.9",
    packages=['torch_memory_saver'],
)
