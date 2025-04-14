import logging
import os
import shutil
from pathlib import Path

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

def _find_cuda_dir(cuda_home, dir_name, target_file=None, alternative_dir=None, subdir=None):
    """Find CUDA directories."""
    cuda_home = Path(cuda_home)
    dirs = []

    # Check if main directory exists
    if (cuda_dir := cuda_home / dir_name).is_dir():
        dirs.append(str(cuda_dir.resolve()))
    # Check alternative directory paths
    elif alternative_dir and (cuda_dir := cuda_home / alternative_dir).is_dir():
        dirs.append(str(cuda_dir.resolve()))
    # Search by marker file if specified and main directory doesn't exist
    elif target_file:
        for path in cuda_home.rglob(target_file):
            cuda_dir = path.parent
            dirs.append(str(cuda_dir.resolve()))
            break
        else:
            raise RuntimeError(f"Could not find CUDA {dir_name} directory nor {target_file} file.")
    else:
        raise RuntimeError(f"Could not find CUDA {dir_name} directory.")

    # Check for subdirectories if specified
    if subdir:
        if (cuda_dir := cuda_dir / subdir).is_dir():
            dirs.append(str(cuda_dir.resolve()))
        else:
            raise RuntimeError(f"Could not find CUDA {cuda_dir} sub-directory.")

    return dirs


cuda_home = _find_cuda_home()
include_dirs = _find_cuda_dir(cuda_home, 'include', target_file='cuda_runtime_api.h')
library_dirs = _find_cuda_dir(cuda_home, 'lib', alternative_dir='lib64', subdir='stubs')

setup(
    name='torch_memory_saver',
    version='0.0.6',
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
