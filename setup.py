
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

cuda_home = Path(_find_cuda_home())

include_dirs = [
    str((cuda_home / 'include').resolve()),
]

library_dirs = [
    str((cuda_home / 'lib64').resolve()),
    str((cuda_home / 'lib64/stubs').resolve()),
]

setup(
    name='torch_memory_saver',
    version='0.0.8',
    ext_modules=[
        setuptools.Extension(
            name,
            [
                'csrc/api_forwarder.cpp',
                'csrc/core.cpp',
                'csrc/entrypoint.cpp',
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=['cuda'],
            define_macros=[
                ('Py_LIMITED_API', '0x03090000'),
                *extra_macros,
            ],
            py_limited_api=True,
        )
        for name, extra_macros in [
            ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
            ('torch_memory_saver_hook_mode_torch', [('TMS_HOOK_MODE_TORCH', '1')]),
        ]
    ],
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["torch_memory_saver", "torch_memory_saver.*"]),
)
