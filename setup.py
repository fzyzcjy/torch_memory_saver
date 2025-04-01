import os
import glob
import shutil
import logging
import setuptools
from pathlib import Path
from setuptools import setup

logger = logging.getLogger(__name__)

# Find the ROCm/HIP install path
def _find_rocm_home():
    """Find the ROCm/HIP install path."""
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        # Check if hipcc exists in PATH
        hipcc_path = shutil.which("hipcc")
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(hipcc_path))
        else:
            # Default location
            rocm_home = '/opt/rocm'
    return rocm_home

rocm_home = Path(_find_rocm_home())
include_dirs = [
    str(rocm_home.resolve() / 'include'),
]

# Configure for HIP compilation
from setuptools.command.build_ext import build_ext

class HipExtension(setuptools.Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)

class build_hip_ext(build_ext):
    def build_extensions(self):
        # Set hipcc as the compiler
        self.compiler.set_executable("compiler_so", "hipcc")
        self.compiler.set_executable("compiler_cxx", "hipcc")
        self.compiler.set_executable("linker_so", "hipcc --shared")  # Specify shared library mode
        
        # Add extra compiler and linker flags
        for ext in self.extensions:
            ext.extra_compile_args = ['-fPIC']
            ext.extra_link_args = ['-shared']  # Explicitly specify shared mode
        
        build_ext.build_extensions(self)

setup(
    name='torch_memory_saver',
    version='0.0.5',
    ext_modules=[HipExtension(
        'torch_memory_saver_cpp',
        ['csrc/torch_memory_saver_hip.cpp'],
        include_dirs=include_dirs,
        libraries=['amdhip64', 'dl'],  # Add dl library for dlsym, dlerror
        extra_link_args=['-shared', '-fPIC'],  # Ensure proper shared library linking
    )],
    cmdclass={
        'build_ext': build_hip_ext,
    },
    python_requires=">=3.9",
    packages=['torch_memory_saver'],
)
