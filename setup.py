import logging
import os
import shutil
from pathlib import Path

import setuptools
from setuptools import setup
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)


def _find_cuda_home():
    """Find the CUDA install path."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            cuda_home = '/usr/local/cuda'
    return cuda_home


def _find_rocm_home():
    """Find the ROCm/HIP install path."""
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        hipcc_path = shutil.which("hipcc")
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(hipcc_path))
        else:
            rocm_home = '/opt/rocm'
    return rocm_home


def _detect_platform():
    """Detect whether to use CUDA or HIP based on available tools."""
    # Check for HIP first (since it might be preferred on AMD systems)
    if shutil.which("hipcc") is not None:
        return "hip"
    elif shutil.which("nvcc") is not None:
        return "cuda"
    else:
        # Default to CUDA if neither is found
        return "cuda"


class HipExtension(setuptools.Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)


class CudaExtension(setuptools.Extension):
    def __init__(self, name, sources, *args, **kwargs):
        super().__init__(name, sources, *args, **kwargs)


class build_hip_ext(build_ext):
    def build_extensions(self):
        # Set hipcc as the compiler
        self.compiler.set_executable("compiler_so", "hipcc")
        self.compiler.set_executable("compiler_cxx", "hipcc")
        self.compiler.set_executable("linker_so", "hipcc --shared")
        
        # Add extra compiler and linker flags
        for ext in self.extensions:
            ext.extra_compile_args = ['-fPIC']
            ext.extra_link_args = ['-shared']
        
        build_ext.build_extensions(self)


class build_cuda_ext(build_ext):
    def build_extensions(self):
        # Use default compiler for CUDA
        build_ext.build_extensions(self)


# Detect platform and set up accordingly
platform = _detect_platform()
print(f"Detected platform: {platform}")

if platform == "hip":
    # HIP/ROCm configuration
    rocm_home = Path(_find_rocm_home())
    include_dirs = [
        str(rocm_home.resolve() / 'include'),
    ]
    library_dirs = [
        str(rocm_home.resolve() / 'lib'),
    ]
    
    ext_modules = [HipExtension(
        'torch_memory_saver_cpp',
        ['csrc/torch_memory_saver_hip.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['amdhip64', 'dl'],
        define_macros=[('Py_LIMITED_API', '0x03090000')],
        py_limited_api=True,
        extra_link_args=['-shared', '-fPIC'],
    )]
    
    cmdclass = {'build_ext': build_hip_ext}
    
else:
    # CUDA configuration
    cuda_home = Path(_find_cuda_home())
    include_dirs = [
        str(cuda_home.resolve() / 'targets/x86_64-linux/include'),
    ]
    library_dirs = [
        str(cuda_home.resolve() / 'lib64'),
        str(cuda_home.resolve() / 'lib64/stubs'),
    ]
    
    ext_modules = [CudaExtension(
        'torch_memory_saver_cpp',
        ['csrc/torch_memory_saver.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['cuda'],
        define_macros=[('Py_LIMITED_API', '0x03090000')],
        py_limited_api=True,
    )]
    
    cmdclass = {'build_ext': build_cuda_ext}


setup(
    name='torch_memory_saver',
    version='0.0.6',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    packages=['torch_memory_saver'],
)
