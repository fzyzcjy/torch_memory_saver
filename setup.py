import logging
import os
import shutil
from pathlib import Path
import setuptools
from setuptools import setup
from setuptools.command.build_ext import build_ext

logger = logging.getLogger(__name__)


# copy & modify from torch/utils/cpp_extension.py
def _find_platform_home(platform):
    """Find the install path for the specified platform (cuda/rocm/xpu)."""
    if platform == "cuda":
        # Find CUDA home
        home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if home is None:
            compiler_path = shutil.which("nvcc")
            if compiler_path is not None:
                home = os.path.dirname(os.path.dirname(compiler_path))
            else:
                home = '/usr/local/cuda'
    elif platform == "xpu":
        # Find oneAPI home (for icpx + SYCL headers/libs)
        home = os.environ.get('ONEAPI_ROOT')
        if home is None:
            icpx = _find_icpx()
            if icpx is not None:
                # bin/ -> latest/ -> compiler/ -> oneapi root
                home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(icpx))))
            else:
                home = '/opt/intel/oneapi'
    else:  # rocm/hip
        # Find ROCm home
        home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
        if home is None:
            compiler_path = shutil.which("hipcc")
            if compiler_path is not None:
                home = os.path.dirname(os.path.dirname(compiler_path))
            else:
                home = '/opt/rocm'
    return home


def _find_icpx():
    """Return the absolute path to icpx (Intel oneAPI SYCL compiler).

    Honors $ICPX, then $PATH, then $ONEAPI_ROOT and common install locations.
    """
    explicit = os.environ.get("ICPX")
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("icpx")
    if found:
        return found
    oneapi_root = os.environ.get("ONEAPI_ROOT")
    search_roots = ([oneapi_root] if oneapi_root else []) + ['/opt/intel/oneapi']
    for root in search_roots:
        guess = os.path.join(root, 'compiler', 'latest', 'bin', 'icpx')
        if os.path.isfile(guess):
            return guess
    return None


def _detect_platform():
    """Detect whether to use CUDA, HIP or XPU based on available tools."""
    # Allow explicit override.
    forced = os.environ.get("TMS_PLATFORM")
    if forced:
        return forced
    # Check for HIP first (since it might be preferred on AMD systems)
    if shutil.which("hipcc") is not None:
        return "hip"
    elif shutil.which("nvcc") is not None:
        return "cuda"
    elif _find_icpx() is not None:
        return "xpu"
    else:
        # Default to CUDA if neither is found
        return "cuda"


class PlatformExtension(setuptools.Extension):
    """Unified extension class for both CUDA and HIP platforms."""
    def __init__(self, name, sources, platform="cuda", *args, **kwargs):
        self.platform = platform
        super().__init__(name, sources, *args, **kwargs)


class build_platform_ext(build_ext):
    """Unified build extension class that handles both CUDA and HIP."""
    
    def __init__(self, dist, platform="cuda"):
        super().__init__(dist)
        self.platform = platform
    
    def build_extensions(self):
        if self.platform == "hip":
            # Set hipcc as the compiler for HIP
            self.compiler.set_executable("compiler_so", "hipcc")
            self.compiler.set_executable("compiler_cxx", "hipcc")
            self.compiler.set_executable("linker_so", "hipcc --shared")

            # Add extra compiler and linker flags for HIP
            for ext in self.extensions:
                ext.extra_compile_args = ['-fPIC']
                ext.extra_link_args = ['-shared']

        if self.platform == "xpu":
            # Set icpx (Intel oneAPI SYCL compiler) for XPU.
            icpx = _find_icpx()
            self.compiler.set_executable("compiler_so", icpx)
            self.compiler.set_executable("compiler_cxx", icpx)
            self.compiler.set_executable("linker_so", f"{icpx} -shared")
            for ext in self.extensions:
                ext.extra_compile_args = ['-fPIC', '-fsycl', '-fsycl-targets=spir64']
                ext.extra_link_args = ['-fsycl', '-fsycl-targets=spir64', '-shared']
        # For CUDA, use default compiler (no special setup needed)

        build_ext.build_extensions(self)

    def finalize_options(self):
        # Must set CC/CXX BEFORE super().finalize_options(): that is where
        # setuptools calls new_compiler() + customize_compiler(), reading CC/CXX
        # from os.environ. Setting them in build_extensions() is too late.
        if self.platform == "xpu":
            icpx = _find_icpx()
            if icpx is None:
                raise RuntimeError(
                    "Cannot find icpx (Intel oneAPI C++ compiler). "
                    "Install oneAPI, source setvars.sh, or set ICPX=/path/to/icpx."
                )
            os.environ["CC"] = icpx
            os.environ["CXX"] = icpx
            os.environ["LDSHARED"] = icpx + " -shared"
        super().finalize_options()


def _create_ext_modules(platform):
    """Create extension modules based on the specified platform."""
    
    # Common sources for all extensions
    sources = [
        'csrc/api_forwarder.cpp',
        'csrc/core.cpp',
        'csrc/disk_backend.cpp',
        'csrc/entrypoint.cpp',
    ]
    
    # Common define macros
    common_macros = [('Py_LIMITED_API', '0x03090000')]

    # Common compile arguments
    extra_compile_args = ['-std=c++17', '-O3']
    
    # Platform-specific configurations
    platform_home = Path(_find_platform_home(platform))
    
    if platform == "hip":
        # Add ROCm-specific source file for legacy chunked allocation (ROCm 6.x)
        sources.append('csrc/hardware_amd_support.cpp')
        include_dirs = [str(platform_home.resolve() / 'include')]
        library_dirs = [str(platform_home.resolve() / 'lib')]
        libraries = ['amdhip64', 'dl']
        platform_macros = [('USE_ROCM', '1'), ('__HIP_PLATFORM_AMD__', '1')]
    elif platform == "xpu":
        # Add Intel XPU (Level Zero) source file.
        sources.append('csrc/hardware_xpu_support.cpp')
        oneapi_root = str(platform_home)
        include_dirs = [
            os.path.join(oneapi_root, 'compiler', 'latest', 'include'),
            os.path.join(oneapi_root, 'compiler', 'latest', 'include', 'sycl'),
        ]
        library_dirs = [
            os.path.join(oneapi_root, 'compiler', 'latest', 'lib'),
        ]
        # Level Zero headers (ze_api.h / zes_api.h) usually live in the system
        # include path (e.g. /usr/include/level_zero) and link against ze_loader.
        libraries = ['sycl', 'ze_loader']
        platform_macros = [('USE_XPU', '1')]
    else:  # cuda
        include_dirs = [str((platform_home / 'include').resolve())]
        library_dirs = [
            str((platform_home / 'lib64').resolve()),
            str((platform_home / 'lib64/stubs').resolve()),
        ]
        libraries = ['cuda', 'cudart']
        platform_macros = [('USE_CUDA', '1')]
    
    # Suffix the compiled .so files with the CUDA major they link against, so
    # a single wheel can ship cu12 and cu13 variants side-by-side. `utils.py`
    # picks the right one at runtime via torch.version.cuda / libcudart probe.
    # ROCm builds are single-variant for now and use unsuffixed binaries.
    if platform == "cuda":
        cuda_major = os.environ.get("TMS_CUDA_MAJOR")
        if not cuda_major:
            raise RuntimeError(
                "TMS_CUDA_MAJOR env var must be set for CUDA builds "
                "(use `make build-wheel-multi-cuda` or scripts/build_multi_cuda.sh)."
            )
        name_suffix = f"_cu{cuda_major}"
    else:
        name_suffix = ""

    # XPU only supports hook_mode='torch' (in-process pluggable allocator);
    # preload/LD_PRELOAD interception of cudaMalloc is CUDA/HIP-only.
    hook_variants = [
        (f'torch_memory_saver_hook_mode_torch{name_suffix}', [('TMS_HOOK_MODE_TORCH', '1')]),
    ]
    if platform != "xpu":
        hook_variants.insert(
            0,
            (f'torch_memory_saver_hook_mode_preload{name_suffix}', [('TMS_HOOK_MODE_PRELOAD', '1')]),
        )

    ext_modules = [
        PlatformExtension(
            name,
            sources,
            platform=platform,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            define_macros=[
                *common_macros,
                *platform_macros,
                *extra_macros,
            ],
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
        )
        for name, extra_macros in hook_variants
    ]

    return ext_modules


# Detect platform and set up accordingly
platform = _detect_platform()
print(f"Detected platform: {platform}")

# Create extension modules using unified function
ext_modules = _create_ext_modules(platform)

# Create unified build command class instance
class build_ext_for_platform(build_platform_ext):
    def __init__(self, dist):
        super().__init__(dist, platform=platform)

setup(
    name='torch_memory_saver',
    version='0.0.9.post1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext_for_platform},
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["torch_memory_saver", "torch_memory_saver.*"]),
)
