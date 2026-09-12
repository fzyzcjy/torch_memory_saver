"""The shipped CUDA preload hook must resolve libcudart when LD_PRELOADed into a fresh process.

Linux only; needs the installed package and a loadable libcudart.so.<major> (e.g. torch installed).
"""

import ctypes
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import torch_memory_saver
from torch_memory_saver import utils
from torch_memory_saver.hooks import mode_preload
from torch_memory_saver.hooks.mode_preload import configure_subprocess

_DT_NEEDED, _DT_RPATH, _DT_RUNPATH = 1, 15, 29
_SHT_DYNAMIC = 6
_PACKAGE_DIR = Path(torch_memory_saver.__file__).parent

_CHILD = "\n".join([
    "paths = []",
    "for line in open('/proc/self/maps'):",
    "    fields = line.split(maxsplit=5)",
    "    if len(fields) == 6 and ('libcudart.so' in fields[5] or 'torch_memory_saver_hook' in fields[5]):",
    "        paths.append(fields[5].strip())",
    "print('\\n'.join(dict.fromkeys(paths)))",
])


def _runtime_rel(major):
    return f"nvidia/cu{major}/lib" if major >= 13 else "nvidia/cuda_runtime/lib"


def _shipped_cuda_hooks():
    """(path, CUDA major) of every installed CUDA hook, unsuffixed CUDA 12 compatibility copies included."""
    hooks = {}
    for root in (_PACKAGE_DIR, _PACKAGE_DIR.parent):
        for path in root.glob("torch_memory_saver_hook_mode_*.abi3.so"):
            hooks.setdefault(path.name, path)
    if not any("_cu" in name for name in hooks):
        return []
    result = []
    for name, path in sorted(hooks.items()):
        match = re.search(r"_cu(\d+)\.abi3\.so$", name)
        result.append((path, int(match.group(1)) if match else 12))
    return result


_CUDA_HOOKS = _shipped_cuda_hooks()

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="LD_PRELOAD loader test needs Linux"),
    pytest.mark.skipif(not _CUDA_HOOKS, reason="no CUDA hooks installed (ROCm/XPU build)"),
]


def _dynamic_section(data):
    """[(file offset, tag, value)] of the .dynamic entries plus the .dynstr bytes of a little-endian ELF64 object."""
    assert data[:6] == b"\x7fELF\x02\x01", "expected a little-endian ELF64 object"
    (shoff,) = struct.unpack_from("<Q", data, 40)
    shentsize, shnum = struct.unpack_from("<HH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    dynamic = next(h for h in headers if h[1] == _SHT_DYNAMIC)
    strtab = headers[dynamic[6]]
    entries = []
    for offset in range(dynamic[4], dynamic[4] + dynamic[5], 16):
        tag, value = struct.unpack_from("<qQ", data, offset)
        if tag == 0:
            break
        entries.append((offset, tag, value))
    return entries, data[strtab[4]:strtab[4] + strtab[5]]


def _dynamic_strings(path, wanted_tag):
    entries, strings = _dynamic_section(path.read_bytes())
    return [strings[value:strings.index(b"\0", value)].decode() for _, tag, value in entries if tag == wanted_tag]


def _empty_runpath(path):
    """Point DT_RUNPATH/DT_RPATH at the empty string, leaving every other byte intact."""
    data = bytearray(path.read_bytes())
    entries, _ = _dynamic_section(bytes(data))
    for offset, tag, _ in entries:
        if tag in (_DT_RPATH, _DT_RUNPATH):
            struct.pack_into("<Q", data, offset + 8, 0)
    path.write_bytes(data)


def _env_without_cudart_dirs(env):
    """Copy of `env` without LD_PRELOAD and without any LD_LIBRARY_PATH directory that holds libcudart."""
    env = {key: value for key, value in env.items() if key != "LD_PRELOAD"}
    kept = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d and not any(Path(d).glob("libcudart.so*"))]
    if kept:
        env["LD_LIBRARY_PATH"] = ":".join(kept)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def _preload(hook, env):
    return subprocess.run(
        [sys.executable, "-c", _CHILD],
        env={**env, "LD_PRELOAD": str(hook)},
        capture_output=True,
        text=True,
    )


def _mapped(result, needle):
    return [Path(p) for p in result.stdout.split() if needle in p]


@pytest.fixture(scope="module")
def cuda_major():
    return utils._detect_cuda_major()


@pytest.fixture(scope="module")
def soname(cuda_major):
    return f"libcudart.so.{cuda_major}"


@pytest.fixture(scope="module")
def installed_hook():
    return Path(utils.get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))


@pytest.fixture(scope="module")
def runtime_dir(soname):
    try:
        ctypes.CDLL(soname)
    except OSError:
        pytest.skip(f"cannot load {soname} in the test process")
    found = mode_preload._mapped_cudart_dir()
    assert found is not None, f"{soname} loaded but not found in /proc/self/maps"
    return Path(found)


@pytest.fixture(scope="module")
def loader_cannot_resolve_cudart(soname):
    """The loader tests only prove something when the default search path cannot already find libcudart."""
    cache = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    if soname in cache:
        pytest.skip(f"ldconfig already resolves {soname}")


@pytest.mark.parametrize("hook, major", _CUDA_HOOKS, ids=[path.name for path, _ in _CUDA_HOOKS])
def test_shipped_cuda_hook_carries_origin_runpath(hook, major):
    rel = _runtime_rel(major)
    assert f"libcudart.so.{major}" in _dynamic_strings(hook, _DT_NEEDED), hook
    assert _dynamic_strings(hook, _DT_RUNPATH) == [f"$ORIGIN/{rel}:$ORIGIN/../{rel}"], hook


def test_colocated_hook_preloads_without_ld_library_path(
    installed_hook, cuda_major, soname, loader_cannot_resolve_cudart
):
    """pip layout: wheel and NVIDIA runtime wheel share a site-packages root, so DT_RUNPATH alone resolves libcudart."""
    rel = _runtime_rel(cuda_major)
    candidates = [installed_hook.parent / rel / soname, installed_hook.parent.parent / rel / soname]
    colocated = [path for path in candidates if path.exists()]
    if not colocated:
        pytest.skip(f"{installed_hook} is not co-located with {rel}")

    result = _preload(installed_hook, _env_without_cudart_dirs(os.environ))

    assert result.returncode == 0, result.stderr
    assert _mapped(result, "torch_memory_saver_hook") == [installed_hook.resolve()]
    assert _mapped(result, "libcudart.so") == [colocated[0].resolve()]


def test_split_site_packages_needs_runtime_dir_injection(
    tmp_path, monkeypatch, installed_hook, soname, runtime_dir, loader_cannot_resolve_cudart
):
    """Hook in a site-packages root without nvidia/: DT_RUNPATH cannot help, configure_subprocess() must inject the runtime dir."""
    hook = tmp_path / "site_a" / installed_hook.name
    hook.parent.mkdir()
    shutil.copy2(installed_hook, hook)
    base_env = _env_without_cudart_dirs(os.environ)

    failed = _preload(hook, base_env)
    assert failed.returncode != 0
    assert f"{soname}: cannot open shared object file" in failed.stderr

    monkeypatch.setattr(mode_preload, "get_binary_path_from_package", lambda stem: hook)
    with patch.dict(os.environ, base_env, clear=True):
        with configure_subprocess():
            assert os.environ["LD_PRELOAD"] == str(hook)
            assert Path(os.environ["LD_LIBRARY_PATH"].split(":")[0]) == runtime_dir
            child_env = dict(os.environ)
        assert os.environ.get("LD_LIBRARY_PATH") == base_env.get("LD_LIBRARY_PATH")

    passed = _preload(hook, child_env)
    assert passed.returncode == 0, passed.stderr
    assert _mapped(passed, "torch_memory_saver_hook") == [hook.resolve()]
    assert _mapped(passed, "libcudart.so") == [(runtime_dir / soname).resolve()]


def test_hook_with_emptied_runpath_cannot_preload(
    tmp_path, installed_hook, cuda_major, soname, runtime_dir, loader_cannot_resolve_cudart
):
    """Negative control: same co-located layout and bytes with DT_RUNPATH emptied must fail in the loader."""
    site = tmp_path / "site"
    hook = site / installed_hook.name
    lib_dir = site / _runtime_rel(cuda_major)
    lib_dir.mkdir(parents=True)
    shutil.copy2(installed_hook, hook)
    (lib_dir / soname).symlink_to(runtime_dir / soname)
    env = _env_without_cudart_dirs(os.environ)

    passed = _preload(hook, env)
    assert passed.returncode == 0, passed.stderr
    assert _mapped(passed, "libcudart.so") == [(runtime_dir / soname).resolve()]

    _empty_runpath(hook)
    assert _dynamic_strings(hook, _DT_RUNPATH) == [""]
    failed = _preload(hook, env)
    assert failed.returncode != 0
    assert f"{soname}: cannot open shared object file" in failed.stderr
