import builtins
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from torch_memory_saver import utils
from torch_memory_saver.entrypoint import TorchMemorySaver
from torch_memory_saver import entrypoint as entrypoint_mod
from torch_memory_saver.utils import change_env


def _fake_torch(cuda=None, hip=None, musa=None):
    fake = SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip))
    if musa is not None:
        fake.musa = musa
    return fake


def test_mlu_detection_does_not_import_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_mlu" or name.startswith("torch_mlu."):
            raise AssertionError("backend detection must not register torch_mlu")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert utils._is_mlu_torch() is False
    assert utils.is_mlu() is False


def test_mlu_detection_observes_late_registration(monkeypatch):
    fake_torch = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_mlu", SimpleNamespace())
    assert utils._is_mlu_torch() is False
    assert utils.is_mlu() is False
    fake_torch.mlu = SimpleNamespace(is_available=lambda: True)
    assert utils._is_mlu_torch() is True
    assert utils.is_mlu() is True
    fake_torch.mlu.is_available = lambda: False
    assert utils._is_mlu_torch() is True
    assert utils.is_mlu() is False


def test_mlu_detection_respects_disabled_autoload():
    subprocess.run(
        [sys.executable, "-c", """
import sys
import torch
assert 'torch_mlu' not in sys.modules
from torch_memory_saver.utils import _is_mlu_torch, is_mlu
assert not _is_mlu_torch()
is_mlu()
assert 'torch_mlu' not in sys.modules
"""],
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_binary_selection_does_not_probe_devices(monkeypatch, tmp_path, mode):
    stem = f"torch_memory_saver_hook_mode_{mode}"
    package_dir = tmp_path / "torch_memory_saver"
    package_dir.mkdir()
    binary = tmp_path / f"{stem}.abi3.so"
    binary.touch()
    monkeypatch.setattr(utils, "__file__", str(package_dir / "utils.py"))
    def unexpected_device_probe():
        raise AssertionError("binary selection must not query device availability")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        mlu=SimpleNamespace(is_available=unexpected_device_probe)
    ))
    def unexpected_cuda_probe():
        raise AssertionError("MLU must not probe the CUDA runtime")
    monkeypatch.setattr(utils, "_detect_cuda_major", unexpected_cuda_probe)
    assert utils.get_binary_path_from_package(stem) == binary


@pytest.mark.parametrize("enabled", [False, True])
def test_mlu_expandable_segments_uses_runtime_settings(monkeypatch, enabled):
    monkeypatch.setattr(entrypoint_mod, "is_mlu", lambda: True)
    monkeypatch.delenv("PYTORCH_MLU_ALLOC_CONF", raising=False)
    monkeypatch.setattr(entrypoint_mod, "torch", SimpleNamespace(mlu=SimpleNamespace(
        memory=SimpleNamespace(_snapshot=lambda: {
            "allocator_settings": {"expandable_segments": enabled}
        })
    )))
    if enabled:
        with pytest.raises(RuntimeError, match="expandable_segments"):
            entrypoint_mod._sanity_checks()
    else:
        entrypoint_mod._sanity_checks()


@pytest.mark.parametrize(
    "musa,config_name,rejected",
    [
        (False, "PYTORCH_CUDA_ALLOC_CONF", True),
        (False, "PYTORCH_MUSA_ALLOC_CONF", False),
        (True, "PYTORCH_CUDA_ALLOC_CONF", True),
        (True, "PYTORCH_MUSA_ALLOC_CONF", True),
        (False, "PYTORCH_MLU_ALLOC_CONF", False),
        (True, "PYTORCH_MLU_ALLOC_CONF", False),
    ],
)
def test_non_mlu_allocator_checks_preserve_existing_behavior(
    monkeypatch, musa, config_name, rejected
):
    monkeypatch.setattr(entrypoint_mod, "is_mlu", lambda: False)
    monkeypatch.setattr(entrypoint_mod, "is_musa", lambda: musa)
    # No device API is needed for the existing environment-variable checks.
    monkeypatch.setattr(entrypoint_mod, "torch", SimpleNamespace())
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_MUSA_ALLOC_CONF", "PYTORCH_MLU_ALLOC_CONF"
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(config_name, "expandable_segments:True")
    if rejected:
        with pytest.raises(RuntimeError, match="expandable_segments"):
            entrypoint_mod._sanity_checks()
    else:
        entrypoint_mod._sanity_checks()


def test_get_binary_path_from_package_uses_unsuffixed_rocm_binary(monkeypatch, tmp_path):
    stem = "torch_memory_saver_hook_mode_preload"
    package_dir = tmp_path / "torch_memory_saver"
    package_dir.mkdir()
    binary = tmp_path / f"{stem}.abi3.so"
    binary.touch()

    monkeypatch.setattr(utils, "__file__", str(package_dir / "utils.py"))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip="7.2.0"))

    assert utils.get_binary_path_from_package(stem) == binary


def test_get_binary_path_from_package_keeps_cuda_suffix_selection(monkeypatch, tmp_path):
    stem = "torch_memory_saver_hook_mode_preload"
    package_dir = tmp_path / "torch_memory_saver"
    package_dir.mkdir()
    binary = tmp_path / f"{stem}_cu12.abi3.so"
    binary.touch()

    monkeypatch.setattr(utils, "__file__", str(package_dir / "utils.py"))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda="12.9"))

    assert utils.get_binary_path_from_package(stem) == binary


def test_get_binary_path_from_package_uses_unsuffixed_musa_binary(monkeypatch, tmp_path):
    stem = "torch_memory_saver_hook_mode_preload"
    package_dir = tmp_path / "torch_memory_saver"
    package_dir.mkdir()
    binary = tmp_path / f"{stem}.abi3.so"
    binary.touch()

    monkeypatch.setattr(utils, "__file__", str(package_dir / "utils.py"))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(musa=SimpleNamespace()))

    assert utils.get_binary_path_from_package(stem) == binary


def test_change_env_leaves_absent_key_absent(monkeypatch):
    key = "TMS_TEST_CHANGE_ENV_ABSENT"
    monkeypatch.delenv(key, raising=False)

    with change_env(key, "1"):
        assert os.environ[key] == "1"

    assert key not in os.environ


def test_change_env_restores_previous_value(monkeypatch):
    key = "TMS_TEST_CHANGE_ENV_PRESENT"
    monkeypatch.setenv(key, "old")

    with change_env(key, "1"):
        assert os.environ[key] == "1"

    assert os.environ[key] == "old"


def test_cpu_backup_backend_requires_enable():
    tms = TorchMemorySaver()
    with pytest.raises(ValueError, match="requires enable_cpu_backup"):
        with tms.region(enable_cpu_backup=False, cpu_backup_backend="mmap"):
            pass


def test_cpu_backup_backend_rejected_on_xpu(monkeypatch):
    tms = TorchMemorySaver()
    monkeypatch.setattr(entrypoint_mod, "is_xpu", lambda: True)
    with pytest.raises(ValueError, match="not supported on XPU"):
        with tms.region(enable_cpu_backup=True, cpu_backup_backend="pinned"):
            pass
    with pytest.raises(ValueError, match="not supported on XPU"):
        with tms.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
            pass


def test_cpu_backup_backend_rejects_mmap_on_rocm(monkeypatch):
    tms = TorchMemorySaver()
    monkeypatch.setattr(entrypoint_mod, "is_xpu", lambda: False)
    monkeypatch.setattr(torch.version, "hip", "6.2.0")
    with pytest.raises(ValueError, match="not supported on ROCm"):
        with tms.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
            pass


if utils.is_musa():
    # Defined only on MUSA so other backends do not collect MUSA-only skips.

    def test_disable_rejects_region_nesting():
        tms = TorchMemorySaver()
        tms.hook_mode = "torch"
        with tms.region():
            with pytest.raises(RuntimeError, match="inside a region"):
                with tms.disable():
                    pass

    def test_disable_rejects_nested_disable(monkeypatch):
        monkeypatch.setenv("TMS_INIT_ENABLE", "1")
        tms = TorchMemorySaver()
        tms.hook_mode = "torch"
        tms._ensure_initialized()
        tms._impl._binary_wrapper.cdll.tms_set_interesting_region(True)
        with tms.disable():
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with tms.disable():
                    pass

    def test_preload_binary_is_unsuffixed():
        try:
            path = utils.get_binary_path_from_package("torch_memory_saver_hook_mode_preload")
        except RuntimeError:
            pytest.skip("preload hook binary is not built in this tree")
        assert path.name == "torch_memory_saver_hook_mode_preload.abi3.so"
