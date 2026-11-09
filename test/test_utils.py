import os
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
