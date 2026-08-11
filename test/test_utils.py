import os
import sys
from types import SimpleNamespace

import pytest
import torch

from torch_memory_saver import utils
from torch_memory_saver.entrypoint import TorchMemorySaver
from torch_memory_saver import entrypoint as entrypoint_mod
from torch_memory_saver.utils import change_env


def _fake_torch(cuda=None, hip=None):
    return SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip))


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


def test_cpu_backup_backend_rejects_without_enable():
    tms = TorchMemorySaver()
    with pytest.raises(ValueError, match="requires enable_cpu_backup"):
        tms._cpu_backup_backend(False, "mmap")
    assert tms._cpu_backup_backend(False, None) is None


def test_cpu_backup_backend_rejects_unknown():
    tms = TorchMemorySaver()
    with pytest.raises(ValueError, match="must be \"mmap\" or \"pinned\""):
        tms._cpu_backup_backend(True, "disk")  # type: ignore[arg-type]


def test_cpu_backup_backend_defaults_and_env(monkeypatch):
    tms = TorchMemorySaver()
    monkeypatch.delenv("TMS_INIT_CPU_BACKUP_BACKEND", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(entrypoint_mod, "is_xpu", lambda: False)

    assert tms._cpu_backup_backend(True, None) == "pinned"

    monkeypatch.setenv("TMS_INIT_CPU_BACKUP_BACKEND", "")
    assert tms._cpu_backup_backend(True, None) == "pinned"

    monkeypatch.setenv("TMS_INIT_CPU_BACKUP_BACKEND", "mmap")
    assert tms._cpu_backup_backend(True, None) == "mmap"
    assert tms._cpu_backup_backend(True, "pinned") == "pinned"


@pytest.mark.parametrize("platform", ["rocm", "xpu"])
def test_cpu_backup_backend_rejects_mmap_on_rocm_xpu(monkeypatch, platform):
    tms = TorchMemorySaver()
    monkeypatch.delenv("TMS_INIT_CPU_BACKUP_BACKEND", raising=False)
    monkeypatch.setattr(torch.version, "hip", "6.2.0" if platform == "rocm" else None)
    monkeypatch.setattr(entrypoint_mod, "is_xpu", lambda: platform == "xpu")

    assert tms._cpu_backup_backend(True, None) == "pinned"
    with pytest.raises(ValueError, match="not supported on ROCm/XPU"):
        tms._cpu_backup_backend(True, "mmap")
