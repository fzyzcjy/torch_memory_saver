import sys
from types import SimpleNamespace

from torch_memory_saver import utils
from torch_memory_saver.entrypoint import _TorchMemorySaverImpl


def _fake_torch(cuda=None, hip=None):
    return SimpleNamespace(version=SimpleNamespace(cuda=cuda, hip=hip))


def test_cuda_exit_cleanup_order():
    calls = []
    impl = object.__new__(_TorchMemorySaverImpl)
    impl._binary_wrapper = SimpleNamespace(
        cdll=SimpleNamespace(
            tms_set_interesting_region=lambda value: calls.append(("region", value)),
            tms_release_cpu_backups=lambda: calls.append(("release",)),
        )
    )
    impl._mem_pools = SimpleNamespace(clear=lambda: calls.append(("clear",)))

    impl._cleanup_cuda_at_exit()

    assert calls == [("region", True), ("clear",), ("release",)]


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
