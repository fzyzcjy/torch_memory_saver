"""CUDA extensions carry an $ORIGIN DT_RUNPATH so the LD_PRELOAD hook finds the pip CUDA runtime."""

import importlib.util
from pathlib import Path

import setuptools


def _setup_ext_modules(monkeypatch, platform, cuda_major=None):
    monkeypatch.setenv("TMS_PLATFORM", platform)
    monkeypatch.delenv("TMS_CUDA_MAJOR", raising=False)
    if cuda_major is not None:
        monkeypatch.setenv("TMS_CUDA_MAJOR", cuda_major)
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    spec = importlib.util.spec_from_file_location(
        f"tms_setup_{platform}_{cuda_major}", Path(__file__).parents[1] / "setup.py"
    )
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return captured["ext_modules"]


def _rpath_args(runtime_rel):
    return [
        "-Wl,--enable-new-dtags",
        f"-Wl,-rpath,$ORIGIN/{runtime_rel}",
        f"-Wl,-rpath,$ORIGIN/../{runtime_rel}",
    ]


def test_cu13_extensions_carry_pip_runtime_rpath(monkeypatch):
    ext_modules = _setup_ext_modules(monkeypatch, "cuda", cuda_major="13")
    assert {ext.name for ext in ext_modules} == {
        "torch_memory_saver_hook_mode_preload_cu13",
        "torch_memory_saver_hook_mode_torch_cu13",
    }
    for ext in ext_modules:
        assert ext.extra_link_args == _rpath_args("nvidia/cu13/lib"), ext.name


def test_cu12_extensions_carry_component_wheel_rpath(monkeypatch):
    ext_modules = _setup_ext_modules(monkeypatch, "cuda", cuda_major="12")
    for ext in ext_modules:
        assert ext.extra_link_args == _rpath_args("nvidia/cuda_runtime/lib"), ext.name


def test_hip_extensions_are_untouched(monkeypatch):
    ext_modules = _setup_ext_modules(monkeypatch, "hip")
    for ext in ext_modules:
        assert ext.extra_link_args == [], ext.name
