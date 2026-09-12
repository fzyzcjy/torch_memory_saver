"""Regression test: configure_subprocess() must preserve existing LD_PRELOAD entries."""
import os
from unittest.mock import patch

import pytest

FAKE_LIB = "/fake/torch_memory_saver_hook_mode_preload.so"


def _invoke(initial_ld_preload):
    """Call configure_subprocess() and return the LD_PRELOAD seen inside the context."""
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    env_patch = {} if initial_ld_preload is None else {"LD_PRELOAD": initial_ld_preload}
    with patch.dict(os.environ, env_patch, clear=False):
        if initial_ld_preload is None:
            os.environ.pop("LD_PRELOAD", None)
        with patch(
            "torch_memory_saver.hooks.mode_preload.get_binary_path_from_package",
            return_value=FAKE_LIB,
        ):
            with configure_subprocess():
                return os.environ.get("LD_PRELOAD", "")


def test_sets_ld_preload_when_unset():
    """LD_PRELOAD unset → should be set to the lib path only."""
    result = _invoke(None)
    assert result == FAKE_LIB


def test_prepends_without_overwriting_existing():
    """LD_PRELOAD already set → TMS lib is prepended, existing entry preserved."""
    existing = "/other/lib.so"
    result = _invoke(existing)
    parts = result.split(":")
    assert parts[0] == FAKE_LIB, "TMS lib must come first"
    assert existing in parts, "Pre-existing LD_PRELOAD entry must not be dropped"


def test_restores_ld_preload_after_context():
    """After the context exits, LD_PRELOAD must be restored to its original value."""
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    original = "/original/lib.so"
    with patch.dict(os.environ, {"LD_PRELOAD": original}):
        with patch(
            "torch_memory_saver.hooks.mode_preload.get_binary_path_from_package",
            return_value=FAKE_LIB,
        ):
            with configure_subprocess():
                assert os.environ["LD_PRELOAD"] != original
        assert os.environ.get("LD_PRELOAD") == original


def test_prepends_mapped_cudart_dir_and_restores(monkeypatch):
    """Split site-packages layout: the child must see the parent's libcudart dir first on LD_LIBRARY_PATH."""
    from torch_memory_saver.hooks import mode_preload

    monkeypatch.setattr(mode_preload, "_mapped_cudart_dir", lambda: "/site_b/nvidia/cu13/lib")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    with patch("torch_memory_saver.hooks.mode_preload.get_binary_path_from_package", return_value=FAKE_LIB):
        with mode_preload.configure_subprocess():
            assert os.environ["LD_LIBRARY_PATH"] == "/site_b/nvidia/cu13/lib:/existing/lib"
    assert os.environ["LD_LIBRARY_PATH"] == "/existing/lib"


def test_no_mapped_cudart_leaves_ld_library_path_untouched(monkeypatch):
    from torch_memory_saver.hooks import mode_preload

    monkeypatch.setattr(mode_preload, "_mapped_cudart_dir", lambda: None)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    with patch("torch_memory_saver.hooks.mode_preload.get_binary_path_from_package", return_value=FAKE_LIB):
        with mode_preload.configure_subprocess():
            assert "LD_LIBRARY_PATH" not in os.environ


def test_mapped_cudart_dir_parses_proc_maps(tmp_path):
    from torch_memory_saver.hooks.mode_preload import _mapped_cudart_dir

    maps = tmp_path / "maps"
    maps.write_text(
        "aaaa0000-aaaa1000 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n"
        "aaaa2000-aaaa3000 rw-p 00000000 00:00 0\n"
        "aaaa4000-aaaa5000 r-xp 00000000 08:01 2 /site_b/nvidia/cu13/lib/libcudart.so.13\n"
    )
    assert _mapped_cudart_dir(str(maps)) == "/site_b/nvidia/cu13/lib"
    assert _mapped_cudart_dir(str(tmp_path / "absent")) is None
