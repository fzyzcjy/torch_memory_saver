import pytest
from contextlib import nullcontext

import multiprocessing
import sys
import traceback
import torch
import torch_memory_saver
from torch_memory_saver.testing_utils import is_xpu
from torch_memory_saver.utils import change_env

from examples import (
    simple,
    cuda_graph,
    cuda_vmm_granularity,
    cpu_backup,
    disk_backup,
    rl_example,
    multi_device,
    multi_device_torch_mode,
    training_engine,
    nested_region,
    xpu_scenarios,
)

# XPU only supports hook_mode='torch'
_IS_XPU = is_xpu()
_HOOK_MODES = ["torch"] if _IS_XPU else ["preload", "torch"]

# Skip reason for tests that exercise CUDA/HIP-only paths on XPU.
_skip_on_xpu = pytest.mark.skipif(_IS_XPU, reason="CUDA/HIP-only path, not supported on XPU")
_xpu_only = pytest.mark.skipif(not _IS_XPU, reason="XPU-specific path")
_device_module = torch.xpu if _IS_XPU else torch.cuda
_multi_device_only = pytest.mark.skipif(
    _device_module.device_count() < 2,
    reason="Multi-device test requires at least two devices",
)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_simple(hook_mode):
    _test_core(simple.run, hook_mode=hook_mode)


@_skip_on_xpu
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cuda_graph(hook_mode):
    _test_core(cuda_graph.run, hook_mode=hook_mode)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cpu_backup(hook_mode):
    _test_core(cpu_backup.run, hook_mode=hook_mode)


@_skip_on_xpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or sys.platform != "linux",
    reason="mmap RSS reclaim is CUDA-only; needs Linux /proc",
)
def test_cpu_backup_rss():
    _test_core(cpu_backup.run_rss, hook_mode="torch")


@_skip_on_xpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="TMS_INIT_CPU_BACKUP_BACKEND=mmap is CUDA-only",
)
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cpu_backup_backend_from_env(hook_mode):
    with change_env("TMS_INIT_CPU_BACKUP_BACKEND", "mmap"):
        _test_core(cpu_backup.run_backend_from_env, hook_mode=hook_mode)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_disk_backup(hook_mode):
    _test_core(disk_backup.run, hook_mode=hook_mode)


@_skip_on_xpu
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cpu_backup_retain_from_env(hook_mode):
    with change_env("TMS_RETAIN_CPU_BACKUP", "1"):
        _test_core(cpu_backup.run_retain_from_env, hook_mode=hook_mode)


@_skip_on_xpu
@_multi_device_only
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_multi_device(hook_mode):
    _test_core(multi_device.run, hook_mode=hook_mode)


@_multi_device_only
def test_multi_device_torch_mode():
    _test_core(multi_device_torch_mode.run, hook_mode="torch")


@_xpu_only
def test_disable_unsupported_xpu():
    _test_core(xpu_scenarios.run_disable_unsupported, hook_mode="torch")


@_xpu_only
def test_resume_failure_injection_xpu():
    _test_core(xpu_scenarios.run_resume_failure, hook_mode="torch")


@_xpu_only
def test_cleanup_failure_injection_xpu():
    _test_core(xpu_scenarios.run_cleanup_failure, hook_mode="torch")


@_xpu_only
def test_free_failure_injection_xpu():
    _test_core(xpu_scenarios.run_free_failure, hook_mode="torch")


@_xpu_only
def test_multi_device_sync_xpu():
    _test_core(xpu_scenarios.run_multi_device_sync, hook_mode="torch")


@_xpu_only
def test_memory_margin_unsupported_xpu():
    _test_core(xpu_scenarios.run_memory_margin_unsupported, hook_mode="torch")


@_skip_on_xpu
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_rl_example(hook_mode):
    _test_core(rl_example.run, hook_mode=hook_mode)


@_skip_on_xpu
def test_training_engine():
    with (
        change_env("TMS_INIT_ENABLE", "1"),
        change_env("TMS_INIT_ENABLE_CPU_BACKUP", "1")
    ):
        _test_core(training_engine.run, hook_mode="preload")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VMM free-while-paused / granularity test needs a GPU",
)
def test_cuda_vmm_granularity():
    with change_env("TMS_INIT_ENABLE", "1"):
        _test_core(cuda_vmm_granularity.run, hook_mode="preload")


@_skip_on_xpu
def test_nested_region():
    with (
        change_env("TMS_INIT_ENABLE", "1"),
        change_env("TMS_INIT_ENABLE_CPU_BACKUP", "1")
    ):
        _test_core(nested_region.run, hook_mode="preload")


def _test_core(fn, hook_mode):
    ctx = torch_memory_saver.configure_subprocess() if hook_mode == "preload" else nullcontext()
    with ctx:
        _run_in_subprocess(fn, fn_kwargs=dict(hook_mode=hook_mode))


def _run_in_subprocess(fn, fn_kwargs):
    ctx = multiprocessing.get_context('spawn')
    output_queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_fn_wrapper, args=(fn, fn_kwargs, output_queue))
    proc.start()
    proc.join()
    success = output_queue.get() if fn_kwargs.get("hook_mode", "torch") == "torch" else proc.exitcode == 0
    assert success


def _subprocess_fn_wrapper(fn, fn_kwargs, output_queue):
    try:
        print(f"Subprocess execution start")
        fn(**fn_kwargs)
        print(f"Subprocess execution end (may see error messages when CUDA exit which is normal)", flush=True)
        output_queue.put(True)
    except Exception as e:
        print(f"Subprocess has error: {e}", flush=True)
        traceback.print_exc()
        output_queue.put(False)
        raise
