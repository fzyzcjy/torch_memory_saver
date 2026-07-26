import pytest
from contextlib import nullcontext

import multiprocessing
import sys
import traceback
from pathlib import Path

import torch
import torch_memory_saver
from torch_memory_saver.testing_utils import is_xpu
from torch_memory_saver.utils import change_env

# Import examples by absolute path rather than `from examples import ...`: the
# latter collides with Intel's pyzes `examples` package on sys.path (XPU envs).
sys.path.insert(0, str(Path(__file__).resolve().parent / "examples"))

import simple
import cuda_graph
import cuda_vmm_granularity
import cpu_backup
import disk_backup
import rl_example
import multi_device
import training_engine
import nested_region
import multi_device_torch_mode
# All XPU-only scenarios live in one module (run_* per scenario), each still run
# in its own spawned subprocess so a fault-injection abort can't affect others.
import xpu_scenarios

# XPU only supports hook_mode='torch' (in-process pluggable allocator);
# LD_PRELOAD-based preload is CUDA/HIP-only.
_IS_XPU = is_xpu()
_HOOK_MODES = ["torch"] if _IS_XPU else ["preload", "torch"]

# Skip reason for tests that exercise CUDA/HIP-only paths on XPU.
_skip_on_xpu = pytest.mark.skipif(_IS_XPU, reason="CUDA/HIP-only path, not supported on XPU")
# Mirror: tests that exercise an XPU-specific path (skipped elsewhere).
_xpu_only = pytest.mark.skipif(not _IS_XPU, reason="XPU-specific path")


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_simple(hook_mode):
    _test_core(simple.run, hook_mode=hook_mode)


@_skip_on_xpu
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cuda_graph(hook_mode):
    # Pauseable graph capture is preload-only and CUDA/HIP-specific.
    _test_core(cuda_graph.run, hook_mode=hook_mode)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cpu_backup(hook_mode):
    _test_core(cpu_backup.run, hook_mode=hook_mode)


@_skip_on_xpu
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_disk_backup(hook_mode):
    _test_core(disk_backup.run, hook_mode=hook_mode)


@_skip_on_xpu
@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_multi_device(hook_mode):
    # Allocates on a non-current device in one region -- only the preload (global
    # cudaMalloc) hook captures that; XPU/torch-mode uses test_multi_device_torch_mode.
    _test_core(multi_device.run, hook_mode=hook_mode)


def test_multi_device_torch_mode():
    # Torch-mode multi-device (pins each device at alloc time), so it runs on
    # both CUDA and XPU. Skips at runtime if <2 devices are present.
    _test_core(multi_device_torch_mode.run, hook_mode="torch")


@_xpu_only
def test_disable_unsupported_xpu():
    # disable() is rejected (NotImplementedError) on XPU, not left as a
    # process-killing path; the CUDA-only training_engine test covers it elsewhere.
    _test_core(xpu_scenarios.run_disable_unsupported, hook_mode="torch")


@_xpu_only
def test_resume_failure_injection_xpu():
    # resume() is transactional on XPU: injected create/map/restore failures roll
    # back to a clean PAUSED state (backup intact) and raise, never half-resume.
    _test_core(xpu_scenarios.run_resume_failure, hook_mode="torch")


@_xpu_only
def test_cleanup_failure_injection_xpu():
    # pause()/free() retain ownership until Level Zero confirms release: a destroy
    # failure keeps the handle (shown as leaked bytes) and re-maps it on resume.
    _test_core(xpu_scenarios.run_cleanup_failure, hook_mode="torch")


@_xpu_only
def test_free_failure_injection_xpu():
    # free() is transactional: injected unmap/destroy/free-VA failures RETAIN the
    # record (proven via tms_xpu_tracked_bytes) and each retry advances one step.
    _test_core(xpu_scenarios.run_free_failure, hook_mode="torch")


@_xpu_only
def test_multi_device_sync_xpu():
    # pause()/resume() drain EXACTLY the devices the backend unmaps (from
    # tms_xpu_affected_devices), incl. non-current ones. Skips if <2 devices.
    _test_core(xpu_scenarios.run_multi_device_sync, hook_mode="torch")


@_xpu_only
def test_memory_margin_unsupported_xpu():
    # memory_margin_bytes is rejected (NotImplementedError) on XPU, not silently
    # ignored: the OOM-margin guard needs free-bytes the Intel driver reports frozen.
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
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="CUDA VMM test requires a CUDA GPU",
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
