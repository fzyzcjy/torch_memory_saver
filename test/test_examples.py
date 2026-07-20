import pytest
from contextlib import nullcontext

import multiprocessing
import traceback
import torch
import torch_memory_saver
from torch_memory_saver.utils import change_env

from examples import (
    simple,
    cuda_graph,
    cuda_vmm_granularity,
    cpu_backup,
    rl_example,
    multi_device,
    multi_device_torch_mode,
    training_engine,
    nested_region,
)

_HOOK_MODES = ["preload", "torch"]


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_simple(hook_mode):
    _test_core(simple.run, hook_mode=hook_mode)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cuda_graph(hook_mode):
    _test_core(cuda_graph.run, hook_mode=hook_mode)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_cpu_backup(hook_mode):
    _test_core(cpu_backup.run, hook_mode=hook_mode)


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_multi_device(hook_mode):
    _test_core(multi_device.run, hook_mode=hook_mode)


def test_multi_device_torch_mode():
    # Torch-mode multi-device: pins each device at alloc time. Protects the
    # per-device MemPool keying (skips at runtime if <2 devices are present).
    _test_core(multi_device_torch_mode.run, hook_mode="torch")


@pytest.mark.parametrize("hook_mode", _HOOK_MODES)
def test_rl_example(hook_mode):
    _test_core(rl_example.run, hook_mode=hook_mode)


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
