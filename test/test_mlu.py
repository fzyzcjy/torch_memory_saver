"""Real-device VMM lifecycle checks, isolated from the pytest allocator."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_mlu(code, mode, allocator_config=None, device_count=1):
    from torch_memory_saver.utils import is_mlu
    import torch
    if not is_mlu():
        pytest.skip("MLU-specific path")
    if torch.mlu.device_count() < device_count:
        pytest.skip("Multi-device test requires at least two devices")
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
    )
    if mode == "preload":
        binaries = list(root.glob("torch_memory_saver_hook_mode_preload.*.so"))
        assert len(binaries) == 1
        env["LD_PRELOAD"] = str(binaries[0])
    if allocator_config is not None:
        for name in (
            "PYTORCH_MLU_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
        ):
            env.pop(name, None)
        env.update(allocator_config)
    subprocess.run(
        [sys.executable, "-c", code.replace("MODE", repr(mode))],
        env=env, check=True, timeout=120,
    )


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_tagged_release_restore(mode):
    code = '''
import gc
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
torch.mlu.set_device(0)
with saver.region("weights", enable_cpu_backup=True):
    weights = torch.full((64 * 1024 * 1024,), 7, dtype=torch.uint8, device="mlu")
with saver.region("kv_cache"):
    cache = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="mlu")
wp, cp = weights.data_ptr(), cache.data_ptr()
torch.mlu.synchronize()
free_before = torch.mlu.mem_get_info()[0]
for cycle in range(3):
    torch.mlu.empty_cache()
    free_before = torch.mlu.mem_get_info()[0]
    saver.pause("weights")
    free_paused = torch.mlu.mem_get_info()[0]
    assert free_paused - free_before >= 64 * 1024 * 1024, (free_before, free_paused)
    assert torch.count_nonzero(cache).item() == 0
    saver.resume("weights")
    assert weights.data_ptr() == wp
    assert torch.all(weights == 7).item()
    torch.mlu.empty_cache()
    free_before_cache = torch.mlu.mem_get_info()[0]
    saver.pause("kv_cache")
    assert torch.mlu.mem_get_info()[0] - free_before_cache >= 64 * 1024 * 1024
    assert torch.all(weights == 7).item()
    saver.resume("kv_cache")
    assert cache.data_ptr() == cp
    cache.zero_()
    torch.mlu.synchronize()
    print("cycle", cycle, "released bytes", free_paused - free_before, flush=True)
if MODE == "preload":
    x = torch.ones(16 * 1024 * 1024, device="mlu")
    stream = torch.mlu.Stream()
    with torch.mlu.stream(stream):
        for _ in range(3):
            y = x + 2
    torch.mlu.synchronize()
    del y
    graph = torch.mlu.MLUGraph()
    with saver.cuda_graph(graph, stream=stream, tag="cuda_graph", enable_cpu_backup=True):
        y = x + 2
    graph.replay()
    torch.mlu.synchronize()
    assert torch.all(y == 3).item()
    yp = y.data_ptr()
    for _ in range(3):
        torch.mlu.empty_cache()
        before = torch.mlu.mem_get_info()[0]
        saver.pause("cuda_graph")
        released = torch.mlu.mem_get_info()[0] - before
        assert released >= 64 * 1024 * 1024, released
        saver.resume("cuda_graph")
        graph.replay()
        torch.mlu.synchronize()
        assert y.data_ptr() == yp and torch.all(y == 3).item()
    print("MLU graph replay after restore passed", released, flush=True)
saver.pause()
del weights, cache
gc.collect()
saver.resume()
print("MLU lifecycle passed", MODE, flush=True)
'''
    _run_mlu(code, mode)


def test_mlu_preload_from_parent_without_visible_devices():
    from torch_memory_saver.utils import is_mlu

    if not is_mlu():
        pytest.skip("MLU-specific path")
    child_code = """
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
assert torch.mlu.is_available()
with saver.region("child", enable_cpu_backup=True):
    x = torch.full((1024,), 7, device="mlu")
pointer = x.data_ptr()
saver.pause("child")
assert torch.all(saver.get_cpu_backup(x) == 7).item()
saver.resume("child")
assert x.data_ptr() == pointer and torch.all(x == 7).item()
print("child preload pause/resume passed")
"""
    parent_code = """
import os
import subprocess
import sys
import torch
import torch_mlu
from torch_memory_saver import configure_subprocess
from torch_memory_saver.utils import _is_mlu_torch, is_mlu
assert _is_mlu_torch()
assert not is_mlu()
with configure_subprocess():
    child_env = dict(os.environ)
    if sys.argv[1] == "inherit-all":
        child_env.pop("CN_VISIBLE_DEVICES", None)
    else:
        child_env["CN_VISIBLE_DEVICES"] = sys.argv[1]
    subprocess.run([sys.executable, "-c", sys.argv[2]], env=child_env,
                   check=True, timeout=60)
"""
    env = dict(os.environ, CN_VISIBLE_DEVICES="-1")
    env.pop("LD_PRELOAD", None)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(root), os.environ.get("PYTHONPATH")])
    )
    subprocess.run(
        [sys.executable, "-c", parent_code,
         os.environ.get("CN_VISIBLE_DEVICES", "inherit-all"), child_code],
        env=env, check=True, timeout=90,
    )


def test_mlu_torch_oom_fails_fast(capfd):
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_mlu('''
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
saver.memory_margin_bytes = 1 << 60
try:
    with saver.region("oom"):
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="mlu")
except torch.OutOfMemoryError:
    pass
''', "torch", allocator_config={})
    assert error.value.returncode == 1
    stderr = capfd.readouterr().err
    assert "cudaError error:" in stderr
    assert "func=tms_torch_malloc" in stderr


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_allocator_config(mode):
    configs = [
        ({name: "expandable_segments: True"}, True)
        for name in ("PYTORCH_MLU_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    ] + [
        ({"PYTORCH_MLU_ALLOC_CONF": "expandable_segments:False",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, False),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
          "PYTORCH_ALLOC_CONF": "expandable_segments:True"}, False),
    ]
    for config, enabled in configs:
        code = '''
import os
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
torch.mlu.init()
assert torch.mlu.memory._snapshot()["allocator_settings"]["expandable_segments"] == ENABLED
# The allocator has already read its configuration; later env changes must not
# change the guard's decision.
for name in ("PYTORCH_MLU_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    os.environ.pop(name, None)
if ENABLED:
    try:
        with saver.region():
            raise AssertionError("unsupported allocator was accepted")
    except RuntimeError as error:
        assert "expandable_segments" in str(error)
else:
    with saver.region(enable_cpu_backup=True):
        x = torch.full((64 * 1024 * 1024,), 7, dtype=torch.uint8, device="mlu")
    ptr = x.data_ptr()
    torch.mlu.synchronize()
    before = torch.mlu.mem_get_info()[0]
    saver.pause()
    assert torch.mlu.mem_get_info()[0] - before >= x.nbytes
    saver.resume()
    assert x.data_ptr() == ptr and torch.all(x == 7).item()
'''.replace("ENABLED", repr(enabled))
        _run_mlu(code, mode, allocator_config=config)


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_region_cross_device(mode):
    _run_mlu('''
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
for device in range(2):
    warm = torch.ones(1024, device=f"mlu:{device}")
    torch.mlu.synchronize(device)
    del warm
torch.mlu.set_device(0)
with saver.region("outer", enable_cpu_backup=True):
    assert torch.mlu.current_device() == 0
    a = torch.full((1024,), 7, device="mlu:0")
    with torch.mlu.device(1):
        with saver.region("outer", enable_cpu_backup=True):
            b = torch.full((1024,), 8, device="mlu:1")
            c = torch.full((1024,), 9, device="mlu:1")
        assert torch.mlu.current_device() == 1
    assert saver._impl._affected_devices("outer") == [0, 1]
    assert torch.mlu.current_device() == 0
    with saver.region("inner", enable_cpu_backup=True):
        d = torch.full((1024,), 10, device="mlu:0")
        assert saver._impl._affected_devices("inner") == [0]
        with saver.disable():
            assert not saver._impl._binary_wrapper.cdll.tms_get_interesting_region()
            disabled = torch.full((1024,), 11, device="mlu:0")
        assert saver._impl._binary_wrapper.cdll.tms_get_interesting_region()
    assert saver._impl._binary_wrapper.cdll.tms_get_current_tag() == b"outer"
    e = torch.full((1024,), 12, device="mlu:0")
assert torch.mlu.current_device() == 0
assert not saver._impl._binary_wrapper.cdll.tms_get_interesting_region()
outer = [a, b, c, e]
ptrs = [x.data_ptr() for x in outer]
for _ in range(2):
    saver.pause("outer")
    for x, value in zip(outer, (7, 8, 9, 12)):
        assert torch.all(saver.get_cpu_backup(x) == value).item()
    assert torch.all(d == 10).item()
    assert torch.all(disabled == 11).item()
    saver.resume("outer")
    for x, value, ptr in zip(outer, (7, 8, 9, 12), ptrs):
        assert x.data_ptr() == ptr and torch.all(x == value).item()
    saver.pause("inner")
    assert torch.all(saver.get_cpu_backup(d) == 10).item()
    assert torch.all(a == 7).item()
    saver.resume("inner")
    assert torch.all(d == 10).item()
''', mode, allocator_config={}, device_count=2)


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_thread_unmanaged_allocation(mode):
    _run_mlu('''
import threading
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
torch.mlu.set_device(0)
saver.retain_cpu_backup = False
ready, start, done = [threading.Event() for _ in range(3)]
errors = []
def worker():
    try:
        torch.mlu.set_device(1)
        ready.set()
        assert start.wait(10)
        assert not saver._impl._binary_wrapper.cdll.tms_get_interesting_region()
        x = torch.ones(1024, device="mlu:1")
        assert x.sum().item() == 1024
        torch.mlu.synchronize(1)
        del x
    except BaseException as error:
        errors.append(error)
    finally:
        done.set()
thread = threading.Thread(target=worker, daemon=True)
thread.start()
assert ready.wait(10)
with saver.region("main", enable_cpu_backup=True):
    a = torch.full((1024,), 7, device="mlu:0")
    start.set()
    assert done.wait(20)
thread.join(timeout=10)
assert not thread.is_alive() and not errors, errors
assert saver._impl._affected_devices("main") == [0]
with torch.mlu.device(1):
    with saver.region("main", enable_cpu_backup=True):
        b = torch.full((1024,), 9, device="mlu:1")
assert saver._impl._affected_devices("main") == [0, 1]
ptrs = a.data_ptr(), b.data_ptr()
saver.pause("main")
assert torch.all(saver.get_cpu_backup(a) == 7).item()
assert torch.all(saver.get_cpu_backup(b) == 9).item()
saver.resume("main")
assert (a.data_ptr(), b.data_ptr()) == ptrs
assert torch.all(a == 7).item() and torch.all(b == 9).item()
''', mode, allocator_config={}, device_count=2)


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_thread_regions(mode):
    _run_mlu('''
import threading
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
torch.mlu.set_device(0)
saver.retain_cpu_backup = False
cdll = saver._impl._binary_wrapper.cdll
ready, start, done = [threading.Event() for _ in range(3)]
errors, tensors = [], []
def worker():
    try:
        torch.mlu.set_device(1)
        assert not cdll.tms_get_interesting_region()
        ready.set()
        assert start.wait(10)
        with saver.region("same", enable_cpu_backup=True):
            assert cdll.tms_get_interesting_region()
            assert cdll.tms_get_current_tag() == b"same"
            tensors.append(torch.full((1024,), 13, device="mlu:1"))
        assert not cdll.tms_get_interesting_region()
    except BaseException as error:
        errors.append(error)
    finally:
        done.set()
thread = threading.Thread(target=worker, daemon=True)
thread.start()
assert ready.wait(10)
assert not cdll.tms_get_interesting_region()
with saver.region("same", enable_cpu_backup=True):
    a = torch.full((1024,), 7, device="mlu:0")
    start.set()
    assert done.wait(20)
    assert cdll.tms_get_interesting_region()
    assert cdll.tms_get_current_tag() == b"same"
thread.join(timeout=10)
assert not thread.is_alive() and not errors, errors
assert not cdll.tms_get_interesting_region()
b, = tensors
assert saver._impl._affected_devices("same") == [0, 1]
saver.pause("same")
assert torch.all(saver.get_cpu_backup(a) == 7).item()
assert torch.all(saver.get_cpu_backup(b) == 13).item()
saver.resume("same")
assert torch.all(a == 7).item() and torch.all(b == 13).item()
''', mode, allocator_config={}, device_count=2)


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_runtime_allocator_config(mode):
    _run_mlu('''
import torch
import torch_mlu
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
with saver.region("initial", enable_cpu_backup=True):
    x = torch.full((1024,), 7, device="mlu")
cdll = saver._impl._binary_wrapper.cdll
assert not cdll.tms_get_interesting_region()
tag = cdll.tms_get_current_tag()
torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
try:
    with saver.region("rejected"):
        raise AssertionError("unsupported allocator was accepted")
except RuntimeError as error:
    assert "expandable_segments" in str(error)
assert not cdll.tms_get_interesting_region()
assert cdll.tms_get_current_tag() == tag
assert saver._impl._affected_devices("rejected") == []
if MODE == "preload":
    graph = torch.mlu.MLUGraph()
    try:
        with saver.cuda_graph(graph):
            raise AssertionError("unsupported graph allocator was accepted")
    except RuntimeError as error:
        assert "expandable_segments" in str(error)
    assert not torch.mlu.is_current_stream_capturing()
torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
with saver.region("rejected", enable_cpu_backup=True):
    y = torch.full((1024,), 9, device="mlu")
ptrs = x.data_ptr(), y.data_ptr()
saver.pause()
assert torch.all(saver.get_cpu_backup(x) == 7).item()
assert torch.all(saver.get_cpu_backup(y) == 9).item()
saver.resume()
assert (x.data_ptr(), y.data_ptr()) == ptrs
assert torch.all(x == 7).item() and torch.all(y == 9).item()
''', mode, allocator_config={})


@pytest.mark.parametrize("mode", ["torch", "preload"])
@pytest.mark.parametrize("autoload", [True, False])
def test_mlu_backend_registration(mode, autoload, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1" if autoload else "0")
    code = '''
import torch
from torch_memory_saver import torch_memory_saver as saver
from torch_memory_saver.utils import is_mlu
if not AUTOLOAD:
    assert not hasattr(torch, "mlu")
    assert not is_mlu()
    import torch_mlu
assert is_mlu()
assert torch.get_device_module() is torch.mlu
saver.hook_mode = MODE
with saver.region("registration", enable_cpu_backup=True):
    tensor = torch.full((1024,), 7, device="mlu")
assert saver._impl._device_module is torch.mlu
ptr = tensor.data_ptr()
saver.pause("registration")
assert torch.all(saver.get_cpu_backup(tensor) == 7).item()
saver.resume("registration")
assert tensor.data_ptr() == ptr
assert torch.all(tensor == 7).item()
'''
    _run_mlu(code.replace("AUTOLOAD", repr(autoload)), mode, allocator_config={})


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_cpu_backup_released_on_free(mode):
    _run_mlu('''
import sys
from pathlib import Path
import torch
import torch_memory_saver
sys.path.insert(0, str(Path(torch_memory_saver.__file__).resolve().parents[1] / "test" / "examples"))
from cpu_backup import _assert_cpu_backup_released_on_allocation_free
saver = torch_memory_saver.torch_memory_saver
saver.hook_mode = MODE
for paused in (False, True):
    _assert_cpu_backup_released_on_allocation_free(device=0, paused=paused)
    assert saver.retain_cpu_backup is False
    tag = "retained_free_" + ("paused" if paused else "active") + "_0"
    assert saver._impl._affected_devices(tag) == []
''', mode, allocator_config={})


@pytest.mark.parametrize("mode", ["torch", "preload"])
def test_mlu_noncurrent_device_region(mode):
    _run_mlu('''
import torch
from torch_memory_saver import torch_memory_saver as saver
saver.hook_mode = MODE
size = 32 * 1024 * 1024
torch.mlu.set_device(1)
with saver.region("routing", enable_cpu_backup=True):
    noncurrent = torch.full((size,), 7, dtype=torch.uint8, device="mlu:0")
assert saver._impl._affected_devices("routing") == ([0] if MODE == "preload" else [])
with saver.region("routing", enable_cpu_backup=True):
    current = torch.full((size,), 9, dtype=torch.uint8, device="mlu:1")
assert saver._impl._affected_devices("routing") == ([0, 1] if MODE == "preload" else [1])
with torch.mlu.device(0):
    with saver.region("routing", enable_cpu_backup=True):
        explicit = torch.full((size,), 11, dtype=torch.uint8, device="mlu:0")
assert torch.mlu.current_device() == 1
assert saver._impl._affected_devices("routing") == [0, 1]
pointers = [x.data_ptr() for x in (noncurrent, current, explicit)]
for device in range(2):
    torch.mlu.synchronize(device)
free_before = [torch.mlu.mem_get_info(device)[0] for device in range(2)]
saver.pause("routing")
assert torch.mlu.current_device() == 1
for device in range(2):
    assert torch.mlu.mem_get_info(device)[0] - free_before[device] >= size
assert torch.all(saver.get_cpu_backup(current) == 9).item()
assert torch.all(saver.get_cpu_backup(explicit) == 11).item()
if MODE == "preload":
    assert torch.all(saver.get_cpu_backup(noncurrent) == 7).item()
else:
    assert torch.all(noncurrent == 7).item()
saver.resume("routing")
assert torch.mlu.current_device() == 1
for tensor, value, pointer in zip((noncurrent, current, explicit), (7, 9, 11), pointers):
    assert tensor.data_ptr() == pointer
    assert torch.all(tensor == value).item()
''', mode, allocator_config={}, device_count=2)


@pytest.mark.parametrize("paused", [False, True])
def test_mlu_cpu_backup_free_outside_region(paused):
    code = '''
import gc
import sys
from pathlib import Path
import torch
import torch_memory_saver
sys.path.insert(0, str(Path(torch_memory_saver.__file__).resolve().parents[1] / "test" / "examples"))
from cpu_backup import _mlu_host_pointer_is_allocated
saver = torch_memory_saver.torch_memory_saver
saver.hook_mode = MODE
saver.retain_cpu_backup = True
cdll = saver._impl._binary_wrapper.cdll
assert not cdll.tms_get_interesting_region()
original_tag = cdll.tms_get_current_tag()
with saver.region("outside_free", enable_cpu_backup=True, cpu_backup_backend="pinned"):
    assert cdll.tms_get_interesting_region()
    tensor = torch.full((4096,), 7, dtype=torch.uint8, device="mlu:0")
assert not cdll.tms_get_interesting_region()
assert cdll.tms_get_current_tag() == original_tag
saver.pause("outside_free")
backup = saver.get_cpu_backup(tensor, zero_copy=True)
pointer = backup.data_ptr()
assert torch.all(backup == 7).item()
assert _mlu_host_pointer_is_allocated(pointer)
if not PAUSED:
    saver.resume("outside_free")
    assert torch.all(tensor == 7).item()
    assert saver.get_cpu_backup(tensor, zero_copy=True).data_ptr() == pointer
assert saver._impl._affected_devices("outside_free") == [0]
pool = saver._impl._mem_pools.pop(("outside_free", True, False, "pinned", 0))
torch.mlu.synchronize()
del backup, tensor, pool
gc.collect()
torch.mlu.empty_cache()
torch.mlu.synchronize()
assert not cdll.tms_get_interesting_region()
assert cdll.tms_get_current_tag() == original_tag
assert saver._impl._affected_devices("outside_free") == []
assert not _mlu_host_pointer_is_allocated(pointer)
saver.retain_cpu_backup = False
'''
    _run_mlu(code.replace("PAUSED", repr(paused)), "torch", allocator_config={})
