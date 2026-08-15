import ctypes
import gc
import logging
import os
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import empty_cache, get_device


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    assert torch_memory_saver.retain_cpu_backup is False
    device = get_device()

    # Pause/resume keeps backup contents; without backup, contents are lost.
    # mmap is CUDA-only. ROCm default is pinned; XPU does not take a backend.
    if device == "cuda" and not torch.version.hip:
        backends = ["mmap", "pinned"]
    else:
        backends = [None]
    for backend in backends:
        print(f"Allocate tensor_with_backup backend={backend or 'default'}")
        region_kwargs = dict(enable_cpu_backup=True)
        if backend is not None:
            region_kwargs["cpu_backup_backend"] = backend
        with torch_memory_saver.region(**region_kwargs):
            tensor_with_backup = torch.full((20_000_000,), 10, dtype=torch.uint8, device=device)
            typed_tensor_with_backup = torch.randn((10, 20, 30), dtype=torch.float32, device=device)
            typed_tensor_with_backup_cpu_expected = typed_tensor_with_backup.cpu()

        print("Allocate tensor_without_backup")
        with torch_memory_saver.region(enable_cpu_backup=False):
            tensor_without_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device=device)

        assert tensor_with_backup[:3].tolist() == [10, 10, 10]
        assert tensor_without_backup[:3].tolist() == [20, 20, 20]

        torch_memory_saver.pause()
        typed_actual = torch_memory_saver.get_cpu_backup(typed_tensor_with_backup)
        assert typed_actual is not None
        assert torch.all(typed_tensor_with_backup_cpu_expected == typed_actual)

        # occupy some space
        tensor_unrelated = torch.full((20_000_000,), 30, dtype=torch.uint8, device=device)
        torch_memory_saver.resume()

        assert tensor_with_backup[:3].tolist() == [10, 10, 10]
        assert tensor_without_backup[:3].tolist() != [20, 20, 20]
        # ROCm retains host backup across resume; CUDA/XPU release it.
        if not torch.version.hip:
            assert torch_memory_saver.get_cpu_backup(typed_tensor_with_backup) is None

        del tensor_with_backup, typed_tensor_with_backup, tensor_without_backup, tensor_unrelated
        empty_cache()

    if device == "cuda" and not torch.version.hip:
        # Pageable H2D must finish before resume returns (non-default stream).
        n = 64 * 1024 * 1024
        with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
            restored = torch.full((n,), 7, dtype=torch.uint8, device=device)
        torch_memory_saver.pause()
        torch_memory_saver.resume()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            out = restored.clone()
        stream.synchronize()
        assert int(out[0].item()) == 7
        assert torch.all(out == 7)
        del restored, out
        empty_cache()

        # mmap + retain: same host pointer across resume; released after non-retain resume.
        expected = torch.arange(4096, dtype=torch.int64).remainder(251).to(torch.uint8)
        with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
            retained = expected.to(device)
        torch_memory_saver.pause()
        first = torch_memory_saver.get_cpu_backup(retained, zero_copy=True)
        first_ptr = first.data_ptr()
        assert torch.equal(first, expected)
        torch_memory_saver.retain_cpu_backup = True
        del first
        torch_memory_saver.resume()
        after = torch_memory_saver.get_cpu_backup(retained, zero_copy=True)
        assert after.data_ptr() == first_ptr
        assert torch.equal(after, expected)
        assert torch.equal(retained.cpu(), expected)
        del after
        torch_memory_saver.retain_cpu_backup = False
        torch_memory_saver.pause()
        torch_memory_saver.resume()
        assert torch_memory_saver.get_cpu_backup(retained) is None
        assert torch.equal(retained.cpu(), expected)
        del retained
        empty_cache()

    # Tags remain independent, and retained host storage is reused at the same
    # address while its bytes are refreshed. Exercise every visible device.
    if device == "cuda" and not torch.version.hip:
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                selected_expected = _make_pattern(size=1024, offset=40 + device_id)
                other_expected = _make_pattern(size=1024, offset=80 + device_id)

                with torch_memory_saver.region(tag=f"retained_{device_id}", enable_cpu_backup=True):
                    selected = selected_expected.to(device_id)
                with torch_memory_saver.region(tag=f"other_{device_id}", enable_cpu_backup=True):
                    other = other_expected.to(device_id)

                torch_memory_saver.pause(tag=f"retained_{device_id}")
                first_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
                first_pointer = first_backup.data_ptr()
                assert torch.equal(first_backup, selected_expected)
                assert torch.equal(other.cpu(), other_expected)

                torch_memory_saver.retain_cpu_backup = True
                assert torch_memory_saver.retain_cpu_backup is True
                del first_backup
                torch_memory_saver.resume(tag=f"retained_{device_id}")
                retained_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
                assert retained_backup.data_ptr() == first_pointer
                assert torch.equal(retained_backup, selected_expected)
                assert torch.equal(selected.cpu(), selected_expected)
                del retained_backup

                torch_memory_saver.retain_cpu_backup = False
                assert torch_memory_saver.retain_cpu_backup is False
                retained_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
                assert retained_backup.data_ptr() == first_pointer
                assert torch.equal(retained_backup, selected_expected)
                del retained_backup

                selected_expected = _make_pattern(size=1024, offset=50 + device_id)
                selected.copy_(selected_expected)
                torch_memory_saver.pause(tag=f"retained_{device_id}")
                second_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
                assert second_backup.data_ptr() == first_pointer
                assert torch.equal(second_backup, selected_expected)

                torch_memory_saver.retain_cpu_backup = True
                del second_backup
                torch_memory_saver.resume(tag=f"retained_{device_id}")
                retained_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
                assert retained_backup.data_ptr() == first_pointer
                assert torch.equal(retained_backup, selected_expected)
                assert torch.equal(selected.cpu(), selected_expected)
                del retained_backup

                selected_expected = _make_pattern(size=1024, offset=60 + device_id)
                selected.copy_(selected_expected)
                torch_memory_saver.pause(tag=f"retained_{device_id}")
                third_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
                assert third_backup.data_ptr() == first_pointer
                assert torch.equal(third_backup, selected_expected)

                torch_memory_saver.retain_cpu_backup = False
                del third_backup
                torch_memory_saver.resume(tag=f"retained_{device_id}")
                assert torch.equal(selected.cpu(), selected_expected)
                assert torch_memory_saver.get_cpu_backup(selected) is None

                _assert_cpu_backup_released_on_allocation_free(device=device_id, paused=False)
                _assert_cpu_backup_released_on_allocation_free(device=device_id, paused=True)


def run_retain_from_env(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    assert torch_memory_saver.retain_cpu_backup is True


def run_multi_device_mmap_restore(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    tensor_size = 64 * 1024 * 1024
    tensors: list[torch.Tensor] = []

    for device_id, value in enumerate((7, 13)):
        torch.cuda.set_device(device_id)
        with torch_memory_saver.region(
            tag=f"mmap_restore_{device_id}",
            enable_cpu_backup=True,
            cpu_backup_backend="mmap",
        ):
            tensors.append(
                torch.full(
                    (tensor_size,),
                    value,
                    dtype=torch.uint8,
                    device=f"cuda:{device_id}",
                )
            )

    for device_id in range(2):
        torch.cuda.set_device(device_id)
        torch_memory_saver.pause(tag=f"mmap_restore_{device_id}")

    torch.cuda.set_device(0)
    torch_memory_saver.resume()
    assert torch.cuda.current_device() == 0

    restored: list[torch.Tensor] = []
    for device_id, (tensor, value) in enumerate(zip(tensors, (7, 13), strict=True)):
        stream = torch.cuda.Stream(device=device_id)
        with torch.cuda.stream(stream):
            restored.append(tensor.clone())
        stream.synchronize()
        assert bool(torch.all(restored[-1] == value).item())

    del tensors, restored
    for device_id in range(2):
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()


def run_backend_from_env(hook_mode: str):
    # TMS_INIT_CPU_BACKUP_BACKEND is set by the test harness (not here).
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    assert os.environ.get("TMS_INIT_CPU_BACKUP_BACKEND") == "mmap"
    device = get_device()
    torch_memory_saver._ensure_initialized()
    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    assert cdll.tms_get_cpu_backup_backend() == b"mmap"

    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor = torch.full((20_000_000,), 10, dtype=torch.uint8, device=device)
    assert cdll.tms_get_cpu_backup_backend() == b"mmap"

    torch_memory_saver.pause()
    assert torch_memory_saver.get_cpu_backup(tensor) is not None
    torch_memory_saver.resume()
    assert tensor[:3].tolist() == [10, 10, 10]
    assert torch_memory_saver.get_cpu_backup(tensor) is None

    del tensor
    empty_cache()


def run_preload_backend_from_env(hook_mode: str):
    # Env is set by the test harness. Allocate outside region() so C++ TLS
    # construction-time parse is what selects mmap.
    assert hook_mode == "preload"
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert os.environ["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert os.environ["TMS_INIT_CPU_BACKUP_BACKEND"] == "mmap"

    tensor_bytes = 64 * 1024 * 1024
    tolerance_bytes = int(0.25 * tensor_bytes)

    def rss_bytes() -> int:
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmRSS:"))

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    torch_memory_saver._ensure_initialized()
    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    assert cdll.tms_get_interesting_region()
    assert cdll.tms_get_enable_cpu_backup()
    assert cdll.tms_get_cpu_backup_backend() == b"mmap"

    torch.cuda.synchronize()
    tensor = torch.full((tensor_bytes,), 10, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    rss_after_alloc = rss_bytes()

    torch_memory_saver.pause()
    torch.cuda.synchronize()
    rss_after_pause = rss_bytes()
    assert torch_memory_saver.get_cpu_backup(tensor) is not None

    torch_memory_saver.resume()
    torch.cuda.synchronize()
    rss_after_resume = rss_bytes()
    assert tensor[:3].tolist() == [10, 10, 10]
    assert torch_memory_saver.get_cpu_backup(tensor) is None

    pause_delta = rss_after_pause - rss_after_alloc
    resume_delta = rss_after_resume - rss_after_pause
    assert pause_delta >= tensor_bytes - tolerance_bytes, pause_delta
    assert resume_delta <= -(tensor_bytes - tolerance_bytes), resume_delta

    del tensor
    empty_cache()


def run_rss(hook_mode: str):
    # mmap pause grows process RSS by ~tensor size; non-retaining resume reclaims it.
    tensor_bytes = 2 * 1024**3
    tolerance_bytes = int(0.25 * 1024**3)

    def rss_bytes() -> int:
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmRSS:"))

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    torch.cuda.synchronize()
    with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
        tensor = torch.full((tensor_bytes,), 7, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    rss_after_alloc = rss_bytes()

    torch_memory_saver.pause()
    torch.cuda.synchronize()
    rss_after_pause = rss_bytes()

    torch_memory_saver.resume()
    torch.cuda.synchronize()
    rss_after_resume = rss_bytes()

    assert int(tensor[0].item()) == 7
    pause_delta = rss_after_pause - rss_after_alloc
    resume_delta = rss_after_resume - rss_after_pause
    assert pause_delta >= tensor_bytes - tolerance_bytes, pause_delta
    assert resume_delta <= -(tensor_bytes - tolerance_bytes), resume_delta

    del tensor
    torch.cuda.empty_cache()


def _assert_cpu_backup_released_on_allocation_free(device: int, paused: bool) -> None:
    state = "paused" if paused else "active"
    tag = f"retained_free_{state}_{device}"
    expected = _make_pattern(size=4096, offset=100 + device + int(paused))

    with torch_memory_saver.region(tag=tag, enable_cpu_backup=True):
        tensor = expected.to(device)

    torch_memory_saver.retain_cpu_backup = True
    torch_memory_saver.pause(tag=tag)
    backup = torch_memory_saver.get_cpu_backup(tensor, zero_copy=True)
    backup_pointer = backup.data_ptr()
    assert torch.equal(backup, expected)
    assert _cuda_host_pointer_is_allocated(backup_pointer)

    if not paused:
        del backup
        torch_memory_saver.resume(tag=tag)
        backup = torch_memory_saver.get_cpu_backup(tensor, zero_copy=True)
        assert backup.data_ptr() == backup_pointer
        assert torch.equal(backup, expected)
        assert torch.equal(tensor.cpu(), expected)

    # MemPool key includes resolved cpu_backup_backend (default pinned).
    pool_key = (tag, True, False, "pinned", device)
    pool = torch_memory_saver._impl._mem_pools.pop(pool_key)
    torch.cuda.synchronize(device)
    del backup

    with torch_memory_saver._impl._with_region_config(
        tag=tag,
        enable_cpu_backup=True,
        cpu_backup_backend="pinned",
    ):
        del tensor
        del pool
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    assert not _cuda_host_pointer_is_allocated(backup_pointer)
    torch_memory_saver.retain_cpu_backup = False


def _cuda_host_pointer_is_allocated(pointer: int) -> bool:
    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    cuda_host_get_flags = cdll.cudaHostGetFlags
    cuda_host_get_flags.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    cuda_host_get_flags.restype = ctypes.c_int
    cuda_get_last_error = cdll.cudaGetLastError
    cuda_get_last_error.argtypes = []
    cuda_get_last_error.restype = ctypes.c_int

    flags = ctypes.c_uint()
    status = cuda_host_get_flags(ctypes.byref(flags), ctypes.c_void_p(pointer))
    if status != 0:
        cuda_get_last_error()
    return status == 0


def _make_pattern(size: int, offset: int) -> torch.Tensor:
    return torch.arange(size, dtype=torch.int64).add(offset).remainder(251).to(torch.uint8)


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
