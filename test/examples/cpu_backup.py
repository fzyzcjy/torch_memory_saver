import ctypes
import gc
import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_device


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    assert torch_memory_saver.retain_cpu_backup is False

    device = get_device()

    print("Allocate tensor_with_backup")
    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor_with_backup = torch.full((20_000_000,), 10, dtype=torch.uint8, device=device)
        typed_tensor_with_backup = torch.randn((10, 20, 30), dtype=torch.float32, device=device)
        typed_tensor_with_backup_cpu_expected = typed_tensor_with_backup.cpu()

    print("Allocate tensor_without_backup")
    with torch_memory_saver.region(enable_cpu_backup=False):
        tensor_without_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device=device)

    print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() == [20, 20, 20]

    torch_memory_saver.pause()

    typed_tensor_with_backup_cpu_actual = torch_memory_saver.get_cpu_backup(typed_tensor_with_backup)
    assert torch.all(typed_tensor_with_backup_cpu_expected == typed_tensor_with_backup_cpu_actual)

    # occupy some space
    tensor_unrelated = torch.full((20_000_000,), 30, dtype=torch.uint8, device=device)

    torch_memory_saver.resume()

    print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() != [20, 20, 20]
    assert torch_memory_saver.get_cpu_backup(tensor_with_backup) is None

    # Tags remain independent, and retained host storage is reused at the same
    # address while its bytes are refreshed. Exercise every visible device.
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            selected_expected = _make_pattern(size=1024, offset=40 + device)
            other_expected = _make_pattern(size=1024, offset=80 + device)

            with torch_memory_saver.region(tag=f"retained_{device}", enable_cpu_backup=True):
                selected = selected_expected.to(device)
            with torch_memory_saver.region(tag=f"other_{device}", enable_cpu_backup=True):
                other = other_expected.to(device)

            torch_memory_saver.pause(tag=f"retained_{device}")
            first_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
            first_pointer = first_backup.data_ptr()
            assert torch.equal(first_backup, selected_expected)
            assert torch.equal(other.cpu(), other_expected)

            torch_memory_saver.retain_cpu_backup = True
            assert torch_memory_saver.retain_cpu_backup is True
            del first_backup
            torch_memory_saver.resume(tag=f"retained_{device}")
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

            selected_expected = _make_pattern(size=1024, offset=50 + device)
            selected.copy_(selected_expected)
            torch_memory_saver.pause(tag=f"retained_{device}")
            second_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
            assert second_backup.data_ptr() == first_pointer
            assert torch.equal(second_backup, selected_expected)

            torch_memory_saver.retain_cpu_backup = True
            del second_backup
            torch_memory_saver.resume(tag=f"retained_{device}")
            retained_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
            assert retained_backup.data_ptr() == first_pointer
            assert torch.equal(retained_backup, selected_expected)
            assert torch.equal(selected.cpu(), selected_expected)
            del retained_backup

            selected_expected = _make_pattern(size=1024, offset=60 + device)
            selected.copy_(selected_expected)
            torch_memory_saver.pause(tag=f"retained_{device}")
            third_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
            assert third_backup.data_ptr() == first_pointer
            assert torch.equal(third_backup, selected_expected)

            torch_memory_saver.retain_cpu_backup = False
            del third_backup
            torch_memory_saver.resume(tag=f"retained_{device}")
            assert torch.equal(selected.cpu(), selected_expected)
            assert torch_memory_saver.get_cpu_backup(selected) is None

            _assert_cpu_backup_released_on_allocation_free(device=device, paused=False)
            _assert_cpu_backup_released_on_allocation_free(device=device, paused=True)


def run_retain_from_env(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    assert torch_memory_saver.retain_cpu_backup is True


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

    pool_key = (tag, True, False, device)
    pool = torch_memory_saver._impl._mem_pools.pop(pool_key)
    torch.cuda.synchronize(device)
    del backup

    with torch_memory_saver._impl._with_region_config(
        tag=tag,
        enable_cpu_backup=True,
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


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
