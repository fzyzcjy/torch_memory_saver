import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    assert torch_memory_saver.retain_cpu_backup is False

    print("Allocate tensor_with_backup")
    with torch_memory_saver.region(enable_cpu_backup=True):
        tensor_with_backup = torch.full((20_000_000,), 10, dtype=torch.uint8, device='cuda')
        typed_tensor_with_backup = torch.randn((10, 20, 30), dtype=torch.float32, device='cuda')
        typed_tensor_with_backup_cpu_expected = typed_tensor_with_backup.cpu()

    print("Allocate tensor_without_backup")
    with torch_memory_saver.region(enable_cpu_backup=False):
        tensor_without_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device='cuda')

    print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() == [20, 20, 20]

    torch_memory_saver.pause()

    typed_tensor_with_backup_cpu_actual = torch_memory_saver.get_cpu_backup(typed_tensor_with_backup)
    assert torch.all(typed_tensor_with_backup_cpu_expected == typed_tensor_with_backup_cpu_actual)

    # occupy some space
    tensor_unrelated = torch.full((20_000_000,), 30, dtype=torch.uint8, device='cuda')

    torch_memory_saver.resume()

    print(f"{tensor_with_backup[:3]=} {tensor_without_backup[:3]=}")
    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() != [20, 20, 20]

    # Retention keeps the pinned allocation across resume. A later pause must
    # refresh its contents rather than restoring the previous snapshot.
    torch_memory_saver.retain_cpu_backup = True
    tensor_with_backup.fill_(11)
    torch_memory_saver.pause()
    retained = torch_memory_saver.get_cpu_backup(tensor_with_backup)
    assert retained[:3].tolist() == [11, 11, 11]
    torch_memory_saver.resume()
    assert tensor_with_backup[:3].tolist() == [11, 11, 11]

    tensor_with_backup.fill_(12)
    torch_memory_saver.pause()
    retained = torch_memory_saver.get_cpu_backup(tensor_with_backup)
    assert retained[:3].tolist() == [12, 12, 12]
    torch_memory_saver.resume()
    assert tensor_with_backup[:3].tolist() == [12, 12, 12]
    torch_memory_saver.retain_cpu_backup = False

    # Tags remain independent, and retained host storage is reused at the same
    # address while its bytes are refreshed. Exercise every visible device.
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            with torch_memory_saver.region(tag=f"retained_{device}", enable_cpu_backup=True):
                selected = torch.full((1024,), 40 + device, dtype=torch.uint8, device=device)
            with torch_memory_saver.region(tag=f"other_{device}", enable_cpu_backup=True):
                other = torch.full((1024,), 80 + device, dtype=torch.uint8, device=device)

            torch_memory_saver.retain_cpu_backup = True
            torch_memory_saver.pause(tag=f"retained_{device}")
            first_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
            first_pointer = first_backup.data_ptr()
            assert other[0].item() == 80 + device
            torch_memory_saver.resume(tag=f"retained_{device}")

            selected.fill_(50 + device)
            torch_memory_saver.pause(tag=f"retained_{device}")
            second_backup = torch_memory_saver.get_cpu_backup(selected, zero_copy=True)
            assert second_backup.data_ptr() == first_pointer
            assert second_backup[0].item() == 50 + device
            torch_memory_saver.resume(tag=f"retained_{device}")
            assert selected[0].item() == 50 + device
            torch_memory_saver.retain_cpu_backup = False


def run_retain_from_env(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    assert torch_memory_saver.retain_cpu_backup is True


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
