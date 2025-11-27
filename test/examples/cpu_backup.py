import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

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


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
