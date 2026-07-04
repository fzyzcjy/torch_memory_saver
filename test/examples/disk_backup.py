import logging
import os
import sys
import tempfile

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    disk_dir = tempfile.mkdtemp(prefix="tms_disk_backup_")
    print(f"disk_backup_dir={disk_dir}")

    print("Allocate tensor_with_disk_backup")
    with torch_memory_saver.region(enable_disk_backup=True, disk_backup_dir=disk_dir):
        # Larger than the default staging chunk to exercise multi-chunk streaming.
        tensor_with_backup = torch.full((20_000_000,), 10, dtype=torch.uint8, device='cuda')
        typed_tensor_with_backup = torch.randn((10, 20, 30), dtype=torch.float32, device='cuda')
        typed_tensor_with_backup_expected = typed_tensor_with_backup.clone()

    print("Allocate tensor_without_backup")
    with torch_memory_saver.region(enable_disk_backup=False):
        tensor_without_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device='cuda')

    assert tensor_with_backup[:3].tolist() == [10, 10, 10]
    assert tensor_without_backup[:3].tolist() == [20, 20, 20]

    # Multiple pause/resume cycles: bytes must be restored bit-identically and
    # the on-disk footprint must stay constant (files are overwritten in place).
    disk_bytes = set()
    for _ in range(3):
        torch_memory_saver.pause()
        disk_bytes.add(sum(os.path.getsize(os.path.join(disk_dir, f)) for f in os.listdir(disk_dir)))

        # occupy the freed GPU space to prove it was actually reclaimed
        tensor_unrelated = torch.full((20_000_000,), 30, dtype=torch.uint8, device='cuda')
        del tensor_unrelated
        torch.cuda.empty_cache()

        torch_memory_saver.resume()

        assert tensor_with_backup[:3].tolist() == [10, 10, 10]
        assert tensor_without_backup[:3].tolist() != [20, 20, 20]
        assert torch.equal(typed_tensor_with_backup, typed_tensor_with_backup_expected)

    assert len(disk_bytes) == 1, f"disk usage grew across cycles: {sorted(disk_bytes)}"
    print(f"OK: bit-identical across cycles, disk constant at {next(iter(disk_bytes))} bytes")


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
