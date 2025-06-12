import logging
import sys
import time
import os

import torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

import torch_memory_saver
from examples.util import print_gpu_memory_gb

# Use the global singleton instance
memory_saver = torch_memory_saver.memory_saver

# Check if TorchMemorySaver is properly enabled
print(f"TorchMemorySaver enabled: {memory_saver.enabled}")
print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'NOT SET')}")

if not memory_saver.enabled:
    print("WARNING: TorchMemorySaver is not enabled! Memory pause/resume won't work.")
    print("Make sure to set LD_PRELOAD properly.")

normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Get the virtual address
original_address = pauseable_tensor.data_ptr()
print(f"Pauseable tensor virtual address: 0x{original_address:x}")

print_gpu_memory_gb("Before pause")

print(f'{normal_tensor=} {pauseable_tensor=}')

print('sleep...')
time.sleep(3)

memory_saver.pause()  # Pause all tensors
print_gpu_memory_gb("After pause")

print('sleep...')
time.sleep(3)

memory_saver.resume()  # Resume all tensors
print_gpu_memory_gb("After resume")

new_address = pauseable_tensor.data_ptr()
print(f"Pauseable tensor virtual address: 0x{new_address:x}")

assert original_address == new_address, 'Tensor virtual address should be the same'

print('sleep...')
time.sleep(3)

print(f'{normal_tensor=} {pauseable_tensor=}')
