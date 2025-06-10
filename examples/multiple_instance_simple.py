import logging
import sys
import time
import os
import torch
from torch_memory_saver import TorchMemorySaver
from examples.util import print_gpu_memory_gb

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Create two TorchMemorySaver instances
memory_saver1 = TorchMemorySaver()
memory_saver2 = TorchMemorySaver()

# Allocate a normal tensor
normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

# Allocate tensors using different savers
with memory_saver1.region():
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with memory_saver2.region():
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Get virtual addresses
address1 = tensor1.data_ptr()
address2 = tensor2.data_ptr()
print(f"Pauseable tensor1 virtual address: 0x{address1:x}")
print(f"Pauseable tensor2 virtual address: 0x{address2:x}")

print_gpu_memory_gb()
print('sleep...')
time.sleep(3)

print('pause memory saver 1')
memory_saver1.pause()
print_gpu_memory_gb()

print('pause memory saver 2')
memory_saver2.pause()
print_gpu_memory_gb()

print('sleep...')
time.sleep(3)

print('resume memory saver 1')
memory_saver1.resume()
print_gpu_memory_gb()

print('resume memory saver 2')
memory_saver2.resume()
print_gpu_memory_gb()

# Verify addresses
new_address1 = tensor1.data_ptr()
new_address2 = tensor2.data_ptr()
print(f"Pauseable tensor1 virtual address: 0x{new_address1:x}")
print(f"Pauseable tensor2 virtual address: 0x{new_address2:x}")

assert address1 == new_address1, 'Pauseable tensor1 virtual address should be the same'
assert address2 == new_address2, 'Pauseable tensor2 virtual address should be the same'
print('Both pauseable tensors are still with the same virtual address')

print('sleep...')
time.sleep(3)

print(f'{normal_tensor=} {tensor1=} {tensor2=}')