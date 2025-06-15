import logging
import sys
import time
import os
import torch
from torch_memory_saver import torch_memory_saver
from examples.util import print_gpu_memory_gb

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Allocate a normal tensor
normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

# Allocate tensors using different tags
with torch_memory_saver.region(tag="type1"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="type2"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Get virtual addresses
address1 = tensor1.data_ptr()
address2 = tensor2.data_ptr()
print(f"Pauseable tensor1 (tag:type1) virtual address: 0x{address1:x}")
print(f"Pauseable tensor2 (tag:type2) virtual address: 0x{address2:x}")

print_gpu_memory_gb()
print('sleep...')
time.sleep(3)

print('pause memory with tag "type1"')
torch_memory_saver.pause("type1")
print_gpu_memory_gb()

print('pause memory with tag "type2"')
torch_memory_saver.pause("type2")
print_gpu_memory_gb()

print('sleep...')
time.sleep(3)

print('resume memory with tag "type1"')
torch_memory_saver.resume("type1")
print_gpu_memory_gb()

print('resume memory with tag "type2"')
torch_memory_saver.resume("type2")
print_gpu_memory_gb()

# Verify addresses
new_address1 = tensor1.data_ptr()
new_address2 = tensor2.data_ptr()
print(f"Pauseable tensor1 (tag:type1) virtual address: 0x{new_address1:x}")
print(f"Pauseable tensor2 (tag:type2) virtual address: 0x{new_address2:x}")

assert address1 == new_address1, 'Pauseable tensor1 virtual address should be the same'
assert address2 == new_address2, 'Pauseable tensor2 virtual address should be the same'
print('Both pauseable tensors are still with the same virtual address')

print('sleep...')
time.sleep(3)

print(f'{normal_tensor=} {tensor1=} {tensor2=}')

# Additional test: pause/resume specific tags while keeping others active
print('\n=== Additional test: selective pause/resume ===')
print('Pause only type1, keep type2 active')
torch_memory_saver.pause("type1")
print('Try to access tensor1 (should work due to virtual memory)')
try:
    _ = tensor1[0]  # This should still work due to virtual memory management
    print('tensor1 access successful')
except:
    print('tensor1 access failed')

print('Try to access tensor2 (should work)')
try:
    _ = tensor2[0]
    print('tensor2 access successful')
except:
    print('tensor2 access failed')

print('Resume type1')
torch_memory_saver.resume("type1")
print('Both tensors should now be active')

# exit this process gracefully, bypassing CUDA cleanup
# Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18 
os._exit(0)
