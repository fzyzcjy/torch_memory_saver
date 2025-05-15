import logging
import sys
import time

import torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from torch_memory_saver import TorchMemorySaver

memory_saver = TorchMemorySaver()

normal_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

print(f'After Created: {normal_tensor=}\n{pauseable_tensor=}')
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


print('sleep...')
time.sleep(3)

memory_saver.pause()

print(f'After Paused: {normal_tensor=}\n{pauseable_tensor=}')
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

print('sleep...')
time.sleep(3)

memory_saver.resume()

print('sleep...')
time.sleep(3)

print(f'After Resumed: {normal_tensor=}\n{pauseable_tensor=}')
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

