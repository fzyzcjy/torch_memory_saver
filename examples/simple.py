import logging
import sys
import time

import torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from torch_memory_saver import TorchMemorySaver

memory_saver = TorchMemorySaver()

normal_tensor = torch.full((4_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with memory_saver.region():
    pauseable_tensor = torch.full((4_000_000_000,), 100, dtype=torch.uint8, device='cuda')

print(f'{normal_tensor=} {pauseable_tensor=}')

print('before sleep...')
time.sleep(10)

memory_saver.pause()
print('after sleep...')
time.sleep(10)

memory_saver.resume()
print('resume from sleep...')
time.sleep(10)

print(f'{normal_tensor=} {pauseable_tensor=}')
