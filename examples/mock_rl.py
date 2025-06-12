import logging
import os
import sys
import time
from typing import Callable

import torch
import torch_memory_saver
from examples.util import print_gpu_memory_gb

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Use the global singleton instance
memory_saver = torch_memory_saver.memory_saver
dummy_tensor_size = (5, 100_000_000,)


def _ptr(x):
    assert isinstance(x, torch.Tensor)
    return hex(x.data_ptr())

class Model:
    def __init__(self, input_size=20_480, output_size=20_480):
        self.input_size = input_size
        self.output_size = output_size
        self.create_weights()
    
    def create_weights(self):
        with memory_saver.region(tag="model_weights"):
            # Single linear layer (no bias)
            self.linear = torch.nn.Linear(self.input_size, self.output_size, bias=False, device='cuda')
            # Initialize with ones for predictable output
            torch.nn.init.ones_(self.linear.weight)
        
        print(f'create_weights {_ptr(self.linear.weight)=}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single linear transformation
        return self.linear(x).mean()  # Return scalar instead of tensor
    
    def clear_weights(self):
        del self.linear

class KVCache:
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        with memory_saver.region(tag="kv_cache"):
            # or model weights, etc
            self.kv_buffer = torch.full(dummy_tensor_size, value, dtype=torch.float32, device='cuda')
        print(f'create_buffers {_ptr(self.kv_buffer)=}')

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg: torch.Tensor) -> torch.Tensor:
        # print(f'KVCache.execute {arg=} {self.kv_buffer=}')
        return (arg + self.kv_buffer.mean(dim=1)).mean()


# https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
def create_cuda_graph(fn: Callable):
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        print('with torch.cuda.stream(s) execute fn')
        fn()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        print('with torch.cuda.graph(g) execute fn')
        fn()

    return g


def run():
    # Check if TorchMemorySaver is properly enabled  
    print(f"TorchMemorySaver enabled: {memory_saver.enabled}")
    print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'NOT SET')}")
    
    if not memory_saver.enabled:
        print("WARNING: TorchMemorySaver is not enabled! Memory pause/resume won't work.")
        print("Make sure to set LD_PRELOAD properly.")

    cache = KVCache()
    model = Model()
    static_input = torch.zeros((20_480,), dtype=torch.float32, device='cuda')
    static_output = torch.zeros((), dtype=torch.float32, device='cuda')
    print(f'{_ptr(static_input)=} {_ptr(static_output)=}')

    def fn():
        nonlocal static_output
        kv_output = cache.execute(static_input[:5])
        model_output = model.forward(static_input)
        static_output = kv_output + model_output

    g = create_cuda_graph(fn)

    print('replay #1')
    static_input[...] = 100
    g.replay()
    print(f'{static_output=}')
    # KV: 101, Model: mean(100 * 1 * 20480) = 2,048,000
    assert static_output == 2048101, f'{static_output=}'

    # cache.clear_buffers()

    # with with_pauseable_mode():
    #     big_tensor = torch.zeros((2_000_000_000,), dtype=torch.uint8, device='cuda')
    #     print(f'{big_tensor=}')

    print('torch.cuda.empty_cache()')
    torch.cuda.empty_cache()

    print('sleep...')
    time.sleep(3)

    print('Before pause kv_cache')
    print_gpu_memory_gb("Before pause kv_cache")

    memory_saver.pause("kv_cache")
    print('After pause kv_cache')
    print_gpu_memory_gb("After pause kv_cache")

    memory_saver.pause("model_weights")
    print('After pause model_weights')
    print_gpu_memory_gb("After pause model_weights")

    print('sleep...')
    time.sleep(3)

    print('when kv cache and model weights are released, we can allocate *other* big tensors')
    other_big_tensor = torch.zeros((2500_000_000,), dtype=torch.uint8, device='cuda')
    print('sleep...')
    time.sleep(3)
    print(f'{other_big_tensor=}')
    del other_big_tensor
    torch.cuda.empty_cache()
    print('sleep...')
    time.sleep(3)

    # this should fail
    # print(f'{cache.kv_buffer=}')

    print('Before resume model_weights and kv_cache')
    print_gpu_memory_gb("Before resume")
    
    memory_saver.resume("model_weights")
    print('After resume model_weights')
    print_gpu_memory_gb("After resume model_weights")
    
    memory_saver.resume("kv_cache")
    print('After resume kv_cache') 
    print_gpu_memory_gb("After resume kv_cache")

    dummy = torch.zeros((3,), device='cuda')
    print(f'{_ptr(dummy)=}')

    # cache.create_buffers(2)

    cache.kv_buffer[...] = 2
    with torch.no_grad():
        model.linear.weight[...] = 2

    print('replay #2')
    static_input[...] = 200
    g.replay()
    print(f'{static_output=}')
    # KV: 202, Model: mean(200 * 2 * 20480) = 8,192,000  
    assert static_output == 8192202, f'{static_output=}'

    print('sleep...')
    time.sleep(3)

    # print(f'{big_tensor=}')
    print(f'{dummy=}')

    print("Succeed!")
    print("=" * 100)
    
    # Additional test: demonstrate selective pause/resume
    print("\n=== Additional test: selective pause/resume ===")
    print("Pause only kv_cache, keep model_weights active")
    memory_saver.pause("kv_cache")
    print_gpu_memory_gb("Only kv_cache paused")
    
    print("Try to access model weights (should work)")
    try:
        _ = model.linear.weight[0, 0]
        print("Model weights access successful")
    except:
        print("Model weights access failed")
    
    print("Resume kv_cache")
    memory_saver.resume("kv_cache")
    print_gpu_memory_gb("All resumed")

    # exit this process gracefully, bypassing CUDA cleanup
    # Checkout for more details: https://github.com/fzyzcjy/torch_memory_saver/pull/18 
    os._exit(0)


if __name__ == '__main__':
    run()
