import logging
import os
import sys
from typing import Callable

import torch
import time
from torch_memory_saver import torch_memory_saver
from util import print_gpu_memory_gb

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Define the size of a large tensor for simulating KV cache
# Size: 5 * 100,000,000 * 4 bytes = 2GB
dummy_tensor_size = (5, 100_000_000,)


def _ptr(x):
    """
    Get the virtual address of a tensor.
    
    Virtual Address is the address that the process sees, not the actual physical memory address.
    It needs to be mapped to actual GPU physical memory through the CUDA driver.
    In torch_memory_saver, this virtual address remains unchanged during pause/resume operations.
    """
    assert isinstance(x, torch.Tensor)
    return hex(x.data_ptr())


class Model:
    """
    Simulate a neural network model for validating memory management of model weights.
    
    Validation objectives:
    1. Whether model weights can maintain functional integrity during pause/resume operations.
    2. Virtual address remains unchanged, but physical memory is reallocated.
    3. Weights need to be reinitialized because physical memory content is lost.
    """
    def __init__(self, input_size=20_480, output_size=20_480):
        self.input_size = input_size
        self.output_size = output_size
        self.create_weights()
    
    def create_weights(self):
        """
        Create model weights in the region managed by torch_memory_saver.
        
        Use torch_memory_saver.region() to mark this memory allocation,
        enabling subsequent pause/resume operations through tags.
        """
        with torch_memory_saver.region(tag="model_weights"):
            # Create a large linear layer weight matrix.
            # Size: 20480 * 20480 * 4 bytes ≈ 1.6GB
            self.linear = torch.nn.Linear(self.input_size, self.output_size, bias=False, device='cuda')
            torch.nn.init.ones_(self.linear.weight)
        
        print(f'Model weights created: {_ptr(self.linear.weight)}')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).mean()
    
    def clear_weights(self):
        del self.linear


class KVCache:
    """
    Simulate KV cache for validating memory management of cache data.
    
    Verify that virtual address remains unchanged while physical memory is reallocated.
    """
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        """
        Create KV cache in the region managed by torch_memory_saver.
        
        Use torch_memory_saver.region() to mark this memory allocation,
        enabling subsequent pause/resume operations through tags.
        """
        with torch_memory_saver.region(tag="kv_cache"):
            # Create a large KV cache tensor.
            # Size: 5 * 100,000,000 * 4 bytes = 2GB
            self.kv_buffer = torch.full(dummy_tensor_size, value, dtype=torch.float32, device='cuda')
        print(f'KV cache created: {_ptr(self.kv_buffer)}')

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg: torch.Tensor) -> torch.Tensor:
        """Execute KV cache operation, returns the combined result of input and cache data."""
        return (arg + self.kv_buffer.mean(dim=1)).mean()


# https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
def create_cuda_graph(fn: Callable):
    """
    Create CUDA graph for validating the impact of memory management on CUDA graphs.
    
    Validation objectives:
    1. Whether CUDA graphs can still work normally after pause/resume.
    2. The importance of virtual address remaining unchanged for CUDA graphs.
    3. CUDA graphs can still execute correctly even when physical memory is reallocated.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    return g


def run():
    """
    Main test function: validate the core functionality of torch_memory_saver.
    
    Core concept validation:
    
    1. Virtual Address vs Physical Memory mapping relationship.
       - Virtual Address remains unchanged during pause/resume operations.
       - Physical Memory is released during pause, reallocated during resume.
    
    2. Pause mechanism.
       - Disconnect the mapping from virtual address to physical memory.
       - Release physical memory back to GPU memory pool.
       - Keep virtual address unchanged.
       - Accessing paused tensors will cause CUDA errors.
    
    3. Resume mechanism.
       - Allocate new physical memory from GPU memory pool.
       - Re-establish mapping from virtual address to new physical memory.
       - Newly allocated physical memory content is undefined (usually 0).
       - Data needs to be reinitialized.
    
    4. Data consistency.
       - Data is lost during pause (physical memory is released).
       - Data is undefined after resume (new physical memory).
       - Data must be reset to be used normally.
    
    5. Functional integrity.
       - Tensor operations can still work normally even when physical memory is reallocated.
       - CUDA graphs can still execute correctly (because virtual address remains unchanged).
       - Selective pause/resume functionality works normally.
    """
    
    cache = KVCache()
    model = Model()
    original_kv_cache_ptr = _ptr(cache.kv_buffer)
    original_model_weights_ptr = _ptr(model.linear.weight)
    print(f'Original addresses - KV cache: {original_kv_cache_ptr}, Model weights: {original_model_weights_ptr}')
    
    # Create static input/output tensors for CUDA graphs
    # These tensors are not managed by torch_memory_saver, addresses will change normally
    static_input = torch.zeros((20_480,), dtype=torch.float32, device='cuda')
    static_output = torch.zeros((), dtype=torch.float32, device='cuda')

    def fn():
        """Function executed in CUDA graph: combine KV cache and model computation"""
        nonlocal static_output
        kv_output = cache.execute(static_input[:5])
        model_output = model.forward(static_input)
        static_output = kv_output + model_output

    # Create CUDA graph
    g = create_cuda_graph(fn)

    # First execution of CUDA graph
    static_input[...] = 100
    g.replay()
    print(f'First execution result: {static_output.item()}')
    if static_output != 2048101:
        raise ValueError(f'Expected 2048101, got {static_output}')
    print("✓ CUDA graph first execution passed!")

    time.sleep(3)

    # Pause memory regions
    print('\n=== Pausing memory regions ===')
    print_gpu_memory_gb("model_weights: allocated, kv_cache: allocated")
    torch_memory_saver.pause("kv_cache")
    print_gpu_memory_gb("model_weights: allocated, kv_cache: released")
    torch_memory_saver.pause("model_weights")
    print_gpu_memory_gb("model_weights: released, kv_cache: released")

    time.sleep(3)

    # Resume memory regions
    print('\n=== Resuming memory regions ===')
    torch_memory_saver.resume("model_weights")
    print_gpu_memory_gb("model_weights: resumed, kv_cache: released")
    torch_memory_saver.resume("kv_cache")
    print_gpu_memory_gb("model_weights: resumed, kv_cache: resumed")

    time.sleep(3)

    # Verify virtual addresses remain unchanged
    print('\n=== Virtual Address Verification ===')
    kv_cache_ptr_after_resume = _ptr(cache.kv_buffer)
    model_weights_ptr_after_resume = _ptr(model.linear.weight)
    
    kv_address_unchanged = kv_cache_ptr_after_resume == original_kv_cache_ptr
    model_address_unchanged = model_weights_ptr_after_resume == original_model_weights_ptr
    
    print(f'KV cache address unchanged: {kv_address_unchanged}')
    print(f'Model weights address unchanged: {model_address_unchanged}')
    
    assert kv_address_unchanged, f"KV cache virtual address changed"
    assert model_address_unchanged, f"Model weights virtual address changed"
    print("✓ Virtual addresses verification passed!")

    time.sleep(3)

    # Reinitialize data and test functionality
    print('\n=== Testing functionality after resume ===')
    cache.kv_buffer[...] = 2
    with torch.no_grad():
        model.linear.weight[...] = 2

    # Second execution of CUDA graph, validate functional integrity
    # Even when physical memory is reallocated, CUDA graph can still work normally
    # This is because virtual address remains unchanged, addresses recorded in CUDA graph are still valid
    static_input[...] = 200
    g.replay()
    print(f'Second execution result: {static_output.item()}')
    if static_output != 8192202:
        raise ValueError(f'Expected 8192202, got {static_output}')
    print("✓ CUDA graph second execution passed!")

    time.sleep(3)

    # Test selective pause/resume
    print('\n=== Testing selective pause/resume ===')
    torch_memory_saver.pause("kv_cache")
    print_gpu_memory_gb("model_weights: resumed, kv_cache: released")
    
    # Verify model weights can still be accessed
    try:
        _ = model.linear.weight[0, 0]
        print("✓ Model weights access successful")
    except:
        print("✗ Model weights access failed")
        raise
    
    torch_memory_saver.resume("kv_cache")
    print_gpu_memory_gb("model_weights: resumed, kv_cache: resumed")
    
    print("✓ Selective pause/resume test passed!")

    print("\n🎉 All tests passed! torch_memory_saver is working correctly.")
    os._exit(0)


if __name__ == '__main__':
    run()
