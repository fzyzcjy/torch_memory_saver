# torch_memory_saver

Allow torch tensor memory to be released and resumed later.

API:

```python
memory_saver = TorchMemorySaver()

# 1. For tensors that wants to be paused, create them within `region`
with memory_saver.region():
    x = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
memory_saver.resume()
```

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

TODO:

- [x] Implementation
- [x] Publish to pypi
- [ ] More tests and infra
- [ ] Documentation


## Development

```bash
pip install -e .
```

An `torch_memory_saver_cpp.abi3.so` will be built under `build/` folder.

You can use this command for testing:
```bash
LD_PRELOAD=/home/jobuser/torch_memory_saver/torch_memory_saver_cpp.abi3.so python examples/simple.py
```

### What do we expect from the example?

1. Created a normal tensor and a pauseable tensor, you should see some memory being used by the GPU
2. Pause the memory saver who is holding the pauseable tensor, you should see the GPU memory
used by the pauseable tensor is released
3. Resume the memory saver, you should see the GPU memory used by the pauseable tensor is restored
4. The virtual address of the pauseable tensor should be the same