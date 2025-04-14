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

## Installation

For the library to work properly, it needs to be preloaded using the `LD_PRELOAD` environment variable.

### Option 1: Run a single command

After installation, you can run your scripts with:

```bash
tms python your_script.py
```

### Option 2: Activate for the current shell session

To activate torch_memory_saver for all Python programs in your current shell:

```bash
source tms
# Now run your Python programs normally
python your_script.py
```

### Option 3: Add to your shell profile

To make torch_memory_saver available in all your shells, add this line to your `~/.bashrc` or `~/.zshrc`:

```bash
source $(which tms)
```

## Technical Details

This library uses CUDA driver API functions to manage the lifetime of CUDA memory allocations. It requires `LD_PRELOAD` to intercept CUDA memory allocation calls made by PyTorch.

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

## TODO:

- [x] Implementation
- [x] Publish to pypi
- [ ] More tests and infra
- [ ] Documentation
