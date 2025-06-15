# Torch Memory Saver

A PyTorch library that allows tensor memory to be temporarily released and resumed later.

During the pause:
- Physical memory is released
- Virtual address is preserved

When resume:
- Virtual address is restored to the original one

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

## Examples

### Basic Example

```python
import torch_memory_saver

memory_saver = torch_memory_saver.memory_saver

# 1. For tensors that wants to be paused, create them within `region`
with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
memory_saver.resume()
```

### Multiple Tags Example

Please refer to https://github.com/sgl-project/sglang/issues/7009 for details.

```python
from torch_memory_saver import torch_memory_saver

# 1. Create tensors with different tags
with torch_memory_saver.region(tag="type1"):
    tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="type2"):
    tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. Pause and resume with different tags selectively
torch_memory_saver.pause("type1")
torch_memory_saver.pause("type2")


torch_memory_saver.resume("type2")
torch_memory_saver.resume("type1")

torch_memory_saver.pause("type1")
torch_memory_saver.resume("type1")
```

## Development

```bash
pip install -e .
```

A `torch_memory_saver_cpp.abi3.so` will be built under `{your_workspace}/torch_memory_saver/` folder.

You can use this command for local testing:
```bash
LD_PRELOAD={your_workspace}/torch_memory_saver/torch_memory_saver_cpp.abi3.so python examples/simple.py

LD_PRELOAD={your_workspace}/torch_memory_saver/torch_memory_saver_cpp.abi3.so python examples/mock_rl.py
```