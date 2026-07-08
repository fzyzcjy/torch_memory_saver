# Torch Memory Saver

A PyTorch library that allows tensor memory to be temporarily released and resumed later.

Please refer to https://github.com/sgl-project/sglang/issues/2542#issuecomment-2563641647 for details.

## Examples and Features

### Basic Example

```python
# 1. For tensors that wants to be paused, create them within `region`
with torch_memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
torch_memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
torch_memory_saver.resume()
```

During the pause, physical memory is released and virtual address is preserved. When resume, virtual address is kept unchanged, while physical memory is re-allocated

### Multiple Tags

Please refer to https://github.com/sgl-project/sglang/issues/7009 for details.

```python
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

### Release Memory in CUDA Graph

Not only does torch_memory_saver make tensors compatible with CUDA graph, but we can also release the memory held by CUDA graph (i.e. the intermediate tensors).

API: Change `torch.cuda.graph(...)` to `torch_memory_saver.cuda_graph(...)`

### CPU Backup

By default, in order to save time, the content is thrown away. This is useful for, for example, KV cache that are to be staled, or model weights that are to be updated.

If you want the tensor content to be kept unchanged, use `enable_cpu_backup`.

```python
with torch_memory_saver.region(enable_cpu_backup=True):
    tensor1 = torch.full((5_000_000_000,), 42, dtype=torch.uint8, device='cuda')

torch_memory_saver.pause()
torch_memory_saver.resume()

assert tensor1[0] == 42, "content is kept unchanged"
```

### Hook Modes

There are two hook modes:

* **preload**: Use `LD_PRELOAD` to hook CUDA's malloc and free API to change allocation behavior.
* **torch**: Use torch's custom allocator API to change allocation behavior.

The mode can be chosen by:

```python
torch_memory_saver.hook_mode = "torch"
```

### Example of RL with CUDA Graph

Please refer to `rl_example.py` for details.

## Platform Support

| Platform | Backend | Hook modes | Install |
|----------|---------|------------|---------|
| NVIDIA (CUDA) | CUDA VMM (`cuMemMap`/`cuMemCreate`) | preload, torch | `pip install torch_memory_saver` (prebuilt wheel) |
| AMD (ROCm) | HIP VMM (`hipMemMap`/`hipMemCreate`) | preload, torch | `pip install torch_memory_saver` (prebuilt wheel) |
| Intel (XPU) | Level Zero VMM (`zeVirtualMemMap`/`zePhysicalMemCreate`) | torch only | `pip install` from source (builds against local oneAPI) |

### Intel XPU

The Intel XPU backend provides the same pause/resume behavior on Intel GPUs,
built natively on Level Zero virtual memory. It is wired into PyTorch through
`torch.xpu.memory.XPUPluggableAllocator` + `torch.xpu.MemPool`, so **only
`hook_mode="torch"` is supported** (the `LD_PRELOAD`-based preload mode is
CUDA/HIP-specific). Pauseable CUDA-graph capture is not available on XPU.

#### Installation

XPU is not shipped as a prebuilt PyPI wheel; install from source. Set
`TMS_PLATFORM=xpu` to force the XPU backend (otherwise `setup.py` only
auto-detects it when neither `nvcc` nor `hipcc` is on `PATH`), and use
`--no-build-isolation` so the build sees your installed `torch+xpu` and can
match the `libsycl` major to it (under build isolation torch is not importable,
the ABI match is skipped, and a mismatched oneAPI silently produces a broken
`.so`):

```bash
# Prerequisites: a torch+xpu install, Intel oneAPI (icpx) + Level Zero headers.
source /opt/intel/oneapi/setvars.sh          # put icpx on PATH (or set ICPX=...)

TMS_PLATFORM=xpu pip install --no-build-isolation \
  "git+https://github.com/fzyzcjy/torch_memory_saver.git"
# or, from a local checkout:
TMS_PLATFORM=xpu pip install --no-build-isolation .
```

> If the sourced oneAPI's `libsycl` major differs from your torch runtime's,
> pin a matching compiler with
> `ICPX=/opt/intel/oneapi/compiler/<ver>/bin/icpx`.

Then use `region` / `pause` / `resume` as on CUDA (see the examples above), with
`hook_mode = "torch"` and `device="xpu"`. `make build-xpu` runs the same install.

#### Notes

Verifying that memory was actually released: `torch.xpu.memory_allocated()` (and
even `torch.xpu.mem_get_info()`) do not reflect physical pages released by
`zeVirtualMemUnmap`. Use sysman (`ZES_ENABLE_SYSMAN=1`) to observe real device
memory.

## Development

```bash
make reinstall
```

You can use this command for local testing:

```bash
pytest /path/to/torch_memory_saver/test
```

Or this one to test a single case (e.g. the `simple` one here):

```bash
pytest /path/to/torch_memory_saver/test/test_examples.py::test_simple -s
```
