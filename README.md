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

XPU is **not** shipped as a prebuilt PyPI wheel (see "Why no XPU wheel" below);
install from source. `setup.py` auto-detects XPU when `icpx` is on `PATH`, so a
normal `pip install` builds the XPU backend with no extra flags:

```bash
# Prerequisites: Intel oneAPI (icpx) + Level Zero headers, and a torch+xpu build.
source /opt/intel/oneapi/setvars.sh          # put icpx on PATH (or set ICPX=...)

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git
# or, from a local checkout:
pip install .
```

> The build itself does not need torch (it links only SYCL + Level Zero), so
> build isolation is fine. XPU is auto-detected from `icpx` on `PATH`; if your
> oneAPI version differs from the one matching your torch runtime, pin it with
> `ICPX=/opt/intel/oneapi/compiler/<ver>/bin/icpx pip install .` (see the SONAME
> note below).

That's it — `import torch_memory_saver` then works exactly as on CUDA:

```python
import torch
from torch_memory_saver import TorchMemorySaver

saver = TorchMemorySaver()
saver.hook_mode = "torch"  # required on XPU

with saver.region(tag="weights"):
    x = torch.empty(1_000_000_000, dtype=torch.uint8, device="xpu")

saver.pause("weights")    # physical memory returned to the device
saver.resume("weights")   # re-committed at the same virtual address
```

For a standalone build (e.g. when integrating into another package's build
flow) you can also use the helper, which prints the resulting SONAME:

```bash
make build-xpu          # == bash scripts/build_xpu.sh
```

#### Why no XPU wheel?

The built `.so` links `libsycl.so.<N>`, whose major **must match the
`intel-sycl-rt` bundled with your `torch+xpu` wheel** (e.g. `torch 2.11.0+xpu` →
`intel-sycl-rt 2025.3.x` → `libsycl.so.8`). A single prebuilt wheel would break
the moment a user has a different torch-XPU build (a 2026.x oneAPI produces
`libsycl.so.9`, and the mismatch fails to load with
`undefined symbol: urDeviceWaitExp ... LIBUR_LOADER`). Building from source
against the locally installed runtime sidesteps this; `scripts/build_xpu.sh`
prints the linked SONAME so you can confirm the match
(`objdump -p ...torch_memory_saver_hook_mode_torch*.so | grep libsycl`).

#### Notes

Verifying that memory was actually released: `torch.xpu.memory_allocated()` (and
even `torch.xpu.mem_get_info()`) do **not** reflect physical pages released by
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
