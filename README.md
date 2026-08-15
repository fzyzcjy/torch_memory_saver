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

The default host shadow is pinned (`cpu_backup_backend="pinned"`). On CUDA, use `cpu_backup_backend="mmap"` (or `TMS_INIT_CPU_BACKUP_BACKEND=mmap`) when you need process RSS to reclaim after a non-retaining resume; both backends release the host shadow on resume unless retention is enabled (`munmap` / `cudaFreeHost`).

```python
with torch_memory_saver.region(enable_cpu_backup=True, cpu_backup_backend="mmap"):
    ...
```

ROCm/XPU stay pinned-only (`cpu_backup_backend="mmap"` is rejected); on the legacy ROCm path, host shadows are retained across resume.

`TMS_INIT_CPU_BACKUP_BACKEND=mmap|pinned` sets the process default for preload / env-driven integrations; an explicit `cpu_backup_backend=` argument overrides it.

On CUDA, pinned or mmap CPU backups can be retained across resume cycles to avoid
reallocating them before every pause:

```python
torch_memory_saver.retain_cpu_backup = True
```

Set `TMS_RETAIN_CPU_BACKUP=1` before process startup to enable the same policy,
including for preload mode. Retention is CUDA-only and can consume host RAM
equal to the backed-up allocations. While enabled, `get_cpu_backup` continues
to expose the retained backup after resume. Without retention on CUDA,
`get_cpu_backup` is only valid while allocations are paused.

The retention policy is consulted on resume. Setting it to `False` does not eagerly free
backups for active allocations; they are released by a later non-retaining
resume, when the allocation is freed, or when the process exits.

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

Same pause/resume behavior on Intel GPUs via Level Zero VMM, wired into PyTorch
through `XPUPluggableAllocator` + `torch.xpu.MemPool`. Only `hook_mode="torch"`
is supported (preload is CUDA/HIP-specific); pauseable CUDA-graph capture is not.

Install from source (no prebuilt wheel). `TMS_PLATFORM=xpu` forces the backend;
`--no-build-isolation` lets the build see `torch+xpu` and ABI-match `libsycl` to
it (otherwise a mismatched oneAPI silently produces a broken `.so`):

```bash
# Prerequisites: torch+xpu, Intel oneAPI (icpx) + Level Zero headers.
source /opt/intel/oneapi/setvars.sh          # put icpx on PATH (or set ICPX=...)
TMS_PLATFORM=xpu pip install --no-build-isolation .   # or `make build-xpu`
```

If the sourced oneAPI's `libsycl` major differs from your torch runtime's, pin a
matching compiler with `ICPX=/opt/intel/oneapi/compiler/<ver>/bin/icpx`. Then use
`region`/`pause`/`resume` as on CUDA with `hook_mode="torch"`, `device="xpu"`.

Note: `torch.xpu.memory_allocated()` / `mem_get_info()` do not reflect pages
released by `zeVirtualMemUnmap`; use sysman (`ZES_ENABLE_SYSMAN=1`) to verify.

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
