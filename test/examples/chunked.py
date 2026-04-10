import logging
import multiprocessing
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


# 32 MiB chunks — safely above the typical 2 MiB allocation granularity on
# current NVIDIA/AMD devices, so the alignment check will pass.
CHUNK_SIZE_BYTES = 32 * 1024 * 1024
NUM_CHUNKS = 8
TOTAL_ELEMENTS = CHUNK_SIZE_BYTES * NUM_CHUNKS  # 256 MiB of uint8


# Module-level error scenarios. They must be picklable because spawn()
# re-imports the module in the grandchild. Each should trigger a fatal
# SIMPLE_CHECK in the C++ layer.

def _error_non_divisible_size(hook_mode):
    torch_memory_saver.hook_mode = hook_mode
    # 100 bytes is not a multiple of 32 MiB — malloc must abort.
    with torch_memory_saver.region(tag="bad_div", enable_cpu_backup=False, chunk_size=CHUNK_SIZE_BYTES):
        _ = torch.zeros(100, dtype=torch.uint8, device='cuda')


def _error_non_aligned_chunk_size(hook_mode):
    torch_memory_saver.hook_mode = hook_mode
    # chunk_size=1 is below the device granularity — malloc must abort.
    with torch_memory_saver.region(tag="bad_align", enable_cpu_backup=False, chunk_size=1):
        _ = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device='cuda')


def _error_pause_chunks_on_plain(hook_mode):
    torch_memory_saver.hook_mode = hook_mode
    with torch_memory_saver.region(tag="plain", enable_cpu_backup=False):
        tensor = torch.zeros(1024, dtype=torch.uint8, device='cuda')
    # Hold the tensor alive so find_unique_by_tag_locked finds it and hits
    # the "non-chunked allocation" check rather than silently returning.
    assert tensor.is_cuda
    torch_memory_saver.pause_chunks(tag="plain", chunk_indices=[0])


def _run_expecting_failure(target, hook_mode, description):
    ctx = multiprocessing.get_context('spawn')
    proc = ctx.Process(target=target, args=(hook_mode,))
    proc.start()
    proc.join(timeout=120)
    assert proc.exitcode is not None and proc.exitcode != 0, \
        f"{description}: expected nonzero exit, got {proc.exitcode}"
    print(f"  {description}: correctly exited with code {proc.exitcode}")


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # --- Test 1: Basic chunked allocation with pause/resume all ---
    print("=== Test 1: Chunked allocation, pause/resume all ===")
    with torch_memory_saver.region(tag="chunked_all", enable_cpu_backup=True, chunk_size=CHUNK_SIZE_BYTES):
        tensor = torch.zeros(TOTAL_ELEMENTS, dtype=torch.uint8, device='cuda')
        for i in range(NUM_CHUNKS):
            tensor[i * CHUNK_SIZE_BYTES:(i + 1) * CHUNK_SIZE_BYTES] = i + 1

    assert torch_memory_saver.get_num_chunks(tag="chunked_all") == NUM_CHUNKS
    original_address = tensor.data_ptr()

    mem_before = get_and_print_gpu_memory("Before pause all")
    torch_memory_saver.pause(tag="chunked_all")
    mem_after_pause = get_and_print_gpu_memory("After pause all")
    assert mem_before - mem_after_pause > 0.9 * TOTAL_ELEMENTS, \
        f"Expected significant memory reduction, got {mem_before - mem_after_pause}"

    torch_memory_saver.resume(tag="chunked_all")
    mem_after_resume = get_and_print_gpu_memory("After resume all")
    assert mem_after_resume - mem_after_pause > 0.9 * TOTAL_ELEMENTS

    assert original_address == tensor.data_ptr(), "Virtual address should be preserved"
    for i in range(NUM_CHUNKS):
        chunk_data = tensor[i * CHUNK_SIZE_BYTES:(i + 1) * CHUNK_SIZE_BYTES]
        assert chunk_data[0].item() == i + 1, f"chunk {i} head lost"
        assert chunk_data[-1].item() == i + 1, f"chunk {i} tail lost"
    print("Test 1 PASSED")

    del tensor
    torch.cuda.empty_cache()

    # --- Test 2: Per-chunk pause/resume ---
    print("\n=== Test 2: Per-chunk pause/resume ===")
    with torch_memory_saver.region(tag="chunked_partial", enable_cpu_backup=True, chunk_size=CHUNK_SIZE_BYTES):
        tensor2 = torch.zeros(TOTAL_ELEMENTS, dtype=torch.uint8, device='cuda')
        for i in range(NUM_CHUNKS):
            tensor2[i * CHUNK_SIZE_BYTES:(i + 1) * CHUNK_SIZE_BYTES] = (i + 1) * 10

    mem_before = get_and_print_gpu_memory("Before partial pause")

    paused_indices = [2, 5, 7]
    torch_memory_saver.pause_chunks(tag="chunked_partial", chunk_indices=paused_indices)
    mem_after_partial = get_and_print_gpu_memory("After pausing chunks 2,5,7")

    expected_freed = 3 * CHUNK_SIZE_BYTES
    actual_freed = mem_before - mem_after_partial
    assert actual_freed > 0.5 * expected_freed, \
        f"Expected ~{expected_freed} bytes freed, only freed {actual_freed}"

    for i in range(NUM_CHUNKS):
        if i not in paused_indices:
            chunk_data = tensor2[i * CHUNK_SIZE_BYTES:(i + 1) * CHUNK_SIZE_BYTES]
            assert chunk_data[0].item() == (i + 1) * 10

    torch_memory_saver.resume_chunks(tag="chunked_partial", chunk_indices=[5])
    mem_after_partial_resume = get_and_print_gpu_memory("After resuming chunk 5")
    assert mem_after_partial_resume > mem_after_partial
    chunk5_data = tensor2[5 * CHUNK_SIZE_BYTES:6 * CHUNK_SIZE_BYTES]
    assert chunk5_data[0].item() == 60

    torch_memory_saver.resume_chunks(tag="chunked_partial", chunk_indices=[2, 7])
    for i in range(NUM_CHUNKS):
        chunk_data = tensor2[i * CHUNK_SIZE_BYTES:(i + 1) * CHUNK_SIZE_BYTES]
        assert chunk_data[0].item() == (i + 1) * 10
    print("Test 2 PASSED")

    del tensor2
    torch.cuda.empty_cache()

    # --- Test 3: Chunked allocation without CPU backup ---
    print("\n=== Test 3: Chunked without CPU backup ===")
    with torch_memory_saver.region(tag="chunked_nobackup", enable_cpu_backup=False, chunk_size=CHUNK_SIZE_BYTES):
        tensor3 = torch.full((4 * CHUNK_SIZE_BYTES,), 42, dtype=torch.uint8, device='cuda')

    assert torch_memory_saver.get_num_chunks(tag="chunked_nobackup") == 4
    torch_memory_saver.pause_chunks(tag="chunked_nobackup", chunk_indices=[1, 3])
    assert tensor3[0].item() == 42
    assert tensor3[2 * CHUNK_SIZE_BYTES].item() == 42

    torch_memory_saver.resume_chunks(tag="chunked_nobackup", chunk_indices=[1, 3])
    _ = tensor3[1 * CHUNK_SIZE_BYTES].item()  # no crash
    print("Test 3 PASSED")

    del tensor3
    torch.cuda.empty_cache()

    # --- Test 4: Virtual address contiguity ---
    print("\n=== Test 4: Virtual address contiguity ===")
    with torch_memory_saver.region(tag="chunked_contig", enable_cpu_backup=True, chunk_size=CHUNK_SIZE_BYTES):
        tensor4 = torch.arange(4 * CHUNK_SIZE_BYTES, dtype=torch.uint8, device='cuda')

    assert tensor4.is_contiguous()
    assert tensor4.data_ptr() != 0
    addr_before = tensor4.data_ptr()
    torch_memory_saver.pause_chunks(tag="chunked_contig", chunk_indices=[0, 2])
    torch_memory_saver.resume_chunks(tag="chunked_contig", chunk_indices=[0, 2])
    assert tensor4.data_ptr() == addr_before
    print("Test 4 PASSED")

    del tensor4
    torch.cuda.empty_cache()

    # --- Test 5: num_chunks == 1 edge case (chunk_size == allocation size) ---
    # Previously this path crashed on free() because dispatch was keyed on
    # num_chunks > 1 and left metadata.allocHandle uninitialized.
    print("\n=== Test 5: num_chunks == 1 (chunk_size == size) ===")
    with torch_memory_saver.region(tag="chunked_one", enable_cpu_backup=True, chunk_size=CHUNK_SIZE_BYTES):
        tensor5 = torch.full((CHUNK_SIZE_BYTES,), 77, dtype=torch.uint8, device='cuda')

    assert torch_memory_saver.get_num_chunks(tag="chunked_one") == 1
    torch_memory_saver.pause(tag="chunked_one")
    torch_memory_saver.resume(tag="chunked_one")
    assert tensor5[0].item() == 77
    assert tensor5[-1].item() == 77
    del tensor5
    torch.cuda.empty_cache()
    print("Test 5 PASSED")

    # --- Test 6: Mixed pause / pause_chunks composition ---
    print("\n=== Test 6: Mixed pause/pause_chunks composition ===")
    with torch_memory_saver.region(tag="chunked_mixed", enable_cpu_backup=True, chunk_size=CHUNK_SIZE_BYTES):
        tensor6 = torch.zeros(4 * CHUNK_SIZE_BYTES, dtype=torch.uint8, device='cuda')
        for i in range(4):
            tensor6[i * CHUNK_SIZE_BYTES:(i + 1) * CHUNK_SIZE_BYTES] = i + 100

    # Pause a subset, then ask the whole-allocation pause to finish the job.
    torch_memory_saver.pause_chunks(tag="chunked_mixed", chunk_indices=[0, 2])
    torch_memory_saver.pause(tag="chunked_mixed")
    torch_memory_saver.resume(tag="chunked_mixed")
    for i in range(4):
        assert tensor6[i * CHUNK_SIZE_BYTES].item() == i + 100, f"round 1 chunk {i} lost"

    # Reverse: pause all, resume a subset, then resume the rest.
    torch_memory_saver.pause(tag="chunked_mixed")
    torch_memory_saver.resume_chunks(tag="chunked_mixed", chunk_indices=[1, 3])
    torch_memory_saver.resume(tag="chunked_mixed")
    for i in range(4):
        assert tensor6[i * CHUNK_SIZE_BYTES].item() == i + 100, f"round 2 chunk {i} lost"

    # Idempotency: chunk-level ops on a chunk already in the target state
    # must be silent no-ops.
    torch_memory_saver.pause_chunks(tag="chunked_mixed", chunk_indices=[0])
    torch_memory_saver.pause_chunks(tag="chunked_mixed", chunk_indices=[0])
    torch_memory_saver.resume_chunks(tag="chunked_mixed", chunk_indices=[0, 1])
    for i in range(4):
        assert tensor6[i * CHUNK_SIZE_BYTES].item() == i + 100, f"round 3 chunk {i} lost"

    del tensor6
    torch.cuda.empty_cache()
    print("Test 6 PASSED")

    # --- Test 7: get_chunk_states introspection ---
    print("\n=== Test 7: get_chunk_states ===")
    with torch_memory_saver.region(tag="chunked_states", enable_cpu_backup=True, chunk_size=CHUNK_SIZE_BYTES):
        tensor7 = torch.zeros(4 * CHUNK_SIZE_BYTES, dtype=torch.uint8, device='cuda')

    assert torch_memory_saver.get_chunk_states(tag="chunked_states") == [True, True, True, True]

    torch_memory_saver.pause_chunks(tag="chunked_states", chunk_indices=[1, 2])
    assert torch_memory_saver.get_chunk_states(tag="chunked_states") == [True, False, False, True]

    torch_memory_saver.pause(tag="chunked_states")
    assert torch_memory_saver.get_chunk_states(tag="chunked_states") == [False, False, False, False]

    torch_memory_saver.resume(tag="chunked_states")
    assert torch_memory_saver.get_chunk_states(tag="chunked_states") == [True, True, True, True]

    # Missing tag returns an empty list rather than erroring.
    assert torch_memory_saver.get_chunk_states(tag="nonexistent_tag_xyz") == []

    del tensor7
    torch.cuda.empty_cache()
    print("Test 7 PASSED")

    # --- Test 8: Fatal-error paths (nested subprocesses expected to abort) ---
    print("\n=== Test 8: Error cases via nested subprocess ===")
    _run_expecting_failure(_error_non_divisible_size, hook_mode, "non-divisible size")
    _run_expecting_failure(_error_non_aligned_chunk_size, hook_mode, "non-granularity-aligned chunk_size")
    _run_expecting_failure(_error_pause_chunks_on_plain, hook_mode, "pause_chunks on non-chunked allocation")
    print("Test 8 PASSED")

    print("\n=== All chunked allocation tests PASSED ===")


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
