import logging
import sys

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    num_chunks = 8
    # Each chunk needs to be >= allocation granularity (typically 2MB).
    # 32MB per chunk * 8 chunks = 256MB total
    chunk_elements = 32 * 1024 * 1024  # 32M elements of uint8 = 32MB
    total_elements = chunk_elements * num_chunks

    # --- Test 1: Basic chunked allocation with pause/resume all ---
    print("=== Test 1: Chunked allocation, pause/resume all ===")
    with torch_memory_saver.region(tag="chunked_all", enable_cpu_backup=True, num_chunks=num_chunks):
        tensor = torch.zeros(total_elements, dtype=torch.uint8, device='cuda')
        # Fill each chunk region with a distinct value
        for i in range(num_chunks):
            tensor[i * chunk_elements:(i + 1) * chunk_elements] = i + 1

    original_address = tensor.data_ptr()
    print(f"Tensor virtual address: 0x{original_address:x}")

    mem_before = get_and_print_gpu_memory("Before pause all")

    torch_memory_saver.pause(tag="chunked_all")
    mem_after_pause = get_and_print_gpu_memory("After pause all")
    assert mem_before - mem_after_pause > 0.9 * total_elements, \
        f"Expected significant memory reduction, got {mem_before - mem_after_pause}"

    torch_memory_saver.resume(tag="chunked_all")
    mem_after_resume = get_and_print_gpu_memory("After resume all")
    assert mem_after_resume - mem_after_pause > 0.9 * total_elements

    # Verify data restored correctly
    assert original_address == tensor.data_ptr(), "Virtual address should be preserved"
    for i in range(num_chunks):
        chunk_data = tensor[i * chunk_elements:(i + 1) * chunk_elements]
        assert chunk_data[0].item() == i + 1, f"Chunk {i} data not restored: expected {i + 1}, got {chunk_data[0].item()}"
        assert chunk_data[-1].item() == i + 1, f"Chunk {i} tail data not restored"
    print("Test 1 PASSED: pause/resume all works with chunked allocation")

    # Clean up test 1
    del tensor
    torch.cuda.empty_cache()

    # --- Test 2: Per-chunk pause/resume ---
    print("\n=== Test 2: Per-chunk pause/resume ===")
    with torch_memory_saver.region(tag="chunked_partial", enable_cpu_backup=True, num_chunks=num_chunks):
        tensor2 = torch.zeros(total_elements, dtype=torch.uint8, device='cuda')
        for i in range(num_chunks):
            tensor2[i * chunk_elements:(i + 1) * chunk_elements] = (i + 1) * 10

    mem_before = get_and_print_gpu_memory("Before partial pause")

    # Pause chunks 2, 5, 7
    paused_indices = [2, 5, 7]
    torch_memory_saver.pause_chunks(tag="chunked_partial", chunk_indices=paused_indices)
    mem_after_partial = get_and_print_gpu_memory("After pausing chunks 2,5,7")

    # Should have freed ~3 chunks worth of memory
    expected_freed = 3 * chunk_elements
    actual_freed = mem_before - mem_after_partial
    print(f"Expected freed: ~{expected_freed}, actual freed: {actual_freed}")
    # Allow for alignment overhead (each chunk may be rounded up)
    assert actual_freed > 0.5 * expected_freed, \
        f"Expected to free ~{expected_freed} bytes, only freed {actual_freed}"

    # Active chunks should still be readable
    for i in range(num_chunks):
        if i not in paused_indices:
            chunk_data = tensor2[i * chunk_elements:(i + 1) * chunk_elements]
            assert chunk_data[0].item() == (i + 1) * 10, \
                f"Active chunk {i} should still be readable, got {chunk_data[0].item()}"
    print("Active chunks verified readable while others are paused")

    # Resume chunk 5
    torch_memory_saver.resume_chunks(tag="chunked_partial", chunk_indices=[5])
    mem_after_partial_resume = get_and_print_gpu_memory("After resuming chunk 5")
    assert mem_after_partial_resume > mem_after_partial, "Memory should increase after resuming a chunk"

    # Chunk 5 data should be restored
    chunk5_data = tensor2[5 * chunk_elements:6 * chunk_elements]
    assert chunk5_data[0].item() == 60, f"Chunk 5 data not restored: expected 60, got {chunk5_data[0].item()}"
    print("Chunk 5 data verified after resume")

    # Resume remaining paused chunks (2, 7)
    torch_memory_saver.resume_chunks(tag="chunked_partial", chunk_indices=[2, 7])

    # Verify all data
    for i in range(num_chunks):
        chunk_data = tensor2[i * chunk_elements:(i + 1) * chunk_elements]
        expected = (i + 1) * 10
        assert chunk_data[0].item() == expected, \
            f"Chunk {i} data not restored: expected {expected}, got {chunk_data[0].item()}"
    print("Test 2 PASSED: per-chunk pause/resume works correctly")

    # Clean up
    del tensor2
    torch.cuda.empty_cache()

    # --- Test 3: Chunked allocation without CPU backup ---
    print("\n=== Test 3: Chunked without CPU backup ===")
    with torch_memory_saver.region(tag="chunked_nobackup", enable_cpu_backup=False, num_chunks=4):
        tensor3 = torch.full((4 * chunk_elements,), 42, dtype=torch.uint8, device='cuda')

    # Pause specific chunks
    torch_memory_saver.pause_chunks(tag="chunked_nobackup", chunk_indices=[1, 3])

    # Active chunks should still work
    assert tensor3[0].item() == 42, "Chunk 0 should still be readable"
    assert tensor3[2 * chunk_elements].item() == 42, "Chunk 2 should still be readable"

    # Resume without backup - data will be uninitialized
    torch_memory_saver.resume_chunks(tag="chunked_nobackup", chunk_indices=[1, 3])
    # Just verify no crash - data content is undefined without backup
    _ = tensor3[1 * chunk_elements].item()
    print("Test 3 PASSED: chunked without CPU backup works (no crash)")

    del tensor3
    torch.cuda.empty_cache()

    # --- Test 4: Virtual address contiguity ---
    print("\n=== Test 4: Virtual address contiguity ===")
    with torch_memory_saver.region(tag="chunked_contig", enable_cpu_backup=True, num_chunks=4):
        tensor4 = torch.arange(4 * chunk_elements, dtype=torch.uint8, device='cuda')

    # Verify the tensor is contiguous
    assert tensor4.is_contiguous(), "Chunked tensor should be contiguous"
    assert tensor4.data_ptr() != 0, "Should have valid data pointer"

    # Pause and resume individual chunks, verify address stability
    addr_before = tensor4.data_ptr()
    torch_memory_saver.pause_chunks(tag="chunked_contig", chunk_indices=[0, 2])
    torch_memory_saver.resume_chunks(tag="chunked_contig", chunk_indices=[0, 2])
    addr_after = tensor4.data_ptr()
    assert addr_before == addr_after, "Virtual address must not change across pause/resume"
    print("Test 4 PASSED: virtual address preserved across per-chunk pause/resume")

    del tensor4
    torch.cuda.empty_cache()

    print("\n=== All chunked allocation tests PASSED ===")


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
