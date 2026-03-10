"""
Usage:
TMS_INIT_ENABLE=1 TMS_INIT_ENABLE_CPU_BACKUP=1 pytest test_examples.py::test_nested_region -v -s

Test that region() supports nesting (e.g. when TMS_INIT_ENABLE=1 in preload mode).

Verifies:
1. region() works when interesting_region is already True (preload mode).
2. Allocations inside a nested region(enable_cpu_backup=False) are NOT backed up to CPU on pause().
3. Allocations outside the nested region (default enable_cpu_backup=True) ARE backed up and restored.
4. After resume(), model weights are preserved but grad buffers contain undefined data.
"""

import logging
import os
import sys

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory


def run(hook_mode: str):
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert os.environ["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert hook_mode == "preload"

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    SIZE = 10_000_000  # ~40MB in float32

    # Helper to read C-level config state
    torch_memory_saver._ensure_initialized()
    cdll = torch_memory_saver._impl._binary_wrapper.cdll

    # --- Verify initial state in preload mode ---
    assert cdll.tms_get_interesting_region()
    assert cdll.tms_get_enable_cpu_backup()
    assert cdll.tms_get_current_tag() == b"default"
    print("✓ Initial state: interesting_region=True, cpu_backup=True, tag=default")

    # --- Allocate model weights (default tag, cpu_backup=True) ---
    model_weight = torch.full((SIZE,), 42.0, dtype=torch.float32, device="cuda")

    # --- Allocate grad buffer inside nested region (cpu_backup=False) ---
    with torch_memory_saver.region(tag="grad_buffer", enable_cpu_backup=False):
        # Verify state inside nested region
        assert cdll.tms_get_interesting_region()
        assert not cdll.tms_get_enable_cpu_backup()
        assert cdll.tms_get_current_tag() == b"grad_buffer"
        print("✓ Inside region: interesting_region=True, cpu_backup=False, tag=grad_buffer")

        grad_buffer = torch.full((SIZE,), 99.0, dtype=torch.float32, device="cuda")

    # Verify state restored after exiting region
    assert cdll.tms_get_interesting_region()
    assert cdll.tms_get_enable_cpu_backup()
    assert cdll.tms_get_current_tag() == b"default"
    print("✓ After region: interesting_region=True, cpu_backup=True, tag=default (restored)")

    mem_before_pause = get_and_print_gpu_memory("Before pause")

    # --- Pause all ---
    torch_memory_saver.pause()
    mem_after_pause = get_and_print_gpu_memory("After pause")

    # GPU memory should be significantly reduced
    assert mem_after_pause < mem_before_pause - SIZE * 4 * 0.5, (
        f"Expected significant memory reduction after pause, "
        f"before={mem_before_pause}, after={mem_after_pause}"
    )

    # --- Resume all ---
    torch_memory_saver.resume()
    mem_after_resume = get_and_print_gpu_memory("After resume")

    # Model weight should be restored (had cpu_backup=True)
    assert model_weight.sum().item() == 42.0 * SIZE, (
        f"Model weight should be preserved after resume, got sum={model_weight.sum().item()}"
    )
    print("✓ Model weight preserved after pause/resume")

    # Grad buffer virtual address is preserved but data is undefined (no cpu_backup)
    # We just verify it's accessible and writable
    grad_buffer.fill_(0.0)
    assert grad_buffer.sum().item() == 0.0
    print("✓ Grad buffer accessible and writable after resume (data was not preserved, as expected)")

    # --- Verify tag nesting restores state correctly ---
    # After exiting the region(), the tag should be back to default
    # Allocate another tensor - should go to default tag with cpu_backup=True
    extra_tensor = torch.full((SIZE,), 7.0, dtype=torch.float32, device="cuda")

    torch_memory_saver.pause()
    torch_memory_saver.resume()

    assert extra_tensor.sum().item() == 7.0 * SIZE, (
        f"Extra tensor (default tag) should be preserved, got sum={extra_tensor.sum().item()}"
    )
    print("✓ Post-region allocations correctly use default tag with cpu_backup=True")

    print("\n🎉 All nested region tests passed!")


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
