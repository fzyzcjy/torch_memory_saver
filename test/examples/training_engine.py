import logging
import os
import sys
from functools import reduce
from typing import List

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_gpu_memory, get_device, get_device_module


def run(hook_mode: str):
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert os.environ["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert hook_mode == "preload"

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    device = get_device()

    initial_tensor = torch.full((1_000_000,), 42, dtype=torch.uint8, device=device)
    mem_initial = get_and_print_gpu_memory("Init")

    model_weights = [
        torch.full((size,), 42, dtype=torch.uint8, device=device)
        for size in [1024 ** 3, 1024 ** 2, 1024 ** 1, 42]
    ]

    _execute_forward_pass_and_assert(model_weights, device)
    mem_after_forward_pass = get_and_print_gpu_memory("After forward pass")
    assert mem_after_forward_pass > mem_initial + 5 * 1024 ** 3

    torch_memory_saver.pause()
    mem_after_pause = get_and_print_gpu_memory("After pause")
    # muDNN keeps forward-pass workspaces resident, so MUSA needs more slack.
    pause_slack = 1024 ** 3 if device == "musa" else 200 * 1024 ** 2
    assert mem_after_pause < mem_initial + pause_slack

    with torch_memory_saver.disable():
        mem_after_disable = get_and_print_gpu_memory("After disable")
        assert mem_after_disable == mem_after_pause

        # Can still execute code in disabled region
        tensor_in_disabled_region = torch.full((1024 ** 3,), 53, dtype=torch.uint8, device=device)
        out = tensor_in_disabled_region.float().mean().item()
        assert out == 53, f"{out=}"

        mem_after_exec_in_disable = get_and_print_gpu_memory("After exec in disable")
        assert mem_after_exec_in_disable > mem_after_disable + 4 * 1024 ** 3

        del tensor_in_disabled_region

    # should do cleanup
    mem_after_exit_disable = get_and_print_gpu_memory("After exiting disable")
    assert mem_after_exit_disable <= mem_after_pause + 10 * 1024 ** 2

    torch_memory_saver.resume()
    mem_after_resume = get_and_print_gpu_memory("After resume")
    delta_expect = mem_after_forward_pass - mem_after_pause
    delta_actual = mem_after_resume - mem_after_exit_disable
    if device == "musa":
        # The MUSA disable release also drops the forward-pass cache, so resume
        # only re-maps the active model weights.
        expected_remap = sum(weight.nbytes for weight in model_weights)
        assert expected_remap - 200 * 1024 ** 2 < delta_actual < expected_remap + 200 * 1024 ** 2
    else:
        assert delta_expect - 50 * 1024 ** 2 < delta_actual < delta_expect + 50 * 1024 ** 2

    _execute_forward_pass_and_assert(model_weights, device)
    mem_after_second_forward_pass = get_and_print_gpu_memory("After second forward pass")
    if device == "musa":
        # mem_get_info() counts the forward-pass temporaries cached by the
        # allocator; the second pass should return to the first pass's level
        # (within the muDNN workspace noise).
        assert mem_after_forward_pass - 1024 ** 3 < mem_after_second_forward_pass < mem_after_forward_pass + 1024 ** 3
    else:
        assert mem_after_resume - 1024 ** 2 < mem_after_second_forward_pass < mem_after_resume + 1024 ** 2

    # simulate cache eviction
    # NOTE: here we assume the device is mostly free (the section needs ~90% of it)
    free_mem, total_mem = get_device_module().mem_get_info()
    if device == "musa" and free_mem < int(total_mem * 0.7):
        print(
            f"skip cache-eviction section: {free_mem / 1e9:.1f}GB free "
            f"of {total_mem / 1e9:.1f}GB"
        )
        return
    print(f"Total memory: {total_mem / 1e9}GB")
    a = torch.full((int(total_mem * 0.3),), 42, dtype=torch.uint8, device=device)
    get_and_print_gpu_memory("[cache-eviction-test] after alloc a")
    b = torch.full((int(total_mem * 0.3),), 42, dtype=torch.uint8, device=device)
    get_and_print_gpu_memory("[cache-eviction-test] after alloc b")
    del a, b
    get_and_print_gpu_memory("[cache-eviction-test] after del a,b")
    c = torch.full((int(total_mem * 0.6),), 42, dtype=torch.uint8, device=device)
    get_and_print_gpu_memory("[cache-eviction-test] after alloc c")
    initial_tensor += 1
    assert (initial_tensor.max() == 43) and (initial_tensor.min() == 43)
    get_and_print_gpu_memory("[cache-eviction-test] after using other tensors")

    del initial_tensor


def _execute_forward_pass_and_assert(weights: List[torch.Tensor], device: str):
    # simulate large activations during forward pass
    ones = torch.ones((1024 ** 3,), dtype=torch.float32, device=device)
    sum_avg_weights = reduce(lambda a, b: (a + b) / 2, [w.float().mean() for w in weights])
    outs = ones * sum_avg_weights
    out = outs.mean().item()
    assert out == 42, f"{out=}"


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
