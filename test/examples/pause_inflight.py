import sys

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_device, is_xpu


def run(hook_mode: str, backup: str, device_count: int):
    assert backup in ("none", "cpu")
    torch_memory_saver.hook_mode = hook_mode
    if not is_xpu() and not torch.version.hip:
        torch_memory_saver.retain_cpu_backup = True
    mod = torch.get_device_module()
    mod.set_device(0)
    device_type = get_device()

    region_kwargs = dict(enable_cpu_backup=backup == "cpu")
    if backup == "cpu" and not is_xpu():
        region_kwargs["cpu_backup_backend"] = "pinned"

    allocations = []
    delay_buffers = []
    for device in range(device_count):
        with mod.device(device):
            with torch_memory_saver.region(tag="inflight", **region_kwargs):
                buffer = torch.full((4 * 1024 * 1024,), 11, dtype=torch.uint8, device=device_type)
            observed = torch.empty_like(buffer)
            stream = mod.Stream()
            stream.wait_stream(mod.current_stream())
            with mod.stream(stream):
                buffer.fill_(11)
                observed.copy_(buffer)
                if is_xpu():
                    delay_input = torch.ones((4096, 4096), device=device_type)
                    delay_output = torch.empty_like(delay_input)
                    torch.mm(delay_input, delay_input, out=delay_output)
                    delay_buffers.append((delay_input, delay_output))
                else:
                    torch.cuda._sleep(1)
            mod.synchronize()
            allocations.append((buffer, observed, stream))

    with torch_memory_saver.region(tag="other", enable_cpu_backup=True):
        untouched = torch.full((1024,), 91, dtype=torch.uint8, device=device_type)
    assert untouched.cpu().tolist() == [91] * 1024
    affected_devices = torch_memory_saver._impl._affected_devices
    assert affected_devices("inflight") == list(range(device_count))
    assert affected_devices("other") == [0]

    # Warm up backups; CUDA retention prevents later host allocations from
    # hiding missing synchronization.
    torch_memory_saver.pause("inflight")
    torch_memory_saver.resume("inflight")

    for iteration, tag in enumerate(("inflight", None)):
        for device, (buffer, observed, stream) in enumerate(allocations):
            before = 11 + device + iteration * 4
            buffer.fill_(before)
            assert torch.all(buffer.cpu() == before)

        for device, (buffer, observed, stream) in enumerate(allocations):
            after = 51 + device + iteration * 4
            with mod.stream(stream):
                # Waiting on device 0 must not incidentally drain the non-current device.
                if device == device_count - 1:
                    if is_xpu():
                        # XPU has no _sleep; queue warmed-up work without new allocations.
                        delay_input, delay_output = delay_buffers[device]
                        for _ in range(64):
                            torch.mm(delay_input, delay_input, out=delay_output)
                    else:
                        torch.cuda._sleep(1_000_000_000)
                observed.copy_(buffer)
                buffer.fill_(after)

        assert not allocations[-1][2].query()
        torch_memory_saver.pause(tag)
        assert mod.current_device() == 0
        assert all(stream.query() for _, _, stream in allocations)

        for device, (buffer, observed, stream) in enumerate(allocations):
            assert torch.all(observed.cpu() == 11 + device + iteration * 4)
            if backup == "cpu":
                saved = torch_memory_saver.get_cpu_backup(buffer)
                assert saved is not None
                assert torch.all(saved == 51 + device + iteration * 4)
        if tag is not None:
            assert untouched.cpu().tolist() == [91] * 1024

        torch_memory_saver.resume(tag)
        assert mod.current_device() == 0
        for device, (buffer, observed, stream) in enumerate(allocations):
            if backup == "cpu":
                assert torch.all(buffer.cpu() == 51 + device + iteration * 4)
            buffer.fill_(97)
            assert torch.all(buffer.cpu() == 97)
        assert untouched.cpu().tolist() == [91] * 1024

    # Torch-mode free callbacks require an active allocator region.
    with torch_memory_saver._impl._with_region_config(
        tag="inflight", enable_cpu_backup=False, cpu_backup_backend=None,
    ):
        del buffer, untouched
        allocations.clear()
        torch_memory_saver._impl._mem_pools.clear()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
