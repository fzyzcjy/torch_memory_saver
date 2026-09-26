from types import SimpleNamespace

import pytest

from torch_memory_saver.entrypoint import _TorchMemorySaverImpl


@pytest.mark.parametrize(
    "devices", [[], [0, 2], list(range(64)), list(range(65))],
    ids=["empty", "sparse", "capacity", "overflow"],
)
def test_affected_devices(devices):
    calls = 0

    def query(tag, output, capacity):
        nonlocal calls
        calls += 1
        assert tag == b"weights"
        for i, device in enumerate(devices[:capacity]):
            output[i] = device
        return len(devices)

    impl = SimpleNamespace(
        _binary_wrapper=SimpleNamespace(cdll=SimpleNamespace(tms_affected_devices=query))
    )
    if len(devices) > 64:
        with pytest.raises(AssertionError, match="Too many affected devices"):
            _TorchMemorySaverImpl._affected_devices(impl, "weights")
    else:
        assert _TorchMemorySaverImpl._affected_devices(impl, "weights") == devices
    assert calls == 1
