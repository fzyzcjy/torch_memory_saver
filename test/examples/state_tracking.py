import sys
import torch
import pytest
import torch_memory_saver
from contextlib import nullcontext
from torch_memory_saver import torch_memory_saver as tms


def _state_tracking_basic(hook_mode: str):
    """Test basic state tracking - pause and resume operations"""
    tms.hook_mode = hook_mode
    with tms.region(tag="test_tag1"):
        tensor1 = torch.randn(1000, 1000, device='cuda')
        tensor2 = torch.randn(500, 500, device='cuda')

        # First pause should work
        tms.pause("test_tag1")

        # Second pause on same tag should throw an error (already paused)
        with pytest.raises(RuntimeError, match="Cannot pause allocation that is already paused"):
            tms.pause("test_tag1")

        # Resume should work
        tms.resume("test_tag1")

        # Second resume on same tag should throw an error (already active)
        with pytest.raises(RuntimeError, match="Cannot resume allocation that is already active"):
            tms.resume("test_tag1")

        # Another pause should work
        tms.pause("test_tag1")

        # Another resume should work
        tms.resume("test_tag1")

def _state_tracking_multiple_tags(hook_mode: str):
    """Test state tracking with multiple tags"""
    tms.hook_mode = hook_mode
    with tms.region(tag="test_tag2"):
        tensor1 = torch.randn(100, 100, device='cuda')

    with tms.region():
        tensor2 = torch.randn(200, 200, device='cuda')

    tms.pause("test_tag2")

    # Pause tag1 again (should throw error)
    with pytest.raises(RuntimeError, match="Cannot pause allocation that is already paused"):
        tms.pause("test_tag2")

    tms.resume()

    # Resume default tag (should throw error)
    with pytest.raises(RuntimeError, match="Cannot resume allocation that is already active"):
        tms.resume()


def _error_message_contains_tag(hook_mode: str):
    """Test that error messages contain the relevant tag information"""
    tms.hook_mode = hook_mode

    with tms.region(tag="test_tag3"):
        tensor = torch.randn(100, 100, device='cuda')

    with pytest.raises(RuntimeError) as exc_info:
        tms.resume("test_tag3")

    error_msg = str(exc_info.value)
    assert "test_tag3" in error_msg
    assert "already active" in error_msg


if __name__=="__main__":
    hook_mode = "preload" if len(sys.argv) < 2 else sys.argv[1]
    ctx = torch_memory_saver.configure_subprocess() if hook_mode == "preload" else nullcontext()
    with ctx:
        _state_tracking_basic(hook_mode)