import logging
import sys
import os

import torch
from torch_memory_saver import torch_memory_saver

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


def test_invalid_resume_without_pause():
    """Test that resume fails when no allocations are paused"""
    print("Testing resume without pause...")

    with torch_memory_saver.region():
        tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

    try:
        torch_memory_saver.resume()
        print("ERROR: Resume succeeded when it should have failed!")
        return False
    except RuntimeError:
        print("SUCCESS: Resume correctly failed when no allocations were paused")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

def test_double_pause():
    """Test that pause fails when allocations are already paused"""
    print("Testing double pause...")

    with torch_memory_saver.region():
        tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

    torch_memory_saver.pause()

    try:
        torch_memory_saver.pause()
        print("ERROR: Double pause succeeded when it should have failed!")
        return False
    except RuntimeError:
        print("SUCCESS: Double pause correctly failed")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

def test_pause_after_free():
    """Test that pause succeeds when allocations have been freed (metadata removed)"""
    print("Testing pause after free...")

    with torch_memory_saver.region():
        tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

    # Free the tensor
    del tensor
    torch.cuda.empty_cache()

    try:
        torch_memory_saver.pause()
        print("SUCCESS: Pause after free correctly succeeded (no allocations to pause)")
        return True
    except RuntimeError as e:
        if "No active allocations found" in str(e):
            print("SUCCESS: Pause after free correctly succeeded (no allocations to pause)")
            return True
        else:
            print(f"ERROR: Unexpected exception: {e}")
            return False
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

def test_resume_after_free():
    """Test that resume succeeds when allocations have been freed (metadata removed)"""
    print("Testing resume after free...")

    with torch_memory_saver.region():
        tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

    # Free the tensor
    del tensor
    torch.cuda.empty_cache()

    try:
        torch_memory_saver.resume()
        print("SUCCESS: Resume after free correctly succeeded (no allocations to resume)")
        return True
    except RuntimeError as e:
        if "No paused allocations found" in str(e):
            print("SUCCESS: Resume after free correctly succeeded (no allocations to resume)")
            return True
        else:
            print(f"ERROR: Unexpected exception: {e}")
            return False
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

def test_tagged_validation():
    """Test validation with tags"""
    print("Testing tagged validation...")

    with torch_memory_saver.region("tag1"):
        tensor1 = torch.full((500_000,), 100, dtype=torch.uint8, device='cuda')

    with torch_memory_saver.region("tag2"):
        tensor2 = torch.full((500_000,), 200, dtype=torch.uint8, device='cuda')

    # Pause tag1
    torch_memory_saver.pause("tag1")

    try:
        torch_memory_saver.resume("tag2")
        print("ERROR: Resume tag2 succeeded when it should have failed!")
        return False
    except RuntimeError:
        print("SUCCESS: Resume tag2 correctly failed")
    torch_memory_saver.resume("tag1")
    print("SUCCESS: Resume tag1 worked correctly")

    return True

def test_correct_usage():
    """Test that correct usage patterns still work"""
    print("Testing correct usage patterns...")

    with torch_memory_saver.region():
        tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device='cuda')

    # Correct pause/resume cycle
    torch_memory_saver.pause()
    torch_memory_saver.resume()

    print("SUCCESS: Correct pause/resume cycle worked")
    return True

def main():
    """Run all validation tests"""
    print("Running comprehensive state validation tests...")

    tests = [
        test_invalid_resume_without_pause,
        test_double_pause,
        test_pause_after_free,
        test_resume_after_free,
        test_tagged_validation,
        test_correct_usage
    ]

    for test in tests:
        try:
            test()
            print("-" * 50)
        except Exception as e:
            print(f"ERROR: Test {test.__name__} failed with exception: {e}")
            os._exit(1)
    os._exit(0)

if __name__ == "__main__":
    main()