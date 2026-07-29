"""XPU-specific torch_memory_saver scenarios, one run_* per scenario.

test/test_examples.py spawns each in its own subprocess, so a fault-injection
abort or exit(1) in one cannot corrupt the others. Standalone:

  python test/examples/xpu_scenarios.py <scenario> [hook_mode]
  (scenarios: free_failure cleanup_failure resume_failure
              memory_margin_unsupported disable_unsupported multi_device_sync)
"""

import ctypes
import os
import sys
import tempfile

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import (
    get_and_print_gpu_memory,
    get_device,
    is_xpu,
)


def _committed_bytes(gpu_id=0):
    return int(torch_memory_saver._impl._binary_wrapper.cdll.tms_xpu_committed_bytes(gpu_id))


def _leaked_bytes(gpu_id=0):
    return int(torch_memory_saver._impl._binary_wrapper.cdll.tms_xpu_leaked_bytes(gpu_id))


def _tracked_bytes(gpu_id=0):
    return int(torch_memory_saver._impl._binary_wrapper.cdll.tms_xpu_tracked_bytes(gpu_id))


def run_free_failure(hook_mode: str):
    """free() retains ownership until every Level Zero release step succeeds (XPU).

    Fails each step via TMS_XPU_FAULT_FREE_*; the record must stay tracked and a
    retry advance exactly one step (unmap -> destroy -> free-VA), erased only once
    all three confirm. Pre-fix free() erased the entry first and ignored return
    codes, permanently leaking physical memory or VA with no way to retry.
    """
    assert hook_mode == "torch", "free failure-injection test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU free-ownership path"

    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()

    impl = torch_memory_saver._impl
    cdll = impl._binary_wrapper.cdll
    for sym in ("tms_xpu_free", "tms_xpu_tracked_bytes", "tms_xpu_committed_bytes",
                "tms_xpu_leaked_bytes"):
        assert hasattr(cdll, sym), f"need a build exposing {sym}; rebuild the XPU extension"

    _size = 4_000_000  # bytes; aligned up to the L0 page granularity by the backend

    cdll.tms_torch_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    cdll.tms_torch_malloc.restype = ctypes.c_void_p

    impl._binary_wrapper.set_config(tag="freetest", interesting_region=True, enable_cpu_backup=False)
    try:
        ptr = cdll.tms_torch_malloc(_size, 0, None)
    finally:
        impl._binary_wrapper.set_config(tag="default", interesting_region=False, enable_cpu_backup=False)
    assert ptr, "raw VMM malloc failed"

    full = _tracked_bytes()
    assert full > 0, "allocation should be tracked after malloc"
    assert _committed_bytes() == full, "freshly mapped allocation is ACTIVE (committed)"
    assert _leaked_bytes() == 0, "nothing leaked before any failure"

    def _free(expect_ok):
        rc = int(cdll.tms_xpu_free(ptr))
        if expect_ok:
            assert rc == 0, f"free expected to succeed, got rc={rc}"
        else:
            assert rc != 0, "free expected to fail (fault armed) but returned success"
        return rc

    os.environ["TMS_XPU_FAULT_FREE_UNMAP"] = "1"
    try:
        _free(expect_ok=False)
        assert _tracked_bytes() == full, "unmap failure must RETAIN the ownership record"
        assert _committed_bytes() == full, "unmap failure leaves the allocation ACTIVE"
        assert _leaked_bytes() == 0, "no handle orphaned yet (unmap never happened)"
    finally:
        del os.environ["TMS_XPU_FAULT_FREE_UNMAP"]

    os.environ["TMS_XPU_FAULT_FREE_DESTROY"] = "1"
    try:
        _free(expect_ok=False)
        assert _tracked_bytes() == full, "destroy failure must RETAIN the ownership record"
        assert _committed_bytes() == 0, "VA unmapped -> not ACTIVE (committed 0)"
        assert _leaked_bytes() == full, "undestroyed physical handle must show as leaked"
    finally:
        del os.environ["TMS_XPU_FAULT_FREE_DESTROY"]


    os.environ["TMS_XPU_FAULT_FREE_VA"] = "1"
    try:
        _free(expect_ok=False)
        assert _committed_bytes() == 0, "not ACTIVE"
        assert _leaked_bytes() == 0, "physical handle already destroyed; nothing leaked"
        assert _tracked_bytes() == full, (
            "free-VA failure must RETAIN the ownership record even though both "
            "committed and leaked read 0 -- this is the state the pre-fix code "
            "dropped silently, permanently leaking the reserved VA"
        )
    finally:
        del os.environ["TMS_XPU_FAULT_FREE_VA"]

    _free(expect_ok=True)
    assert _tracked_bytes() == 0, "successful free must release and erase the record"
    assert _committed_bytes() == 0 and _leaked_bytes() == 0

    assert int(cdll.tms_xpu_free(ptr)) != 0, "double free must report invalid pointer"

    print("free_failure_injection_xpu OK")


def run_cleanup_failure(hook_mode: str):
    """Ownership is retained until Level Zero confirms release (XPU).

    A failed zePhysicalMemDestroy in pause() leaves the handle alive: it must be
    retained (visible as leaked, since committed reads 0 once unmapped) so resume
    re-maps THAT handle, or free destroys it -- pre-fix the handle was cleared and
    the next resume allocated a second one, orphaning the first.
    """
    assert hook_mode == "torch", "cleanup failure-injection test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU cleanup-ownership path"

    device = get_device()
    magic = 77  # sentinel value stored in the pauseable tensor
    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()

    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    assert hasattr(cdll, "tms_xpu_committed_bytes"), (
        "need a build exposing tms_xpu_committed_bytes"
    )
    assert hasattr(cdll, "tms_xpu_leaked_bytes"), (
        "need a build exposing tms_xpu_leaked_bytes to observe retained handles "
        "independently of the ACTIVE-only committed counter"
    )

    with torch_memory_saver.region(tag="t", enable_cpu_backup=True):
        x = torch.full((4_000_000,), magic, dtype=torch.uint8, device=f"{device}:0")
    torch.xpu.synchronize()
    committed_active = _committed_bytes()
    assert committed_active > 0, "allocation should be committed while ACTIVE"
    assert _leaked_bytes() == 0, "nothing leaked before any failure"

    os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"] = "1"
    try:
        raised = False
        try:
            torch_memory_saver.pause("t")
        except RuntimeError:
            raised = True
        assert raised, "pause() must raise when a physical handle cannot be released"

        assert _committed_bytes() == 0, (
            "destroy-failure leaves the VA unmapped (0 committed)"
        )
        assert _leaked_bytes() == committed_active, (
            "retained physical handle must be tracked as leaked, not dropped "
            "(this is the leak the ACTIVE-only counter hides)"
        )
    finally:
        del os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"]

    torch_memory_saver.resume("t")
    assert _committed_bytes() == committed_active, (
        "resume must re-commit the retained handle"
    )
    assert _leaked_bytes() == 0, (
        "resume must reclaim the retained handle (re-map it), clearing the leak"
    )
    assert x[:3].tolist() == [magic, magic, magic], (
        "data must be restored intact after recovering from a pause destroy-failure"
    )

    torch_memory_saver.pause("t")
    assert _committed_bytes() == 0 and _leaked_bytes() == 0, (
        "a clean pause after recovery releases the handle with no leak"
    )
    torch_memory_saver.resume("t")
    assert _committed_bytes() == committed_active and _leaked_bytes() == 0
    assert x[:3].tolist() == [magic, magic, magic]

    os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"] = "1"
    try:
        try:
            torch_memory_saver.pause("t")
        except RuntimeError:
            pass
        assert _leaked_bytes() == committed_active, "handle should be retained as leaked"
    finally:
        del os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"]

    del x
    torch_memory_saver._impl._mem_pools.clear()
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    assert _committed_bytes() == 0, "free must release all committed bytes"
    assert _leaked_bytes() == 0, (
        "free must reclaim the retained physical handle, not leak it"
    )

    print("cleanup_failure_injection_xpu OK")


def run_resume_failure(hook_mode: str):
    """Transactional resume() under injected create/map/restore failures (XPU).

    Each TMS_XPU_FAULT_RESUME_{CREATE,MAP,RESTORE} must raise and roll back to a
    clean PAUSED state with the backup preserved -- never report success with a
    tensor unmapped or holding undefined contents -- and a retry must fully resume.
    """
    assert hook_mode == "torch", "resume failure-injection test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU transactional resume path"

    device = get_device()
    magic = 123
    faults = (
        "TMS_XPU_FAULT_RESUME_CREATE",
        "TMS_XPU_FAULT_RESUME_MAP",
        "TMS_XPU_FAULT_RESUME_RESTORE",
    )
    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()

    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    assert hasattr(cdll, "tms_xpu_committed_bytes"), (
        "need a build exposing tms_xpu_committed_bytes to verify commit/rollback"
    )

    with torch_memory_saver.region(tag="t", enable_cpu_backup=True):
        x = torch.full((4_000_000,), magic, dtype=torch.uint8, device=f"{device}:0")
    torch.xpu.synchronize()
    committed_active = _committed_bytes()
    assert committed_active > 0, "allocation should be committed while ACTIVE"

    for fault in faults:
        torch_memory_saver.pause("t")
        assert _committed_bytes() == 0, f"{fault}: pause should release physical pages"

        os.environ[fault] = "1"
        try:
            raised = False
            try:
                torch_memory_saver.resume("t")
            except RuntimeError:
                raised = True
            assert raised, f"{fault}: resume() must raise on injected failure"
            assert _committed_bytes() == 0, (
                f"{fault}: failed resume must leave the allocation PAUSED"
            )
        finally:
            del os.environ[fault]

        torch_memory_saver.resume("t")
        assert _committed_bytes() == committed_active, (
            f"{fault}: retry after clearing fault should fully re-commit"
        )
        assert x[:3].tolist() == [magic, magic, magic], (
            f"{fault}: data must be restored intact after retry"
        )

    print("resume_failure_injection_xpu OK")


def _reported_free_bytes(gpu_id=0):
    """Best available device free-bytes reading (what a margin check would use);
    both the sysman and torch sources are frozen on current drivers."""
    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    if hasattr(cdll, "tms_xpu_device_free_bytes"):
        return int(cdll.tms_xpu_device_free_bytes(gpu_id))
    free, _total = torch.xpu.mem_get_info(gpu_id)
    return int(free)


def run_memory_margin_unsupported(hook_mode: str):
    """memory_margin_bytes is rejected (not silently ignored) on XPU.

    The CUDA OOM guard needs an accurate device free-bytes reading, which Intel
    drivers freeze, so the XPU malloc path never applies the margin. Asserts (a) the
    Python setter raises, (b) the telemetry really is frozen, (c) the C ABI, which
    bypasses (a), refuses too rather than storing a margin malloc ignores.
    """
    assert hook_mode == "torch", "memory_margin_unsupported_xpu test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU rejection path"

    device = get_device()
    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()

    raised = False
    try:
        torch_memory_saver.memory_margin_bytes = 1 << 30  # 1 GiB
    except NotImplementedError:
        raised = True
    assert raised, "memory_margin_bytes should raise NotImplementedError on XPU"

    free_before = _reported_free_bytes()
    with torch_memory_saver.region(tag="t"):
        big = torch.empty(1_500_000_000, dtype=torch.uint8, device=f"{device}:0")
    torch.xpu.synchronize()
    free_after = _reported_free_bytes()
    assert free_before - free_after < (256 << 20), (
        "device free-bytes telemetry is expected to be frozen on XPU "
        f"(before={free_before} after={free_after}); if this ever starts "
        "tracking allocations, memory_margin_bytes could be implemented for real "
        "instead of rejected"
    )

    del big
    probe = torch.full((1024,), 3.0, dtype=torch.float32, device=f"{device}:0")
    assert float(probe[0]) == 3.0

    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tf:
        os.dup2(tf.fileno(), 2)
        try:
            cdll.set_memory_margin_bytes(ctypes.c_uint64(1 << 60))  # 1 EiB
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        tf.seek(0)
        captured = tf.read().decode("utf-8", "replace")
    assert "NOT supported on Intel XPU" in captured, (
        "C-ABI set_memory_margin_bytes must reject (warn) on XPU rather than "
        f"silently storing a margin malloc ignores; captured stderr: {captured!r}"
    )

    with torch_memory_saver.region(tag="margin_probe"):
        after_margin = torch.full((4096,), 7.0, dtype=torch.float32, device=f"{device}:0")
    torch.xpu.synchronize()
    assert float(after_margin[0]) == 7.0, (
        "allocation after a raw C-ABI set_memory_margin_bytes must still succeed "
        "-- the margin must not be installed as a live OOM guard on XPU"
    )
    del after_margin

    got_error = False
    try:
        _ = torch_memory_saver.memory_margin_bytes
    except NotImplementedError:
        got_error = True
    assert got_error, "memory_margin_bytes getter should raise NotImplementedError"

    print("memory_margin_unsupported_xpu OK")


def run_disable_unsupported(hook_mode: str):
    """disable() is rejected (not process-killing) on XPU.

    disable()'s body must allocate outside the tms allocator; the CUDA way (nested
    default MemPool) is unvalidated on XPU, and reaching pool-scoped
    tms_torch_malloc there hits its exit(1) assert, so disable() raises instead.
    Guarded here because the training_engine test covering disable() is CUDA-only.
    """
    assert hook_mode == "torch", "disable_unsupported_xpu test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU rejection path"

    device = get_device()
    torch_memory_saver.hook_mode = hook_mode

    raised = False
    with torch_memory_saver.region(tag="t"):
        weight = torch.full((1024,), 1.0, dtype=torch.float32, device=f"{device}:0")
        try:
            with torch_memory_saver.disable():
                pass
        except NotImplementedError:
            raised = True

    assert raised, "disable() should raise NotImplementedError on XPU, not proceed"

    probe = torch.full((1024,), 2.0, dtype=torch.float32, device=f"{device}:0")
    assert float(probe[0]) == 2.0
    assert float(weight[0]) == 1.0
    print("disable_unsupported_xpu OK")


def _used_gib(mod, d):
    mod.synchronize(d)
    return get_and_print_gpu_memory(f"dev{d}", gpu_id=d) / 1024**3


def run_multi_device_sync(hook_mode: str):
    """pause()/resume() drain EXACTLY the devices the backend will unmap, incl.
    non-current ones: _xpu_affected_devices reads the same allocation map they
    iterate, so the drain set cannot drift (needs >=2 devices).

    Pre-fix pause() synced only the CURRENT device while unmapping the tag on every
    device, so a non-current device with an in-flight kernel could be unmapped
    mid-flight and hang (DEVICE_LOST). Hence: no manual per-device sync below.
    """
    assert hook_mode == "torch", "multi_device_sync_xpu requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU affected-devices drain path"

    device = get_device()
    alloc_bytes = 256 * 1024 * 1024 * 4  # 256M float32 = 1 GiB
    freed_gib = 0.8  # expected drop/restore, with slack for alignment/noise
    mod = torch.get_device_module()
    if mod.device_count() < 2:
        print(f"skip: need >=2 {device} devices, have {mod.device_count()}")
        return

    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()
    impl = torch_memory_saver._impl
    cdll = impl._binary_wrapper.cdll
    assert hasattr(cdll, "tms_xpu_affected_devices"), (
        "backend missing tms_xpu_affected_devices; rebuild the XPU extension"
    )

    d0, d1 = 0, 1

    mod.set_device(d0)
    with torch_memory_saver.region(tag="t"):
        a = torch.full((alloc_bytes // 4,), 1.0, dtype=torch.float32, device=f"{device}:{d0}")
    mod.set_device(d1)
    with torch_memory_saver.region(tag="t"):
        b = torch.full((alloc_bytes // 4,), 1.0, dtype=torch.float32, device=f"{device}:{d1}")
    with torch_memory_saver.region(tag="only1"):
        c = torch.full((1024,), 1.0, dtype=torch.float32, device=f"{device}:{d1}")

    mod.set_device(d0)
    aff_t = sorted(impl._xpu_affected_devices("t"))
    assert aff_t == [d0, d1], f"affected devices for 't' should be [0, 1], got {aff_t}"

    aff_only1 = sorted(impl._xpu_affected_devices("only1"))
    assert aff_only1 == [d1], f"affected devices for 'only1' should be [1], got {aff_only1}"

    aff_all = sorted(impl._xpu_affected_devices(None))
    assert aff_all == [d0, d1], f"affected devices for all should be [0, 1], got {aff_all}"

    alloc0, alloc1 = _used_gib(mod, d0), _used_gib(mod, d1)

    mod.set_device(d1)
    b.mul_(2.0)  # in-flight work on d1
    mod.set_device(d0)  # make the OTHER device current
    torch_memory_saver.pause("t")
    pause0, pause1 = _used_gib(mod, d0), _used_gib(mod, d1)

    torch_memory_saver.resume("t")
    a.fill_(3.0)
    b.fill_(3.0)
    resume0, resume1 = _used_gib(mod, d0), _used_gib(mod, d1)

    assert (alloc0 - pause0) > freed_gib, "device 0 not freed on pause"
    assert (alloc1 - pause1) > freed_gib, "device 1 (non-current) not freed on pause"
    assert (resume0 - pause0) > freed_gib, "device 0 not re-committed on resume"
    assert (resume1 - pause1) > freed_gib, "device 1 (non-current) not re-committed on resume"
    assert float(a[0]) == 3.0, "device 0 tensor unusable after resume"
    assert float(b[0]) == 3.0, "device 1 tensor unusable after resume"

    assert float(c[0]) == 1.0, "untouched tag 'only1' tensor should be intact"

    print("multi_device_sync_xpu OK")


_SCENARIOS = {
    "free_failure": run_free_failure,
    "cleanup_failure": run_cleanup_failure,
    "resume_failure": run_resume_failure,
    "memory_margin_unsupported": run_memory_margin_unsupported,
    "disable_unsupported": run_disable_unsupported,
    "multi_device_sync": run_multi_device_sync,
}


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "free_failure"
    hook = sys.argv[2] if len(sys.argv) > 2 else "torch"
    _SCENARIOS[scenario](hook_mode=hook)
