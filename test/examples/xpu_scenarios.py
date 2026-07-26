"""XPU-specific torch_memory_saver scenarios (one module, one function each).

Each run_* below is an independent scenario invoked in its own spawned
subprocess by test/test_examples.py, so a fault-injection abort or exit(1) in
one cannot corrupt the others. Every scenario keeps the docstring documenting
the regression it guards. Runnable standalone:

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


# --- Shared saver-metric readers (driver-independent; see testing_utils) ---
def _committed_bytes(gpu_id=0):
    return int(torch_memory_saver._impl._binary_wrapper.cdll.tms_xpu_committed_bytes(gpu_id))


def _leaked_bytes(gpu_id=0):
    return int(torch_memory_saver._impl._binary_wrapper.cdll.tms_xpu_leaked_bytes(gpu_id))


def _tracked_bytes(gpu_id=0):
    return int(torch_memory_saver._impl._binary_wrapper.cdll.tms_xpu_tracked_bytes(gpu_id))


def run_free_failure(hook_mode: str):
    """free() retains ownership until every Level Zero release step succeeds (XPU).

    Regression target: xpu_free() used to erase the metadata entry BEFORE any
    Level Zero cleanup, ignore every unmap/destroy/free return value, and destroy
    a still-mapped physical handle after a failed unmap. Any failure then
    permanently leaked physical memory or VA with no way to retry, because the
    only ownership record was already gone.

    The fix makes free transactional, mirroring pause/resume: the entry is
    retained and its state advanced per step, each Level Zero return code is
    checked, and the record is erased only after unmap -> destroy -> free-VA all
    confirm. A later free retry resumes from exactly where the previous stopped.

    Drives each free step's failure via TMS_XPU_FAULT_FREE_* and after each
    asserts the record is STILL tracked and a retry advances one step, until a
    clean free releases everything. The allocation is a raw VMM region from
    tms_torch_malloc (no torch tensor co-owns it), freed via the test-only
    tms_xpu_free -- the normal free hook wraps free() in a process-fatal
    CUDA_ERROR_CHECK, so a free-step failure cannot be observed through it.

    The free-VA-failure state is the crux: retained PAUSED, not leaked, no
    physical handle, so BOTH committed (ACTIVE-only) and leaked (leaked-only) read
    0 -- the exact state the original bug made invisible. Only tracked_bytes proves
    the record survived.
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

    # tms_torch_malloc is the torch allocator callback; call it directly (not via
    # torch) to get a raw VMM allocation that ONLY tms tracks -- so no torch
    # tensor will also try to free it, and we can drive tms_xpu_free ourselves.
    cdll.tms_torch_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    cdll.tms_torch_malloc.restype = ctypes.c_void_p

    # Allocate inside an "interesting region" (backup off: this finding is about
    # the L0 release steps, not the CPU backup).
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

    # --- Step 1 fails: unmap. Entry stays ACTIVE (physical still live+mapped);
    # free must NOT proceed to destroy a still-mapped handle. ---
    os.environ["TMS_XPU_FAULT_FREE_UNMAP"] = "1"
    try:
        _free(expect_ok=False)
        assert _tracked_bytes() == full, "unmap failure must RETAIN the ownership record"
        assert _committed_bytes() == full, "unmap failure leaves the allocation ACTIVE"
        assert _leaked_bytes() == 0, "no handle orphaned yet (unmap never happened)"
    finally:
        del os.environ["TMS_XPU_FAULT_FREE_UNMAP"]

    # --- Retry advances: unmap now succeeds (-> PAUSED, handle orphaned), but
    # Step 2 destroy fails. Handle retained + surfaced as leaked, entry kept. ---
    os.environ["TMS_XPU_FAULT_FREE_DESTROY"] = "1"
    try:
        _free(expect_ok=False)
        assert _tracked_bytes() == full, "destroy failure must RETAIN the ownership record"
        assert _committed_bytes() == 0, "VA unmapped -> not ACTIVE (committed 0)"
        assert _leaked_bytes() == full, "undestroyed physical handle must show as leaked"
    finally:
        del os.environ["TMS_XPU_FAULT_FREE_DESTROY"]

    # --- Retry advances: destroy now succeeds (handle gone, leak cleared), but
    # Step 3 free-VA fails. THE CRUX: committed AND leaked are both 0, yet the
    # record must still be tracked -- the exact state the old bug made invisible. ---
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

    # --- Clean retry: free-VA now succeeds; entry erased exactly once. ---
    _free(expect_ok=True)
    assert _tracked_bytes() == 0, "successful free must release and erase the record"
    assert _committed_bytes() == 0 and _leaked_bytes() == 0

    # Freeing an already-freed pointer must report the missing entry (not crash /
    # not double-erase), proving the record was erased exactly once above.
    assert int(cdll.tms_xpu_free(ptr)) != 0, "double free must report invalid pointer"

    print("free_failure_injection_xpu OK")


def run_cleanup_failure(hook_mode: str):
    """Ownership is retained until Level Zero confirms release (XPU).

    The XPU backend must not drop ownership of a resource before the driver
    confirms it was freed. Two cleanup-failure paths are exercised here:

    1. pause() destroy failure -- when zePhysicalMemDestroy fails, the physical
       handle is STILL alive (the VA is merely unmapped). The old code cleared the
       handle and force-marked the allocation PAUSED, orphaning the physical
       allocation so the next resume created a second handle and leaked the first.
       The fix retains the handle (marked leaked), surfaces the failure to Python,
       and re-maps that exact handle on the next resume.

    2. free() -- metadata ownership is dropped only after unmap/destroy/free-VA
       all succeed (previously the entry was erased first and codes ignored).

    Crucially the leak is asserted via a SEPARATE metric, tms_xpu_leaked_bytes,
    not the ACTIVE-only tms_xpu_committed_bytes: a destroy failure leaves committed
    bytes at 0 (unmapped) while the retained physical handle shows up as leaked
    bytes. Trusting only the committed counter would report 0 and hide the leak.
    """
    assert hook_mode == "torch", "cleanup failure-injection test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU cleanup-ownership path"

    device = get_device()  # "xpu"
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

    # enable_cpu_backup=True so the retained handle is later re-mapped AND the
    # data restored, proving the recovery path is truly end-to-end.
    with torch_memory_saver.region(tag="t", enable_cpu_backup=True):
        x = torch.full((4_000_000,), magic, dtype=torch.uint8, device=f"{device}:0")
    torch.xpu.synchronize()
    committed_active = _committed_bytes()
    assert committed_active > 0, "allocation should be committed while ACTIVE"
    assert _leaked_bytes() == 0, "nothing leaked before any failure"

    # --- pause() destroy-failure: handle must be retained + leak made visible ---
    os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"] = "1"
    try:
        raised = False
        try:
            torch_memory_saver.pause("t")
        except RuntimeError:
            raised = True
        assert raised, "pause() must raise when a physical handle cannot be released"

        # The VA was unmapped, so the ACTIVE-only committed counter reads 0 --
        # which is exactly why it must NOT be the sole signal. The retained
        # physical handle is instead visible as leaked bytes: ownership was kept,
        # not silently dropped.
        assert _committed_bytes() == 0, (
            "destroy-failure leaves the VA unmapped (0 committed)"
        )
        assert _leaked_bytes() == committed_active, (
            "retained physical handle must be tracked as leaked, not dropped "
            "(this is the leak the ACTIVE-only counter hides)"
        )
    finally:
        del os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"]

    # resume() must re-map the SAME retained handle (not allocate a second one).
    # After it, the leak is reclaimed: committed back to full, leaked back to 0,
    # data intact -- proving no orphaned handle was left behind.
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

    # --- clean pause/resume still works after recovery (no lingering leaked state) ---
    torch_memory_saver.pause("t")
    assert _committed_bytes() == 0 and _leaked_bytes() == 0, (
        "a clean pause after recovery releases the handle with no leak"
    )
    torch_memory_saver.resume("t")
    assert _committed_bytes() == committed_active and _leaked_bytes() == 0
    assert x[:3].tolist() == [magic, magic, magic]

    # --- free() reclaims a RETAINED (leaked) handle rather than skipping it ---
    # Drive the allocation back into the PAUSED+leaked state (destroy fails) and
    # this time free it WITHOUT resuming. free() must notice the handle is still
    # alive (leaked=true), destroy it, and release the VA -- proving ownership was
    # retained across the pause failure and free picks up exactly where pause
    # stopped, instead of leaking the physical handle forever.
    os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"] = "1"
    try:
        try:
            torch_memory_saver.pause("t")
        except RuntimeError:
            pass
        assert _leaked_bytes() == committed_active, "handle should be retained as leaked"
    finally:
        del os.environ["TMS_XPU_FAULT_PAUSE_DESTROY"]

    # Torch's MemPool caches freed blocks, so del + empty_cache alone does NOT
    # invoke the free callback; clearing the pool returns the block and triggers
    # tms free. (This is also why xpu_free must be correct on the leaked path.)
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

    The XPU backend re-creates a physical handle, maps it, and restores the CPU
    backup on resume(). If any step fails it must roll back to a clean PAUSED
    state (backup preserved) and surface the error to Python -- never report
    success with a tensor left unmapped or with undefined contents.

    Arms each failure via TMS_XPU_FAULT_RESUME_{CREATE,MAP,RESTORE}, asserts
    resume() raises, asserts nothing was committed (still unmapped), then clears
    the fault and asserts a retry fully resumes with the original data intact --
    proving the backup survived and resume() is idempotent.
    """
    assert hook_mode == "torch", "resume failure-injection test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU transactional resume path"

    device = get_device()  # "xpu"
    magic = 123  # sentinel value stored in the pauseable tensor
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

    # enable_cpu_backup=True so the restore step (and its rollback) is exercised.
    with torch_memory_saver.region(tag="t", enable_cpu_backup=True):
        x = torch.full((4_000_000,), magic, dtype=torch.uint8, device=f"{device}:0")
    torch.xpu.synchronize()
    committed_active = _committed_bytes()
    assert committed_active > 0, "allocation should be committed while ACTIVE"

    for fault in faults:
        torch_memory_saver.pause("t")
        assert _committed_bytes() == 0, f"{fault}: pause should release physical pages"

        # Arm the fault and confirm resume() raises rather than half-resuming.
        os.environ[fault] = "1"
        try:
            raised = False
            try:
                torch_memory_saver.resume("t")
            except RuntimeError:
                raised = True
            assert raised, f"{fault}: resume() must raise on injected failure"
            # Rolled back: still unmapped (0 committed), so the tensor was NOT
            # left ACTIVE-but-unmapped nor mapped-but-uninitialized.
            assert _committed_bytes() == 0, (
                f"{fault}: failed resume must leave the allocation PAUSED"
            )
        finally:
            del os.environ[fault]

        # Backup survived + resume is idempotent: retry succeeds with data intact.
        torch_memory_saver.resume("t")
        assert _committed_bytes() == committed_active, (
            f"{fault}: retry after clearing fault should fully re-commit"
        )
        assert x[:3].tolist() == [magic, magic, magic], (
            f"{fault}: data must be restored intact after retry"
        )

    print("resume_failure_injection_xpu OK")


def _reported_free_bytes(gpu_id=0):
    """Best available device free-bytes reading (what a margin check would use).

    Prefers the saver's sysman-based reading (tms_xpu_device_free_bytes) and
    falls back to torch.xpu.mem_get_info; both are frozen on current drivers.
    """
    cdll = torch_memory_saver._impl._binary_wrapper.cdll
    if hasattr(cdll, "tms_xpu_device_free_bytes"):
        return int(cdll.tms_xpu_device_free_bytes(gpu_id))
    free, _total = torch.xpu.mem_get_info(gpu_id)
    return int(free)


def run_memory_margin_unsupported(hook_mode: str):
    """memory_margin_bytes is rejected (not silently ignored) on XPU.

    memory_margin_bytes is an OOM guard: on CUDA, malloc rejects an allocation
    when margin + size would leave less than `value` device bytes free (core.cpp).
    That depends on an accurate device-wide free-bytes reading, which the Intel
    GPU stack does not provide -- torch.xpu.mem_get_info().free and sysman
    zesMemoryGetState().free are frozen on current drivers. The XPU malloc path
    never applied the margin, so setting it was silently ignored and a caller
    relying on the safety margin could exhaust the device. The setter now raises.

    This test (a) proves the Python setter rejects, (b) documents WHY by showing
    the telemetry is frozen -- allocating memory does not reduce reported free
    bytes -- and (c) proves the C-ABI set_memory_margin_bytes (bypassing the
    Python guard) does NOT install a live margin: it refuses a non-zero value on
    XPU, so malloc never OOM-rejects against a phantom margin. (c) is the
    defense-in-depth path making the early XPU return in malloc() safe.
    """
    assert hook_mode == "torch", "memory_margin_unsupported_xpu test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU rejection path"

    device = get_device()  # "xpu"
    torch_memory_saver.hook_mode = hook_mode
    torch_memory_saver._ensure_initialized()

    # (a) The setter must reject explicitly rather than silently no-op.
    raised = False
    try:
        torch_memory_saver.memory_margin_bytes = 1 << 30  # 1 GiB
    except NotImplementedError:
        raised = True
    assert raised, "memory_margin_bytes should raise NotImplementedError on XPU"

    # (b) Document why: the device-wide free-bytes telemetry a margin check would
    # rely on is frozen -- committing real memory does not move it, so the guard
    # could never fire. (~1.5 GiB is well above any measurement noise.)
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

    # The process must still be usable (rejection, not a killed process).
    del big
    probe = torch.full((1024,), 3.0, dtype=torch.float32, device=f"{device}:0")
    assert float(probe[0]) == 3.0

    # (c) Defense-in-depth on the C ABI. A caller can invoke the C symbol
    # set_memory_margin_bytes directly, bypassing the Python guard in (a). On XPU
    # that must NOT quietly store a margin the malloc path then ignores (which is
    # exactly what would give a false sense of OOM safety): it must refuse and
    # warn. The refusal is observable only as a stderr warning (there is no getter
    # to read the stored value back, and the XPU malloc branch never reads the
    # margin regardless), so capture the C library's stderr (fd 2) around the
    # call and assert the rejection fired. An absurd 1 EiB value makes the intent
    # unambiguous: on CUDA this would install a margin larger than any device,
    # OOM-rejecting every allocation; on XPU it is refused.
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

    # And the process must still allocate normally afterwards -- the refused
    # margin must not have been installed as a live OOM guard.
    with torch_memory_saver.region(tag="margin_probe"):
        after_margin = torch.full((4096,), 7.0, dtype=torch.float32, device=f"{device}:0")
    torch.xpu.synchronize()
    assert float(after_margin[0]) == 7.0, (
        "allocation after a raw C-ABI set_memory_margin_bytes must still succeed "
        "-- the margin must not be installed as a live OOM guard on XPU"
    )
    del after_margin

    # Getter is unsupported on every platform (only the setter exists upstream).
    got_error = False
    try:
        _ = torch_memory_saver.memory_margin_bytes
    except NotImplementedError:
        got_error = True
    assert got_error, "memory_margin_bytes getter should raise NotImplementedError"

    print("memory_margin_unsupported_xpu OK")


def run_disable_unsupported(hook_mode: str):
    """disable() is rejected (not process-killing) on XPU.

    XPU uses the pool-scoped torch hook allocator, so tms is only "active" inside
    a region()'s use_mem_pool. disable() must let its body allocate WITHOUT the
    tms allocator, which the CUDA path does by entering a fresh default MemPool;
    the equivalent nested use_mem_pool is not yet validated on XPU. Rather than
    leave a path where an allocation in the disabled body routes to the
    pool-scoped tms_torch_malloc and hits its exit(1) assert, disable() raises
    NotImplementedError. This guards that (and that the process survives), since
    the CUDA-only training_engine test that exercises disable() is skipped on XPU.
    """
    assert hook_mode == "torch", "disable_unsupported_xpu test requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU rejection path"

    device = get_device()  # "xpu"
    torch_memory_saver.hook_mode = hook_mode

    # disable() asserts tms is active, so exercise it from inside a region().
    raised = False
    with torch_memory_saver.region(tag="t"):
        weight = torch.full((1024,), 1.0, dtype=torch.float32, device=f"{device}:0")
        try:
            with torch_memory_saver.disable():
                # Must never reach here on XPU; if we did, an allocation would
                # route to the pool-scoped allocator and exit(1) the process.
                pass
        except NotImplementedError:
            raised = True

    assert raised, "disable() should raise NotImplementedError on XPU, not proceed"

    # The process must still be usable (rejection, not a killed process).
    probe = torch.full((1024,), 2.0, dtype=torch.float32, device=f"{device}:0")
    assert float(probe[0]) == 2.0
    assert float(weight[0]) == 1.0
    print("disable_unsupported_xpu OK")


def _used_gib(mod, d):
    mod.synchronize(d)
    return get_and_print_gpu_memory(f"dev{d}", gpu_id=d) / 1024**3


def run_multi_device_sync(hook_mode: str):
    """pause()/resume() drain EXACTLY the devices the backend will unmap, incl.
    non-current ones -- authoritatively, from the backend allocation map.

    Regression target: pause() used to synchronize only the CURRENT device, while
    xpu_pause() unmaps every allocation matching the tag on EVERY device. A
    non-current device with an in-flight kernel could then be unmapped mid-flight
    and hang the device (DEVICE_LOST). The fix drains the affected devices from
    tms_xpu_affected_devices, which reads the same allocation map (under the same
    lock) pause()/resume() iterate -- so the drain set cannot drift from what is
    actually unmapped.

    This test: (a) allocates on device 0 and 1, sets 0 current, calls
    pause()/resume() with NO manual per-device sync -- non-current device 1 must
    still be drained; (b) asserts _affected_devices(tag) == the real allocation
    devices {0, 1}; (c) asserts tag scoping (a tag only on device 1 reports {1});
    (d) leaves an in-flight kernel queued on the non-current device before pause.

    Requires >=2 XPU devices; skips otherwise.
    """
    assert hook_mode == "torch", "multi_device_sync_xpu requires hook_mode='torch'"
    assert is_xpu(), "this test targets the XPU affected-devices drain path"

    device = get_device()  # "xpu"
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

    # Allocate on BOTH devices under tag "t"; only d1 also gets tag "only1".
    mod.set_device(d0)
    with torch_memory_saver.region(tag="t"):
        a = torch.full((alloc_bytes // 4,), 1.0, dtype=torch.float32, device=f"{device}:{d0}")
    mod.set_device(d1)
    with torch_memory_saver.region(tag="t"):
        b = torch.full((alloc_bytes // 4,), 1.0, dtype=torch.float32, device=f"{device}:{d1}")
    with torch_memory_saver.region(tag="only1"):
        c = torch.full((1024,), 1.0, dtype=torch.float32, device=f"{device}:{d1}")

    # (b) The affected-devices set is authoritative: both real allocation devices
    # for tag "t", regardless of which device is current.
    mod.set_device(d0)
    aff_t = sorted(impl._affected_devices("t"))
    assert aff_t == [d0, d1], f"affected devices for 't' should be [0, 1], got {aff_t}"

    # (c) Tag scoping: "only1" lives solely on device 1.
    aff_only1 = sorted(impl._affected_devices("only1"))
    assert aff_only1 == [d1], f"affected devices for 'only1' should be [1], got {aff_only1}"

    # None-tag (all) covers every device that has any allocation.
    aff_all = sorted(impl._affected_devices(None))
    assert aff_all == [d0, d1], f"affected devices for all should be [0, 1], got {aff_all}"

    alloc0, alloc1 = _used_gib(mod, d0), _used_gib(mod, d1)

    # (d) Queue an in-flight kernel on the NON-current device (d1) right before
    # pause, with d0 current. pause() must drain d1 (via the backend set) before
    # unmapping its pages. No manual per-device sync here on purpose.
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

    # tag "only1" was untouched by pause("t") -- still ACTIVE and usable.
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
