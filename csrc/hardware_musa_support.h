#pragma once

#include <cstddef>

namespace MUSAImplementation {
    // MCCL and muDNN own internal buffers that cannot live in MUSA VMM memory:
    // with TMS-backed (muMemCreate/muMemMap) buffers MCCL either aborts with an
    // unhandled musa error or makes later muDNN kernel launches fail with
    // "invalid argument". Keep those library-internal allocations on the normal
    // runtime allocator, so only PyTorch caching-allocator segments are tracked.
    bool caller_requires_plain_malloc(void* caller);

    // Zero a resumed region on the allocation's device and wait for the zeroing
    // to land, so the caller can hand the region back to PyTorch.
    void zero_region(void* ptr, size_t size, int device);
}
