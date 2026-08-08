# Repository Rules

- Keep PRs focused and diffs minimal. Land prerequisite fixes separately; split unrelated fixes, refactors, formatting, and tooling. Do not touch unaffected backends.
- Share logic through small backend-neutral interfaces; isolate platform differences in cohesive adapters. Do not duplicate whole implementations or setup branches. Validate backend fixes on the affected backend.
- Follow existing patterns. Preserve APIs used by known consumers; do not expose internal controls or add speculative compatibility.
- Keep region and configuration state thread-local. Restore the complete previous tag, enable, and backup state on every exit, including nested scopes. Key MemPools by every allocation-affecting setting and device.
- Fail fast on impossible states and lifecycle misuse. Propagate recoverable failures without breaking retry guarantees. Prefer direct data flow; remove redundant state and cleanup/error plumbing.
- Every behavior change needs a focused regression test: fail before, pass after, and assert observable data, state, and recovery.
- Use the existing test tree and runner; preserve coverage across every supported backend and hook mode. Before merge, run the full regression suite and affected-backend tests, then paste complete results in the PR.
- Prefer self-explanatory code. Remove narrative comments and commented-out code; comment only non-obvious constraints or tutorials.
- Keep README changes brief and avoid bold emphasis. Document the simplest supported path and known integrations.
