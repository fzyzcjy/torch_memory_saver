# Repository Guidelines

## Change scope

- Keep each pull request focused. Split unrelated behavior fixes, mechanical renames or moves, formatting, and tooling changes into separate pull requests. If feature work depends on a general prerequisite fix, add its regression test and land it first.
- Keep diffs minimal: do not change unrelated whitespace, indentation, names, or unaffected backend paths. Keep backend-specific fixes scoped to that backend and validate them there.

## Design and invariants

- Put genuinely shared behavior behind a small backend-neutral layer. Isolate unavoidable platform differences in cohesive backend files or adapters; do not duplicate whole implementations or setup branches.
- Extract repeated logic into the narrowest cohesive function, class, or module, and follow adjacent repository structure for configuration, file placement, and lifecycle handling.
- Preserve entry points used by known downstream consumers; check real call sites before removing or renaming an API. Do not expose internal controls or add compatibility machinery for hypothetical consumers.
- Keep region and configuration state thread-local. Scoped APIs must restore the exact prior tag and enable or backup flags on every exit path, including nested scopes.
- MemPool cache keys must include every allocation-affecting configuration value and the device, so allocations with different tags or backup modes cannot share a pool accidentally.
- Treat impossible internal states and invalid lifecycle calls as programmer errors: assert or fail fast instead of adding recoverable error plumbing. For explicitly recoverable runtime failures, propagate the error and preserve the promised recovery contract.
- Prefer direct data flow and standard lifecycle mechanisms. Remove redundant state, ad hoc cleanup registries, and intermediate error-message plumbing when they add no behavior.

## Tests and validation

- Every bug fix or semantic behavior change must add a focused regression test covering the reported failure and any newly promised recovery path. Verify that the test fails without the fix and passes with it.
- Assert observable data and complete state transitions, not only that an API returned successfully. For scoped state, check the relevant values before, during, and after the scope.
- Put new tests in the existing test tree and make them runnable through the repository's standard test convention. Write shared scenarios to run on every backend and hook mode that supports the behavior without weakening existing coverage.
- Before merging, run the full regression suite and the relevant affected-backend tests, then paste the complete passing results in the pull request. Include before-and-after evidence for fixes when practical.

## Comments and documentation

- Prefer self-explanatory code. Remove comments that restate the code and delete commented-out code; keep concise comments for non-obvious constraints or examples explicitly written as tutorials.
- Keep README updates brief and avoid bold emphasis. When a user workflow changes, document the simplest supported path and preserve guidance for known downstream integrations.
