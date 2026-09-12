---
name: tms-testing
description: Use when running the full torch_memory_saver test suite, including build tooling and the CUDA wheel runtime matrix.
---

# 1 Human SoT

# 2 Full suite

- Run from the repository root on `tom-workstation` with an idle GPU, Docker, and ARM64 binfmt available.
- Use [tms-publish-release](../tms-publish-release/SKILL.md) Sections 2–5 for isolated source/artifact paths, environment setup, both wheel builds, and the four-cell GPU harness. For development testing, use the target commit and its `setup.py` version; release-only requirements to use merged `origin/master`, check PyPI, or publish do not apply.
- Run the build-tool tests separately from the GPU runtime suite:

```bash
uv run --no-project --with pytest --with packaging --with typer --with setuptools python -m pytest \
  test/test_merge_cuda_wheels.py \
  .claude/skills/tms-publish-release/scripts/test_*.py -vv -ra
```

- After building both wheels into `dist/`, run the complete runtime matrix with the paths prepared above:

```bash
uv run --script .claude/skills/tms-publish-release/scripts/gpu_validation.py \
  --release-root "$TMS_REMOTE_ROOT" \
  --artifact-root "$TMS_REMOTE_ARTIFACTS" \
  --expected-version "$TMS_RELEASE_VERSION"
```

- Require passing build-tool tests and all four runtime cells: x86_64/ARM64 × CUDA 12/13. Preserve complete logs and report each summary and skip reason.

# 3 Reliability

- Run long commands in the background with output saved under the artifact directory; inspect progress at least every ten minutes.
- The harness tests installed wheels outside the checkout and enforces exact skips. Do not replace it with a source-tree pytest run or deselect failing cases.
- This is the supported single-GPU CUDA matrix; its multi-device, XPU, and Lupine exclusions are not coverage of those paths. Validate affected backends on suitable hardware separately.
- When adding `test/test_*.py`, update the harness runtime/build-tool inventory and run every classified build-tool module; unclassified modules fail the harness preflight.
