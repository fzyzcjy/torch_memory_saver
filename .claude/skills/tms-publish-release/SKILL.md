---
name: tms-publish-release
description: Use when preparing, building, validating, publishing, or verifying torch_memory_saver beta and stable releases across x86_64, aarch64, CUDA 12, and CUDA 13.
---

# 1 Release contract

- Work from the repository root and publish only a clean commit already merged into the latest `origin/master`.
- Use canonical PEP 440 versions in `setup.py`:
    - Beta: `0.0.10b1`, not `0.0.10.beta-1`, `0.0.10-beta.1`, or `0.0.10beta1`.
    - Release candidate: `0.0.10rc1`.
    - Stable: `0.0.10`.
    - Post release: `0.0.10.post1` only for a source-equivalent packaging correction.
- Publish exactly three distributions:
    - `cp39-abi3-manylinux2014_x86_64` wheel.
    - `cp39-abi3-manylinux2014_aarch64` wheel.
    - Source distribution.
- Build and test on `tom-workstation`.
- Publish from the local controller.
- Never persist PyPI credentials on the GPU host.
- Never create a GitHub Release. Publish only the PyPI distributions and an annotated Git tag.
- Treat PyPI upload and Git tag push as irreversible.
- Complete the Section 8 confirmation gate before either action.

# 2 Prepare isolated paths

```bash
set -euxo pipefail
export TMS_RELEASE_VERSION="<VERSION>"
export TMS_RELEASE_SHA="$(git rev-parse --short=12 HEAD)"
export TMS_RELEASE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export TMS_REMOTE_ROOT="/home/tom/workspace/torch_memory_saver_release_${TMS_RELEASE_VERSION}_${TMS_RELEASE_SHA}_${TMS_RELEASE_RUN_ID}"
export TMS_REMOTE_ARTIFACTS="${TMS_REMOTE_ROOT}_artifacts"
export TMS_LOCAL_ARTIFACTS="/Users/tom/domains/human/others/artifacts/torch_memory_saver/${TMS_RELEASE_VERSION}-${TMS_RELEASE_SHA}-${TMS_RELEASE_RUN_ID}-release"
export TMS_RELEASE_CHECKS="$PWD/.claude/skills/tms-publish-release/scripts/release_checks.py"
mkdir "$TMS_LOCAL_ARTIFACTS"
```

- Never reuse a local or remote source or artifact directory, even for the same version and commit.
- Keep every build, validation, environment, and recheck log under the unique artifact directories.

# 3 Preflight

## 3.1 Repository and version

- Append every command, complete output, and exit result to `release-preflight.log`:

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- git fetch origin master
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- git status --short --branch
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- git rev-parse HEAD
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- git rev-parse origin/master
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- git diff --stat origin/master...HEAD
```

- Stop if the tree is dirty, `HEAD` differs from `origin/master`, or the release commit is detached.
- Read and validate the version without importing `setup.py`:

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  uv run --script "$TMS_RELEASE_CHECKS" version \
  --setup-py setup.py \
  --expected-version "$TMS_RELEASE_VERSION"
```

- Confirm the target version does not already exist on PyPI:

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  uv run --script "$TMS_RELEASE_CHECKS" pypi \
  --expected-version "$TMS_RELEASE_VERSION"
```

- Review every change since the previous release commit.
- Use `c29087a58db9d120b3e69623714c5dd043029d77` as the historical source baseline for the first tagged release after `0.0.9.post1`.
- Use the previous release tag after the first tagged release.

## 3.2 Host and credential checks

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  ssh tom-workstation 'set -euxo pipefail; curl --retry 5 --retry-delay 1 --retry-all-errors -x http://127.0.0.1:7890 -I --max-time 20 https://registry-1.docker.io/v2/'
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  ssh tom-workstation 'set -euxo pipefail; test -x /home/tom/.local/bin/uv; nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader; docker version; df -h / /home/tom'
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  /usr/bin/stat -f '%N mode=%Lp size=%z' "$HOME/.pypirc"
```

- Expect HTTP `401` from the Docker Hub registry probe; it proves official registry reachability.
- Stop if GPU 0 is busy, Docker is unavailable, or the host lacks enough space for the PyTorch builder images.
- Require local `$HOME/.pypirc` mode `0600`.
- Never print `$HOME/.pypirc`.
- Never copy it to `tom-workstation`.
- Never mount it into a long-lived container.

## 3.3 ARM64 emulation on the x86_64 host

- The official `manylinuxaarch64-builder` images are Linux ARM64 images, not x86 cross-compiler images.
- Check for an existing ARM64 binfmt handler:

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  ssh tom-workstation 'set -euxo pipefail; test -r /proc/sys/fs/binfmt_misc/qemu-aarch64; cat /proc/sys/fs/binfmt_misc/qemu-aarch64'
```

- If it is missing, stop for approval before installing QEMU binfmt.
- Treat installation as privileged and host-wide.
- Use the pinned `tonistiigi/binfmt` image.
- After approval, install only ARM64 and verify it with an ARM64 Alpine container:

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  ssh tom-workstation 'set -euxo pipefail
docker run --privileged --rm tonistiigi/binfmt@sha256:400a4873b838d1b89194d982c45e5fb3cda4593fbfd7e08a02e76b03b21166f0 --install arm64
cat /proc/sys/fs/binfmt_misc/qemu-aarch64
docker run --rm --platform linux/arm64 alpine:3.22 uname -m'
```

- Require the smoke test to print `aarch64`.
- Treat binfmt as boot-scoped unless the host has an explicit persistent registration. Recheck this section after every workstation reboot.

## 3.4 Release script roles

| Script | Responsibility | Mutates release state |
| --- | --- | --- |
| `.claude/skills/tms-publish-release/scripts/release_checks.py` | Version, namespace, artifact, manifest, one-command pre-upload, and append-only command-log gates | No |
| `.claude/skills/tms-publish-release/scripts/gpu_validation.py` | Four-cell fresh-container GPU runtime harness with exact skip enforcement | Creates only temporary validation resources |
| `.claude/skills/tms-publish-release/scripts/pytest_skip_gate.py` | Exact pytest skipped-node and reason validator loaded by the harness | No |
| `.claude/skills/tms-publish-release/scripts/verify_published_release.sh` | Auditable post-release PyPI, CUDA 12, and CUDA 13 verification | Creates only temporary validation containers |

- Keep upload and tag creation outside every release script.
- Do not create, query, attach artifacts to, or require a GitHub Release.
- Use `sdist` only for focused diagnostics.
- Use `artifacts` for the canonical release workflow after all three distributions exist.

## 3.5 Create and synchronize the remote source tree

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  ssh tom-workstation "set -euxo pipefail; test ! -e '$TMS_REMOTE_ROOT'; test ! -e '$TMS_REMOTE_ARTIFACTS'; mkdir '$TMS_REMOTE_ROOT' '$TMS_REMOTE_ARTIFACTS'"
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  rsync -a \
  --exclude .git \
  --exclude build \
  --exclude dist \
  --exclude torch_memory_saver.egg-info \
  --exclude .pytest_cache \
  --exclude '*.so' \
  ./ "tom-workstation:$TMS_REMOTE_ROOT/"
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- shasum -a 256 setup.py
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/release-preflight.log" -- \
  ssh tom-workstation "set -euxo pipefail; sha256sum '$TMS_REMOTE_ROOT/setup.py'"
```

- Do not use `rsync --delete`.
- Require the local and remote `setup.py` hashes to match.
- The fail-if-present remote `mkdir` makes every rehearsal independent.

# 4 Build release distributions

## 4.1 x86_64 wheel

- Build with the official PyTorch CUDA 12.8 and CUDA 13.0 manylinux images.

```bash
set -euxo pipefail
ssh tom-workstation "set -euxo pipefail; cd '$TMS_REMOTE_ROOT' \
  && make clean \
  && TMS_PYTHON_BUILD_IMAGE=python:3.11 make build-wheel-multi-cuda \
  > '$TMS_REMOTE_ARTIFACTS/build-x86_64.log' 2>&1"
```

- Run long commands in a background exec session and babysit with:

```bash
set -euxo pipefail
ssh tom-workstation "set -euxo pipefail; tail -80 '$TMS_REMOTE_ARTIFACTS/build-x86_64.log'"
```

## 4.2 aarch64 wheel on the x86_64 host

- Build with the official ARM64 PyTorch CUDA 12.8 and CUDA 13.0 manylinux images.

```bash
set -euxo pipefail
ssh tom-workstation "set -euxo pipefail; cd '$TMS_REMOTE_ROOT' \
  && rm -rf build torch_memory_saver.egg-info \
  && DOCKER_DEFAULT_PLATFORM=linux/arm64 \
    TMS_PYTHON_BUILD_IMAGE=python:3.11 \
    make build-wheel-multi-cuda-aarch64 \
  > '$TMS_REMOTE_ARTIFACTS/build-aarch64.log' 2>&1"
```

## 4.3 Source distribution and artifact gate

```bash
set -euxo pipefail
ssh tom-workstation "set -euxo pipefail; cd '$TMS_REMOTE_ROOT' \
  && TMS_PYTHON_BUILD_IMAGE=python:3.11 make build-sdist \
  > '$TMS_REMOTE_ARTIFACTS/build-sdist.log' 2>&1 \
  && HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
    /home/tom/.local/bin/uv run --script .claude/skills/tms-publish-release/scripts/release_checks.py artifacts \
    --dist-dir dist \
    --expected-version '$TMS_RELEASE_VERSION' \
    --repo-root . \
  > '$TMS_REMOTE_ARTIFACTS/validate-artifacts.log' 2>&1"
```

```text
Input: $TMS_REMOTE_ROOT/dist
Exact set: x86_64 wheel, aarch64 wheel, sdist
Wheel gate:
  - Filename and internal metadata tags
  - ELF machine for every binary
  - CUDA 12 and CUDA 13 binary families
  - Unsuffixed CUDA 12 compatibility binaries
  - RECORD integrity
Sdist gate: canonical version, metadata, complete native sources
Ownership: dist, build, and torch_memory_saver.egg-info belong to the SSH user
Failure policy: stop on any mismatch or extra artifact
```

# 5 Validate both wheels in fresh GPU containers

## 5.1 Harness specification

| Case | Runtime path |
| --- | --- |
| x86_64 CUDA 12 | PyTorch CUDA 12 container, direct NVIDIA runtime |
| x86_64 CUDA 13 | PyTorch CUDA 13 container, direct NVIDIA runtime |
| ARM64 CUDA 12 | Official Lupine 12.8.1 PyTorch worker under QEMU to matched x86 server |
| ARM64 CUDA 13 | Official Lupine 13.0.2 PyTorch worker under QEMU to matched x86 server |

### 5.1.1 Common contract

```text
Host: x86_64 tom-workstation
Inputs: final x86_64 wheel, final aarch64 wheel, runtime tests only
Image acquisition: docker run --pull missing
Source mount: release root at /workspace, read-only
Install root: final wheel into site-packages or dist-packages
Test root: /validation, outside the source checkout
x86_64 test suite: full single-GPU runtime pytest
ARM64 test suite: full runtime pytest with in-test Lupine skip markers
Lupine signal: TMS_TEST_LUPINE=1 in ARM64 validation containers only
Lupine skip 1: test_cpu_backup_preload_backend_from_env
Lupine skip 2: test_disk_backup, including both hook-mode parameters
CUDA major: matches the matrix cell
CUDA availability: torch.cuda.is_available() is true
GPU identity: NVIDIA GeForce RTX 4090 D
CUDA runtime: matching packaged libcudart added before preload children start
Process architecture: matches the matrix cell
Binary architecture: every loaded ELF machine matches the matrix cell
Inventory gate: every repository test_*.py classified as runtime or build tooling
Build-tool UT: build phase only
Success: zero failures in all four cells
Allowed skips: documented Lupine host-memory cases, single-GPU multi-device cases, XPU-only cases
Skip gate: exact node ID and normalized reason map passed through TMS_EXPECTED_PYTEST_SKIPS
Skip result: SKIP_GATE_ACTUAL and SKIP_GATE_RESULT recorded in the cell log
CLI deselection: forbidden
Missing, additional, duplicated, or reason-changed skip: cell failure
Evidence: image identity, environment, commands, installed paths, binaries, pytest output
Command log: EXEC line, complete stdout/stderr, terminal RESULT line
Container shell trace: set -euxo pipefail records each in-container command
Cleanup: named clients, servers, containers, and private networks removed on success or failure
```

### 5.1.2 Lupine contract

```text
Matrix source: .claude/skills/tms-publish-release/scripts/gpu_validation.py
Image identity: architecture-specific worker/server refs pinned by digest
Worker: official ARM64 lupine-pytorch-worker
Server: official x86_64 lupine-server with GPU 0 attached
Source revision: ebf4c2784e5756891ed8c2439fec37ed1a4e6b51 for worker and server
Network: private per-cell bridge; RPC port not published to the workstation LAN
Proxy: host.docker.internal:host-gateway -> workstation Clash
Preinstalled: ARM Python, PyTorch, CUDA user-space libraries, CUDA/NVML shims
Added test dependencies: pinned binary NumPy, pytest, nvidia-ml-py wheels
Execution: real ARM64 Python, PyTorch, and TMS wheel under QEMU
GPU path: CUDA driver and NVML calls forwarded to the physical 4090D
Test signal: TMS_TEST_LUPINE=1
Qualification: ARM user-space GPU behavior through Lupine
Not qualified: ARM NVIDIA kernel driver or native ARM PCIe path
Disk-backup boundary: Lupine protects pinned host mirrors read-only, so kernel pread returns EFAULT
RSS boundary: Lupine first-transfer buffers invalidate the preload RSS delta assertion
Product-code workaround: forbidden; keep both behaviors covered by the direct x86_64 cells
Derived client image: forbidden
Server evidence: complete Lupine server output captured with docker logs before cleanup
```

## 5.2 Run the harness

```bash
set -euxo pipefail
ssh tom-workstation "set -euxo pipefail; HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
  /home/tom/.local/bin/uv run --script \
  '$TMS_REMOTE_ROOT/.claude/skills/tms-publish-release/scripts/gpu_validation.py' \
  --release-root '$TMS_REMOTE_ROOT' \
  --artifact-root '$TMS_REMOTE_ARTIFACTS' \
  --expected-version '$TMS_RELEASE_VERSION'"
```

# 6 Pull and recheck artifacts locally

```bash
set -euxo pipefail
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/local-recheck.log" -- \
  rsync -a "tom-workstation:$TMS_REMOTE_ARTIFACTS/" "$TMS_LOCAL_ARTIFACTS/"
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/local-recheck.log" -- \
  rsync -a "tom-workstation:$TMS_REMOTE_ROOT/dist/" "$TMS_LOCAL_ARTIFACTS/dist/"
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/local-recheck.log" -- \
  uv run --script "$TMS_RELEASE_CHECKS" artifacts \
  --dist-dir "$TMS_LOCAL_ARTIFACTS/dist" \
  --expected-version "$TMS_RELEASE_VERSION" \
  --repo-root .
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/local-recheck.log" -- \
  zsh -lc 'set -euxo pipefail; UV_CACHE_DIR="$TMS_LOCAL_ARTIFACTS/uv-cache" uv run --no-project --with twine python -m twine check "$TMS_LOCAL_ARTIFACTS"/dist/*'
uv run --script "$TMS_RELEASE_CHECKS" run --log-path "$TMS_LOCAL_ARTIFACTS/local-recheck.log" -- \
  uv run --script "$TMS_RELEASE_CHECKS" write-manifest \
  --dist-dir "$TMS_LOCAL_ARTIFACTS/dist" \
  --output "$TMS_LOCAL_ARTIFACTS/artifacts.sha256"
```

- Do not upload directly from the remote build directory.
- Keep all three distribution files and full logs at the permanent local artifact path.
- Treat `artifacts.sha256` as the immutable human-review boundary.
- Never overwrite or regenerate the manifest after human review; start a new run instead.

# 7 Write the release evidence report

- Write one Markdown report in the matching local agent-context project `agent-drafts/` directory.
- Use the exact version and run ID in the filename.
- Link every raw artifact and log with its final absolute local path.
- Do not copy raw logs into the report.
- Set `Human review: pending` until the report has been inspected.

```text
Source: full release SHA, branch, clean-tree evidence
Version: canonical version and collision-check results
Host: workstation, GPU, driver, kernel, Docker, binfmt
Artifacts: filename, size, SHA-256, metadata result, twine result, file link
Preflight evidence: release-preflight.log
Build evidence: x86_64 log, aarch64 log, sdist log, artifact-validator log
Local evidence: local-recheck.log and artifacts.sha256
GPU matrix row: architecture, CUDA major, image digest, PyTorch/CUDA/GPU identity
GPU matrix result: pytest summary, every skip reason, terminal RESULT, log link
Lupine boundary: TMS_TEST_LUPINE signal, exact skipped tests, qualification limit
Cleanup: validation containers, server, and network removal evidence
Safety: PyPI upload and release tag not performed; GitHub Release creation forbidden
Human review: pending
```

- Link all five harness logs:
    - `preflight.log`.
    - `x86_64-cuda12-gpu.log`.
    - `x86_64-cuda13-gpu.log`.
    - `aarch64-cuda12-gpu.log`.
    - `aarch64-cuda13-gpu.log`.
- Link `release-preflight.log`, `local-recheck.log`, and `artifacts.sha256`.
- Transcribe the three manifest lines into the artifact table without changing their hashes.
- Require every harness-invoked external command to have an `EXEC` line and a terminal `RESULT` line.
- Require every GPU row to link the exact log containing image identity, pytest output, skip reasons, and cleanup.
- Stop after writing the report.
- Give the human the report link and wait for explicit publish approval.

# 8 Publish

- Continue only after the Section 7 report has been reviewed and the human gives explicit confirmation.
- Run one fail-closed command that appends every immediate pre-upload check and its complete output to `publish-recheck.log`:

```bash
set -euxo pipefail
UV_CACHE_DIR="$TMS_LOCAL_ARTIFACTS/uv-cache" uv run --script "$TMS_RELEASE_CHECKS" pre-upload \
  --dist-dir "$TMS_LOCAL_ARTIFACTS/dist" \
  --manifest "$TMS_LOCAL_ARTIFACTS/artifacts.sha256" \
  --expected-version "$TMS_RELEASE_VERSION" \
  --log-path "$TMS_LOCAL_ARTIFACTS/publish-recheck.log" \
  --repo-root . \
  --remote origin
```

- The command fetches `origin/master`; requires a clean tree and `HEAD == origin/master`; verifies the approved manifest; reruns the complete artifact and Twine gates; and rejects existing PyPI or tag identities.
- Require the verified manifest to equal the three hashes approved in the Section 7 report.
- Upload only the files covered by that manifest.
- Treat an existing PyPI version or remote `v<VERSION>` tag as a hard failure.
- Treat network and authentication errors as hard failures.
- Upload without `--skip-existing`; a collision is a hard failure:

```bash
set -euxo pipefail
UV_CACHE_DIR="$TMS_LOCAL_ARTIFACTS/uv-cache" uv run --no-project --with twine \
  python -m twine upload --non-interactive --repository pypi "$TMS_LOCAL_ARTIFACTS"/dist/*
```

- Poll PyPI with bounded waits until all three files appear under the exact version.
- Create and push an annotated tag only after PyPI confirms the version:

```bash
set -euxo pipefail
git tag -a "v$TMS_RELEASE_VERSION" -m "Release $TMS_RELEASE_VERSION" "$(git rev-parse HEAD)"
git push origin "v$TMS_RELEASE_VERSION"
```

- Do not create a GitHub Release for beta, release-candidate, stable, or post releases.
- Never reuse or move a published version tag.

# 9 Verify the published release from fresh containers

```bash
set -euxo pipefail
rsync -a "$TMS_LOCAL_ARTIFACTS/artifacts.sha256" \
  "tom-workstation:$TMS_REMOTE_ARTIFACTS/artifacts.sha256"
ssh tom-workstation "set -euxo pipefail
cd '$TMS_REMOTE_ROOT'
TMS_PROXY_URL=http://127.0.0.1:7890 \
  .claude/skills/tms-publish-release/scripts/verify_published_release.sh \
  '$TMS_RELEASE_VERSION' \
  '$TMS_REMOTE_ARTIFACTS/artifacts.sha256'" \
  2>&1 | tee "$TMS_LOCAL_ARTIFACTS/verify-published-release.log"
```

- Run this chapter only after PyPI and the Git tag exist.
- Copy the approved `artifacts.sha256` to `$TMS_REMOTE_ARTIFACTS` before invoking the script.
- Keep Chapter 9 implemented as auditable shell; do not move it into a Python orchestration command.
- Require PyPI to contain exactly the two wheels and sdist recorded by the approved manifest, with matching SHA-256 digests.
- Start one fresh `--rm` container for each CUDA major.
- Before installation, require Python to report that `torch_memory_saver` is absent.
- Install the exact published version from PyPI, never from the local `dist/` directory.
- Require the installed package to resolve from `site-packages`, PyTorch to report the matching CUDA major, and the GPU to be the 4090D.
- Run the complete runtime pytest suite with the exact structured skip gate in both containers.
- Require local `verify-published-release.log` to contain two passing pytest summaries and terminal `RESULT: returncode=0`.
- Report any partial release state explicitly.
- Never delete a published PyPI version to make a retry look clean.
- Increment the prerelease number for a retry.

# 10 Reliability

| Symptom | Cause | Action |
| --- | --- | --- |
| Docker Hub pull resets or refuses | Docker daemon is not using `tms-clash-main`, or the selected Clash node is unhealthy | Repair the daemon proxy or refresh the remote Clash config; do not switch registries |
| ARM container reports `exec format error` | `qemu-aarch64` binfmt is absent | Stop; request approval for the privileged binfmt bootstrap |
| Lupine client cannot reach the server | Server startup, Docker network, or matched CUDA pair is wrong | Inspect the cell log; keep the private network and use the exact client/server pair from the matrix |
| ARM PyTorch reports no GPU | The Lupine shim is not first in the client library path or the server lacks the GPU mount | Require `/opt/lupine/lib` in `LD_LIBRARY_PATH`, `LUPINE_SERVER=<container>:14833`, and `--gpus device=0` only on the server |
| ARM and server CUDA tags differ | Client and server protocol/runtime behavior is not the validated pair | Stop and use exactly matched Lupine tags; never mix CUDA 12 and CUDA 13 endpoints |
| ARM validation starts downloading PyTorch | The harness is using a generic client or builder instead of the published worker | Stop and use the digest-pinned `lupine-pytorch-worker`; only NumPy, pytest, and `nvidia-ml-py` should be installed before the final TMS wheel |
| ARM NumPy/pytest install cannot reach `127.0.0.1:7890` | Loopback inside the private-network client is the client itself | Add `host.docker.internal:host-gateway` and use `http://host.docker.internal:7890` inside the ARM client |
| `make build-sdist` cannot import `setuptools` | The sdist ran against an unprepared host Python | Use the repository's containerized target with `TMS_PYTHON_BUILD_IMAGE=python:3.11` |
| `make clean` reports `Permission denied` under `dist/` | An older build left root-owned bind-mount output | Remove only the dedicated rehearsal output through a root Docker container, then use the current scripts that normalize ownership |
| Wheel test imports `/workspace/torch_memory_saver` | Pytest ran against the source tree | Copy only `test/` to `/validation` and rerun against the installed wheel |
| CUDA 13 preload tests report `libcudart.so.13` missing | The runtime image installs CUDART under Python `site-packages/nvidia/cu13/lib` without registering it in `ldconfig` | Discover the matching CUDART path before pytest and export it through `LD_LIBRARY_PATH`; do not skip preload tests |
| `SKIP_GATE_RESULT=fail` | A skip is missing, additional, duplicated, or has a changed reason | Inspect `SKIP_GATE_ACTUAL`; update code or the reviewed matrix contract instead of silently accepting it |
| Cleanup raises after a passing cell | Docker cleanup timed out or could not start | Inspect the complete cell log; the harness attempted every cleanup command and preserved any earlier validation error |
| Manifest verification fails | An artifact changed or the file set differs after human review | Stop and start a new build and review run; never regenerate the approved manifest |
| Post-release verification reports `jq: command not found` | The workstation lacks the host-side JSON verifier | Install `jq` with `sudo apt-get install -y jq`; the script checks every host dependency before network access |
| `dist/` has extra wheels | Stale or partial build output | Stop and start a new run directory; never repair or reuse a release tree |
| PyPI upload reports an existing filename | Version or artifact was already published | Stop; never use `--skip-existing` or overwrite a release |
