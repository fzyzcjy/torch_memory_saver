#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "typer"]
# ///

from __future__ import annotations

import hashlib
import platform
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

import click
import typer

app = typer.Typer(add_completion=False)

_PROXY_URL = "http://127.0.0.1:7890"
_COMMAND_TIMEOUT_SECONDS = 7200
_RUNTIME_TEST_MODULES = (
    "test_configure_subprocess.py",
    "test_examples.py",
    "test_utils.py",
)
_BUILD_TOOL_TEST_MODULES = ("test_merge_cuda_wheels.py",)
_X86_VALIDATION_SCRIPT = r"""
set -euxo pipefail
python --version
python - <<'PY'
import os
import torch

cuda = torch.version.cuda
name = torch.cuda.get_device_name(0)
print(torch.__version__, cuda, torch.cuda.is_available(), name)
assert cuda is not None and cuda.split(".", 1)[0] == os.environ["TMS_CUDA_MAJOR"]
assert torch.cuda.is_available()
assert "4090 D" in name or "4090D" in name
PY
CUDA_RUNTIME_LIB="$(python -c 'from pathlib import Path; import os, site; major=os.environ["TMS_CUDA_MAJOR"]; matches=[path for root in site.getsitepackages() for path in (Path(root) / "nvidia").glob(f"**/libcudart.so.{major}")]; assert matches, matches; print(":".join(sorted({str(path.parent) for path in matches})))')"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "CUDA_RUNTIME_LIB=${CUDA_RUNTIME_LIB}"
python -m pip install --no-cache-dir pytest==8.3.5 nvidia-ml-py==12.570.86
python -m pip install --no-deps "/workspace/dist/torch_memory_saver-${TMS_RELEASE_VERSION}-cp39-abi3-manylinux2014_x86_64.whl"
mkdir -p /validation/test
cp -a /workspace/test/examples /validation/test/examples
cp /workspace/test/test_configure_subprocess.py /workspace/test/test_examples.py /workspace/test/test_utils.py /validation/test/
cd /validation
python -c 'from pathlib import Path; import torch_memory_saver; path=Path(torch_memory_saver.__file__); print(path); assert "site-packages" in path.parts'
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 3600s python -m pytest test -vv -ra
""".strip()
_ARM_VALIDATION_SCRIPT = r"""
set -euxo pipefail
python3 --version
uname -m
python3 -m pip install --break-system-packages --only-binary=:all: --no-cache-dir pytest==8.3.5 numpy==2.2.3 nvidia-ml-py==12.570.86
python3 -m pip install --break-system-packages --no-deps "/workspace/dist/torch_memory_saver-${TMS_RELEASE_VERSION}-cp39-abi3-manylinux2014_aarch64.whl"
mkdir -p /validation/test
cp -a /workspace/test/examples /validation/test/examples
cp /workspace/test/test_configure_subprocess.py /workspace/test/test_examples.py /workspace/test/test_utils.py /validation/test/
cd /validation
python3 - <<'PY'
from pathlib import Path
import os
import platform
import torch
import torch_memory_saver

cuda = torch.version.cuda
name = torch.cuda.get_device_name(0)
package_path = Path(torch_memory_saver.__file__)
print(platform.machine(), torch.__version__, cuda, torch.cuda.is_available(), name)
print(package_path)
assert platform.machine() == "aarch64"
assert cuda is not None and cuda.split(".", 1)[0] == os.environ["TMS_CUDA_MAJOR"]
assert torch.cuda.is_available()
assert "4090 D" in name or "4090D" in name
assert any(part in {"site-packages", "dist-packages"} for part in package_path.parts)
PY
python3 - <<'PY'
from pathlib import Path
import platform
import torch
import torch_memory_saver

paths = [Path(torch._C.__file__), *Path(torch_memory_saver.__file__).parent.parent.glob("torch_memory_saver_hook_mode_*.abi3.so")]
assert paths
for path in paths:
    binary = path.read_bytes()[:20]
    assert binary[:4] == b"\x7fELF", path
    byte_order = "little" if binary[5] == 1 else "big"
    assert int.from_bytes(binary[18:20], byteorder=byte_order) == 183, path
    print(platform.machine(), path)
PY
CUDA_RUNTIME_LIB="$(python3 -c 'from pathlib import Path; import os, site; major=os.environ["TMS_CUDA_MAJOR"]; matches=[path for root in site.getsitepackages() for path in (Path(root) / "nvidia").glob(f"**/libcudart.so.{major}")]; assert matches, matches; print(":".join(sorted({str(path.parent) for path in matches})))')"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "CUDA_RUNTIME_LIB=${CUDA_RUNTIME_LIB}"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 3600s python3 -m pytest test -vv -ra
""".strip()


@dataclass(frozen=True)
class GpuValidationConfig:
    release_root: Path
    run_dir: Path
    expected_version: str
    proxy_url: str


@dataclass(frozen=True)
class GpuValidationCase:
    name: str
    image: str
    architecture: str
    cuda_major: str
    server_image: str | None = None
    test_environment: tuple[tuple[str, str], ...] = ()


_GPU_VALIDATION_CASES: tuple[GpuValidationCase, ...] = (
    GpuValidationCase(
        name="x86_64-cuda12-gpu",
        image="docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        architecture="x86_64",
        cuda_major="12",
    ),
    GpuValidationCase(
        name="x86_64-cuda13-gpu",
        image="docker.io/pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime",
        architecture="x86_64",
        cuda_major="13",
    ),
    GpuValidationCase(
        name="aarch64-cuda12-gpu",
        image="ghcr.io/lupinemachines/lupine-pytorch-worker@sha256:f152fbcb3e2eda5661abafdfb4f3024afe9b775eb702015b5df0527ca0b7f556",
        architecture="aarch64",
        cuda_major="12",
        server_image="ghcr.io/lupinemachines/lupine-server@sha256:e6b1103392165e929ca7f4f910eeec9f0b0f7155c5ff65b7402ca083c6bf9d53",
        test_environment=(("TMS_TEST_LUPINE", "1"),),
    ),
    GpuValidationCase(
        name="aarch64-cuda13-gpu",
        image="ghcr.io/lupinemachines/lupine-pytorch-worker@sha256:a154530bfeed825e8c915be1b6129965f293dbf29ce4764441014dccf9c6c08a",
        architecture="aarch64",
        cuda_major="13",
        server_image="ghcr.io/lupinemachines/lupine-server@sha256:f1d805e14e0b2da5d5912adeed72f3e1c7d0458082c3cbaf3ba9bb2346b869cd",
        test_environment=(("TMS_TEST_LUPINE", "1"),),
    ),
)


@app.command()
def main(
    release_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    artifact_root: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ],
    expected_version: Annotated[str, typer.Option()],
    proxy_url: Annotated[str, typer.Option()] = _PROXY_URL,
) -> None:
    config = _build_config(
        release_root=release_root,
        artifact_root=artifact_root,
        expected_version=expected_version,
        proxy_url=proxy_url,
    )
    try:
        run_gpu_validation(config=config)
    except (
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise click.ClickException(f"{error}; logs: {config.run_dir}") from error
    typer.echo(f"Validated release GPU matrix in fresh containers: {config.run_dir}")


def run_gpu_validation(*, config: GpuValidationConfig) -> None:
    if platform.machine() != "x86_64":
        raise RuntimeError(
            f"GPU validation requires the x86_64 tom-workstation host, found {platform.machine()}"
        )

    _require_release_wheels(config=config)
    _require_runtime_test_contract(release_root=config.release_root)
    preflight_log = config.run_dir / "preflight.log"
    _exec_command(command=["docker", "info"], log_path=preflight_log)
    _exec_command(
        command=[
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        log_path=preflight_log,
    )

    for case in _GPU_VALIDATION_CASES:
        if case.architecture == "aarch64":
            _run_lupine_case(config=config, case=case)
        else:
            _run_x86_case(config=config, case=case)


def _build_config(
    *,
    release_root: Path,
    artifact_root: Path,
    expected_version: str,
    proxy_url: str,
) -> GpuValidationConfig:
    release_root = release_root.resolve()
    artifact_root = artifact_root.resolve()
    run_dir = artifact_root / datetime.now(tz=timezone.utc).strftime(
        "gpu-validation-%Y%m%d-%H%M%S-%fZ"
    )
    run_dir.mkdir()

    return GpuValidationConfig(
        release_root=release_root,
        run_dir=run_dir,
        expected_version=expected_version,
        proxy_url=proxy_url,
    )


def _require_release_wheels(*, config: GpuValidationConfig) -> None:
    expected_wheels = [
        config.release_root
        / "dist"
        / f"torch_memory_saver-{config.expected_version}-cp39-abi3-manylinux2014_{architecture}.whl"
        for architecture in ("x86_64", "aarch64")
    ]
    if missing_wheels := [path for path in expected_wheels if not path.is_file()]:
        raise ValueError(f"Missing release wheels: {missing_wheels}")


def _require_runtime_test_contract(*, release_root: Path) -> None:
    test_dir = release_root / "test"
    actual_modules = {path.name for path in test_dir.glob("test_*.py")}
    expected_modules = set(_RUNTIME_TEST_MODULES) | set(_BUILD_TOOL_TEST_MODULES)
    if actual_modules != expected_modules:
        missing = sorted(expected_modules - actual_modules)
        unclassified = sorted(actual_modules - expected_modules)
        raise ValueError(
            f"Release test classification mismatch: missing={missing}, unclassified={unclassified}"
        )
    if not (test_dir / "examples").is_dir():
        raise ValueError(f"Missing runtime examples: {test_dir / 'examples'}")


def _run_x86_case(*, config: GpuValidationConfig, case: GpuValidationCase) -> None:
    log_path = config.run_dir / f"{case.name}.log"
    container_name = f"{_resource_prefix(config=config, case=case)}-validation"
    identity_logged = False
    try:
        _exec_command(
            command=[
                "docker",
                "run",
                "--rm",
                "--pull",
                "missing",
                "--name",
                container_name,
                "--gpus",
                "device=0",
                "--network",
                "host",
                *_environment_args(
                    config=config,
                    case=case,
                    proxy_url=config.proxy_url,
                ),
                "-v",
                f"{config.release_root}:/workspace:ro",
                case.image,
                "bash",
                "-lc",
                _X86_VALIDATION_SCRIPT,
            ],
            log_path=log_path,
        )
        _inspect_image(image=case.image, log_path=log_path)
        identity_logged = True
    finally:
        if not identity_logged:
            _inspect_image(image=case.image, log_path=log_path, check=False)
        _exec_command(
            command=["docker", "container", "rm", "--force", container_name],
            log_path=log_path,
            check=False,
        )


def _run_lupine_case(*, config: GpuValidationConfig, case: GpuValidationCase) -> None:
    if case.server_image is None:
        raise ValueError(f"Incomplete Lupine case: {case}")

    log_path = config.run_dir / f"{case.name}.log"
    resource_prefix = _resource_prefix(config=config, case=case)
    network_name = f"{resource_prefix}-network"
    server_name = f"{resource_prefix}-server"
    smoke_client_name = f"{resource_prefix}-smoke"
    validation_client_name = f"{resource_prefix}-validation"
    server_identity_logged = False
    worker_identity_logged = False

    try:
        _exec_command(
            command=["docker", "network", "create", network_name],
            log_path=log_path,
        )
        _exec_command(
            command=[
                "docker",
                "run",
                "--rm",
                "--pull",
                "missing",
                "--detach",
                "--name",
                server_name,
                "--network",
                network_name,
                "--platform",
                "linux/amd64",
                "--gpus",
                "device=0",
                case.server_image,
            ],
            log_path=log_path,
        )
        _inspect_image(image=case.server_image, log_path=log_path)
        server_identity_logged = True
        client_prefix = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "missing",
            "--platform",
            "linux/arm64",
            "--network",
            network_name,
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            f"LUPINE_SERVER={server_name}:14833",
        ]
        _exec_command(
            command=[
                *client_prefix,
                "--name",
                smoke_client_name,
                case.image,
                "bash",
                "-lc",
                "for attempt in $(seq 1 30); do timeout --signal=TERM --kill-after=5s 20s python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' && exit 0; sleep 2; done; exit 1",
            ],
            log_path=log_path,
        )
        _inspect_image(image=case.image, log_path=log_path)
        worker_identity_logged = True
        _exec_command(
            command=[
                *client_prefix,
                "--name",
                validation_client_name,
                *_environment_args(
                    config=config,
                    case=case,
                    proxy_url=_host_gateway_proxy_url(proxy_url=config.proxy_url),
                ),
                "-v",
                f"{config.release_root}:/workspace:ro",
                case.image,
                "bash",
                "-lc",
                _ARM_VALIDATION_SCRIPT,
            ],
            log_path=log_path,
        )
    finally:
        if not worker_identity_logged:
            _inspect_image(image=case.image, log_path=log_path, check=False)
        if not server_identity_logged:
            _inspect_image(image=case.server_image, log_path=log_path, check=False)
        for client_name in (smoke_client_name, validation_client_name):
            _exec_command(
                command=["docker", "container", "rm", "--force", client_name],
                log_path=log_path,
                check=False,
            )
        _exec_command(
            command=["docker", "container", "rm", "--force", server_name],
            log_path=log_path,
            check=False,
        )
        _exec_command(
            command=["docker", "network", "rm", network_name],
            log_path=log_path,
            check=False,
        )


def _environment_args(
    *,
    config: GpuValidationConfig,
    case: GpuValidationCase,
    proxy_url: str,
) -> list[str]:
    return [
        argument
        for name, value in (
            ("http_proxy", proxy_url),
            ("https_proxy", proxy_url),
            ("TMS_RELEASE_VERSION", config.expected_version),
            ("TMS_CUDA_MAJOR", case.cuda_major),
            *case.test_environment,
        )
        for argument in ("-e", f"{name}={value}")
    ]


def _host_gateway_proxy_url(*, proxy_url: str) -> str:
    parsed = urlsplit(proxy_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return proxy_url
    if parsed.port is None:
        raise ValueError(f"Loopback proxy URL requires a port: {proxy_url}")

    return urlunsplit(
        (
            parsed.scheme,
            f"host.docker.internal:{parsed.port}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def _resource_prefix(*, config: GpuValidationConfig, case: GpuValidationCase) -> str:
    digest = hashlib.sha256(str(config.run_dir).encode()).hexdigest()[:10]
    return f"tms-{case.name}-{digest}"


def _inspect_image(*, image: str, log_path: Path, check: bool = True) -> None:
    try:
        _exec_command(
            command=[
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                'image_id={{.Id}} repo_digests={{json .RepoDigests}} architecture={{.Architecture}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}',
            ],
            log_path=log_path,
            check=check,
        )
    except (OSError, subprocess.TimeoutExpired):
        if check:
            raise


def _exec_command(
    *,
    command: list[str],
    log_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    rendered_command = shlex.join(command)
    typer.echo(f"EXEC: {rendered_command}")
    with log_path.open(mode="a", encoding="utf-8") as output:
        output.write(f"EXEC: {rendered_command}\n")
        output.flush()
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            output.write(f"RESULT: timeout={error.timeout} seconds\n")
            output.flush()
            raise
        except OSError as error:
            output.write(f"RESULT: error={type(error).__name__}: {error}\n")
            output.flush()
            raise
        output.write(f"RESULT: returncode={result.returncode}\n")
        output.flush()

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=command,
        )
    return result


if __name__ == "__main__":
    app()
