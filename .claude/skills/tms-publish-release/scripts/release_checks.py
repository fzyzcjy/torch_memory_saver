#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "twine", "typer"]
# ///

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import re
import shlex
import subprocess
import sys
import tarfile
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import click
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

_CANONICAL_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:(?:a|b|rc)(?:0|[1-9]\d*)|\.post(?:0|[1-9]\d*))?$"
)
_ARCHITECTURES = ("x86_64", "aarch64")
_ELF_MACHINE_BY_ARCHITECTURE = {"x86_64": 62, "aarch64": 183}
_PYPI_RELEASE_URL = "https://pypi.org/pypi/torch-memory-saver/{version}/json"
_COMMAND_TIMEOUT_SECONDS = 7200
_EXPECTED_BINARY_NAMES = {
    f"torch_memory_saver_hook_mode_{hook_mode}{suffix}.abi3.so"
    for hook_mode in ("preload", "torch")
    for suffix in ("", "_cu12", "_cu13")
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@app.command("version")
def _version_command(
    setup_py: Annotated[
        Path,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    expected_version: Annotated[str, typer.Option()],
) -> None:
    try:
        validate_setup_version(
            setup_py=setup_py,
            expected_version=expected_version,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--expected-version") from error
    typer.echo(f"Validated canonical setup.py version: {expected_version}")


@app.command("pypi")
def _pypi_command(
    expected_version: Annotated[str, typer.Option()],
) -> None:
    validate_canonical_version(version=expected_version)
    try:
        version_exists = pypi_version_exists(version=expected_version)
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error

    if version_exists:
        raise typer.BadParameter(
            f"torch-memory-saver {expected_version!r} already exists on PyPI",
            param_hint="--expected-version",
        )
    typer.echo(f"Confirmed PyPI version is available: {expected_version}")


@app.command("publish-preflight")
def _publish_preflight_command(
    expected_version: Annotated[str, typer.Option()],
    remote: Annotated[str, typer.Option()] = "origin",
) -> None:
    validate_canonical_version(version=expected_version)
    try:
        collisions = find_publish_collisions(
            version=expected_version,
            remote=remote,
        )
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    if collisions:
        raise typer.BadParameter(
            f"Release identity already exists: {', '.join(collisions)}",
            param_hint="--expected-version",
        )
    typer.echo(
        f"Confirmed publish identity is unused on PyPI and {remote}: {expected_version}"
    )


@app.command("pre-upload")
def _pre_upload_command(
    dist_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    manifest: Annotated[
        Path,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    expected_version: Annotated[str, typer.Option()],
    log_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)],
    repo_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ] = Path("."),
    remote: Annotated[str, typer.Option()] = "origin",
) -> None:
    try:
        run_pre_upload_checks(
            dist_dir=dist_dir,
            manifest=manifest,
            expected_version=expected_version,
            log_path=log_path,
            repo_root=repo_root,
            remote=remote,
        )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        raise click.ClickException(str(error)) from error
    typer.echo(f"Completed every pre-upload check: {log_path}")


@app.command("artifacts")
def _artifacts_command(
    dist_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    expected_version: Annotated[str, typer.Option()],
    repo_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ] = Path("."),
) -> None:
    validate_release_artifacts(
        dist_dir=dist_dir,
        expected_version=expected_version,
        repo_root=repo_root,
    )
    typer.echo(f"Validated release artifacts for {expected_version}: {dist_dir}")


@app.command("write-manifest")
def _write_manifest_command(
    dist_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    output: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)],
) -> None:
    write_sha256_manifest(dist_dir=dist_dir, output=output)
    typer.echo(f"Wrote release artifact manifest: {output}")


@app.command("verify-manifest")
def _verify_manifest_command(
    dist_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    manifest: Annotated[
        Path,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    verify_sha256_manifest(dist_dir=dist_dir, manifest=manifest)
    typer.echo(f"Verified release artifact manifest: {manifest}")


@app.command("sdist")
def _sdist_command(
    sdist_path: Annotated[
        Path,
        typer.Option(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    expected_version: Annotated[str, typer.Option()],
    repo_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True),
    ] = Path("."),
) -> None:
    validate_sdist(
        sdist_path=sdist_path,
        expected_version=expected_version,
        repo_root=repo_root,
    )
    typer.echo(f"Validated source distribution for {expected_version}: {sdist_path}")


@app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def _run_command(
    context: typer.Context,
    log_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)],
) -> None:
    command = list(context.args)
    if not command:
        raise typer.BadParameter("A command is required after --")
    try:
        returncode = run_logged_command(command=command, log_path=log_path)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise click.ClickException(str(error)) from error
    if returncode != 0:
        raise typer.Exit(code=returncode)


def read_setup_version(*, setup_py: Path) -> str:
    tree = ast.parse(setup_py.read_text(), filename=str(setup_py))
    versions = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_setup_call(node=node)
        for keyword in node.keywords
        if keyword.arg == "version"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    ]

    if len(versions) != 1:
        raise ValueError(
            f"Expected exactly one literal setup(version=...), found {versions}"
        )
    return versions[0]


def validate_setup_version(*, setup_py: Path, expected_version: str) -> None:
    version = read_setup_version(setup_py=setup_py)
    validate_canonical_version(version=version)
    if version != expected_version:
        raise ValueError(
            f"setup.py version mismatch: expected {expected_version!r}, found {version!r}"
        )


def validate_canonical_version(*, version: str) -> None:
    if not _CANONICAL_VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            f"Version {version!r} is not canonical for TMS; use X.Y.Z, X.Y.ZbN, "
            "X.Y.ZrcN, or X.Y.Z.postN"
        )


def pypi_version_exists(*, version: str) -> bool:
    url = _PYPI_RELEASE_URL.format(version=version)
    try:
        with urlopen(url, timeout=20):
            return True
    except HTTPError as error:
        if error.code == 404:
            return False
        raise RuntimeError(
            f"PyPI returned HTTP {error.code} while checking {version!r}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not query PyPI for {version!r}: {error.reason}"
        ) from error


def find_publish_collisions(*, version: str, remote: str) -> list[str]:
    collisions: list[str] = []
    if pypi_version_exists(version=version):
        collisions.append(f"PyPI version {version}")
    if remote_tag_exists(version=version, remote=remote):
        collisions.append(f"{remote} tag v{version}")
    return collisions


def remote_tag_exists(*, version: str, remote: str) -> bool:
    result = _exec_command(
        command=[
            "git",
            "ls-remote",
            "--exit-code",
            "--tags",
            remote,
            f"refs/tags/v{version}",
        ]
    )
    if result.returncode == 0:
        return True
    if result.returncode == 2:
        return False
    raise RuntimeError(
        f"Could not check {remote} for tag v{version}: {result.stderr.strip()}"
    )


def validate_release_artifacts(
    *,
    dist_dir: Path,
    expected_version: str,
    repo_root: Path,
) -> None:
    validate_canonical_version(version=expected_version)
    expected_names = {
        f"torch_memory_saver-{expected_version}-cp39-abi3-manylinux2014_{architecture}.whl"
        for architecture in _ARCHITECTURES
    } | {f"torch_memory_saver-{expected_version}.tar.gz"}
    actual_names = {path.name for path in dist_dir.iterdir() if path.is_file()}

    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"Release artifact set mismatch: missing={missing}, unexpected={unexpected}"
        )

    for architecture in _ARCHITECTURES:
        validate_wheel(
            wheel_path=dist_dir
            / f"torch_memory_saver-{expected_version}-cp39-abi3-manylinux2014_{architecture}.whl",
            expected_version=expected_version,
            expected_architecture=architecture,
        )
    validate_sdist(
        sdist_path=dist_dir / f"torch_memory_saver-{expected_version}.tar.gz",
        expected_version=expected_version,
        repo_root=repo_root,
    )


def validate_wheel(
    *,
    wheel_path: Path,
    expected_version: str,
    expected_architecture: str,
) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        file_names = [name for name in wheel.namelist() if not name.endswith("/")]
        names = set(file_names)
        if len(file_names) != len(names):
            raise ValueError(f"Duplicate file entry in {wheel_path.name}")
        binary_paths = sorted(name for name in names if name.endswith(".so"))
        binary_path_by_name = {Path(name).name: name for name in binary_paths}
        binary_names = set(binary_path_by_name)
        if binary_names != _EXPECTED_BINARY_NAMES:
            missing = sorted(_EXPECTED_BINARY_NAMES - binary_names)
            unexpected = sorted(binary_names - _EXPECTED_BINARY_NAMES)
            raise ValueError(
                f"Wheel binary set mismatch in {wheel_path.name}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        expected_machine = _ELF_MACHINE_BY_ARCHITECTURE[expected_architecture]
        for binary_path in binary_paths:
            machine = _read_elf_machine(
                binary=wheel.read(binary_path),
                source=f"{wheel_path.name}:{binary_path}",
            )
            if machine != expected_machine:
                raise ValueError(
                    f"ELF machine mismatch in {wheel_path.name}:{binary_path}: "
                    f"expected {expected_machine}, found {machine}"
                )

        for hook_mode in ("preload", "torch"):
            compatibility_name = f"torch_memory_saver_hook_mode_{hook_mode}.abi3.so"
            cuda12_name = f"torch_memory_saver_hook_mode_{hook_mode}_cu12.abi3.so"
            if wheel.read(binary_path_by_name[compatibility_name]) != wheel.read(
                binary_path_by_name[cuda12_name]
            ):
                raise ValueError(
                    f"CUDA 12 compatibility binary differs in {wheel_path.name}: "
                    f"{compatibility_name} != {cuda12_name}"
                )

        metadata_name = _find_one(names=names, suffix=".dist-info/METADATA")
        wheel_metadata_name = _find_one(names=names, suffix=".dist-info/WHEEL")
        _validate_wheel_record(
            wheel=wheel,
            names=names,
            wheel_name=wheel_path.name,
        )
        metadata = wheel.read(metadata_name).decode()
        wheel_metadata = wheel.read(wheel_metadata_name).decode()

    _validate_metadata_version(
        metadata=metadata,
        expected_version=expected_version,
        source=wheel_path.name,
    )
    expected_tag = f"Tag: cp39-abi3-manylinux2014_{expected_architecture}"
    if expected_tag not in wheel_metadata:
        raise ValueError(
            f"Missing {expected_tag!r} in {wheel_path.name} WHEEL metadata"
        )


def _validate_wheel_record(
    *,
    wheel: zipfile.ZipFile,
    names: set[str],
    wheel_name: str,
) -> None:
    record_name = _find_one(names=names, suffix=".dist-info/RECORD")
    rows = list(csv.reader(StringIO(wheel.read(record_name).decode("utf-8"))))
    entries: dict[str, tuple[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            raise ValueError(
                f"Invalid RECORD row {row_number} in {wheel_name}: {row!r}"
            )
        path, digest, size = row
        if path in entries:
            raise ValueError(f"Duplicate RECORD path in {wheel_name}: {path}")
        entries[path] = (digest, size)

    if set(entries) != names:
        missing = sorted(names - set(entries))
        unexpected = sorted(set(entries) - names)
        raise ValueError(
            f"RECORD member set mismatch in {wheel_name}: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for path, (digest, size) in entries.items():
        if path == record_name:
            if digest or size:
                raise ValueError(
                    f"RECORD self-entry must omit hash and size in {wheel_name}"
                )
            continue

        content = wheel.read(path)
        expected_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        if digest != f"sha256={expected_digest}":
            raise ValueError(f"RECORD hash mismatch in {wheel_name}: {path}")
        if size != str(len(content)):
            raise ValueError(f"RECORD size mismatch in {wheel_name}: {path}")


def write_sha256_manifest(*, dist_dir: Path, output: Path) -> None:
    if output.parent.resolve() == dist_dir.resolve():
        raise ValueError("Release artifact manifest must be outside the dist directory")
    artifact_paths = sorted(path for path in dist_dir.iterdir() if path.is_file())
    if not artifact_paths:
        raise ValueError(f"No release artifacts found in {dist_dir}")
    if output.exists():
        raise ValueError(f"Refusing to overwrite release artifact manifest: {output}")

    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in artifact_paths
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_sha256_manifest(*, dist_dir: Path, manifest: Path) -> None:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None:
            raise ValueError(
                f"Invalid SHA-256 manifest line {line_number} in {manifest}: {line!r}"
            )
        digest, name = match.groups()
        if name in entries:
            raise ValueError(f"Duplicate artifact in SHA-256 manifest: {name}")
        entries[name] = digest

    actual_paths = {path.name: path for path in dist_dir.iterdir() if path.is_file()}
    if set(entries) != set(actual_paths):
        missing = sorted(set(entries) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(entries))
        raise ValueError(
            f"SHA-256 manifest artifact set mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for name, expected_digest in entries.items():
        actual_digest = hashlib.sha256(actual_paths[name].read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected_digest}, "
                f"found {actual_digest}"
            )


def run_logged_command(*, command: list[str], log_path: Path) -> int:
    rendered_command = shlex.join(command)
    typer.echo(f"EXEC: {rendered_command}", err=True)
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
    return result.returncode


def run_pre_upload_checks(
    *,
    dist_dir: Path,
    manifest: Path,
    expected_version: str,
    log_path: Path,
    repo_root: Path,
    remote: str,
) -> None:
    validate_canonical_version(version=expected_version)
    git_script = r"""
set -euxo pipefail
git -C "$2" fetch "$1" master
git -C "$2" status --short --branch
test -z "$(git -C "$2" status --porcelain)"
git -C "$2" rev-parse HEAD
git -C "$2" rev-parse "$1/master"
test "$(git -C "$2" rev-parse HEAD)" = "$(git -C "$2" rev-parse "$1/master")"
""".strip()
    if (
        returncode := run_logged_command(
            command=[
                "bash",
                "-lc",
                git_script,
                "tms-pre-upload",
                remote,
                str(repo_root),
            ],
            log_path=log_path,
        )
    ) != 0:
        raise RuntimeError(f"Git pre-upload checks failed with exit code {returncode}")

    _run_logged_validation(
        name="version",
        arguments=[expected_version, str(repo_root / "setup.py")],
        log_path=log_path,
        validation=lambda: validate_setup_version(
            setup_py=repo_root / "setup.py",
            expected_version=expected_version,
        ),
    )
    _run_logged_validation(
        name="verify-manifest",
        arguments=[str(manifest), str(dist_dir)],
        log_path=log_path,
        validation=lambda: verify_sha256_manifest(
            dist_dir=dist_dir,
            manifest=manifest,
        ),
    )
    _run_logged_validation(
        name="artifacts",
        arguments=[expected_version, str(dist_dir), str(repo_root)],
        log_path=log_path,
        validation=lambda: validate_release_artifacts(
            dist_dir=dist_dir,
            expected_version=expected_version,
            repo_root=repo_root,
        ),
    )
    artifact_paths = sorted(path for path in dist_dir.iterdir() if path.is_file())
    if (
        returncode := run_logged_command(
            command=[sys.executable, "-m", "twine", "check", *map(str, artifact_paths)],
            log_path=log_path,
        )
    ) != 0:
        raise RuntimeError(f"Twine check failed with exit code {returncode}")

    def validate_namespaces() -> None:
        if collisions := find_publish_collisions(
            version=expected_version,
            remote=remote,
        ):
            raise ValueError(
                f"Release identity already exists: {', '.join(collisions)}"
            )

    _run_logged_validation(
        name="publish-preflight",
        arguments=[expected_version, remote],
        log_path=log_path,
        validation=validate_namespaces,
    )


def _run_logged_validation(
    *,
    name: str,
    arguments: list[str],
    log_path: Path,
    validation: Callable[[], None],
) -> None:
    rendered_command = shlex.join(["release_checks", name, *arguments])
    typer.echo(f"EXEC: {rendered_command}", err=True)
    with log_path.open(mode="a", encoding="utf-8") as output:
        output.write(f"EXEC: {rendered_command}\n")
        output.flush()
        try:
            validation()
        except Exception as error:
            traceback.print_exc(file=output)
            output.write(f"RESULT: error={type(error).__name__}: {error}\n")
            output.flush()
            raise
        output.write("RESULT: returncode=0\n")
        output.flush()


def validate_sdist(*, sdist_path: Path, expected_version: str, repo_root: Path) -> None:
    with tarfile.open(sdist_path, mode="r:gz") as sdist:
        names = set(sdist.getnames())
        package_info_name = _find_sdist_package_info(names=names)
        package_info_file = sdist.extractfile(package_info_name)
        if package_info_file is None:
            raise ValueError(
                f"Could not read {package_info_name} from {sdist_path.name}"
            )
        metadata = package_info_file.read().decode()

    relative_names = {name.split("/", 1)[1] for name in names if "/" in name}
    expected_sources = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "csrc").iterdir()
        if path.is_file() and path.suffix in {".cpp", ".h"}
    }
    if not expected_sources:
        raise ValueError(f"No C++ or header sources found under {repo_root / 'csrc'}")
    if missing_sources := sorted(expected_sources - relative_names):
        raise ValueError(
            f"Source distribution is missing backend sources: {missing_sources}"
        )

    _validate_metadata_version(
        metadata=metadata,
        expected_version=expected_version,
        source=sdist_path.name,
    )


def _is_setup_call(*, node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "setup"


def _find_one(*, names: set[str], suffix: str) -> str:
    matches = sorted(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file ending in {suffix!r}, found {matches}"
        )
    return matches[0]


def _find_sdist_package_info(*, names: set[str]) -> str:
    matches = sorted(
        name for name in names if name.count("/") == 1 and name.endswith("/PKG-INFO")
    )
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one root PKG-INFO, found {matches}")
    return matches[0]


def _read_elf_machine(*, binary: bytes, source: str) -> int:
    if len(binary) < 20 or binary[:4] != b"\x7fELF":
        raise ValueError(f"Invalid ELF binary in {source}")
    if binary[5] not in {1, 2}:
        raise ValueError(f"Invalid ELF byte order in {source}: {binary[5]}")

    byte_order = "little" if binary[5] == 1 else "big"
    return int.from_bytes(binary[18:20], byteorder=byte_order)


def _validate_metadata_version(
    *, metadata: str, expected_version: str, source: str
) -> None:
    version_lines = [
        line for line in metadata.splitlines() if line.startswith("Version: ")
    ]
    expected_line = f"Version: {expected_version}"
    if version_lines != [expected_line]:
        raise ValueError(
            f"Version metadata mismatch in {source}: expected {expected_line!r}, found {version_lines}"
        )


def _exec_command(*, command: list[str]) -> CommandResult:
    typer.echo(f"EXEC: {shlex.join(command)}", err=True)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


if __name__ == "__main__":
    app()
