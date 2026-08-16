from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError

import pytest
from typer.testing import CliRunner


def _load_release_checks_module() -> ModuleType:
    module_path = Path(__file__).with_name("release_checks.py")
    spec = importlib.util.spec_from_file_location("release_checks", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_checks = _load_release_checks_module()
cli_runner = CliRunner()


class TestVersionValidation:
    def test_version_command_validates_setup_version(self, tmp_path: Path) -> None:
        """The Typer command validates a matching canonical setup.py version."""

        setup_py = tmp_path / "setup.py"
        setup_py.write_text(
            "from setuptools import setup\nsetup(name='example', version='0.0.10b1')\n"
        )

        result = cli_runner.invoke(
            release_checks.app,
            [
                "version",
                "--setup-py",
                str(setup_py),
                "--expected-version",
                "0.0.10b1",
            ],
        )

        assert result.exit_code == 0
        assert result.stdout == "Validated canonical setup.py version: 0.0.10b1\n"

    def test_canonical_beta_is_accepted(self) -> None:
        """Canonical compact beta versions are accepted."""

        release_checks.validate_canonical_version(version="0.0.10b1")

    @pytest.mark.parametrize(
        "version",
        ["0.0.10.beta-1", "0.0.10beta1", "0.0.10-b1", "v0.0.10b1"],
    )
    def test_noncanonical_beta_spellings_are_rejected(self, version: str) -> None:
        """Alternative beta spellings are rejected before artifact creation."""

        with pytest.raises(ValueError, match="not canonical"):
            release_checks.validate_canonical_version(version=version)

    def test_setup_version_reads_literal_keyword(self, tmp_path: Path) -> None:
        """The setup.py version is read without importing build configuration."""

        setup_py = tmp_path / "setup.py"
        setup_py.write_text(
            "from setuptools import setup\nsetup(name='example', version='0.0.10b1')\n"
        )

        assert release_checks.read_setup_version(setup_py=setup_py) == "0.0.10b1"


class TestPyPIVersionValidation:
    def test_pypi_command_accepts_unpublished_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing PyPI release is accepted as available for publication."""

        def raise_not_found(url: str, *, timeout: int) -> None:
            assert url.endswith("/torch-memory-saver/0.0.10b1/json")
            assert timeout == 20
            raise HTTPError(url=url, code=404, msg="Not Found", hdrs=None, fp=None)

        monkeypatch.setattr(release_checks, "urlopen", raise_not_found)

        result = cli_runner.invoke(
            release_checks.app,
            ["pypi", "--expected-version", "0.0.10b1"],
        )

        assert result.exit_code == 0
        assert result.stdout == "Confirmed PyPI version is available: 0.0.10b1\n"

    def test_pypi_command_rejects_published_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing PyPI release is rejected before artifacts are built."""

        monkeypatch.setattr(
            release_checks,
            "urlopen",
            lambda url, *, timeout: BytesIO(b"{}"),
        )

        result = cli_runner.invoke(
            release_checks.app,
            ["pypi", "--expected-version", "0.0.10b1"],
        )

        assert result.exit_code == 2
        assert result.exception is not None
        assert result.exception.__context__ is not None
        assert "already exists on PyPI" in str(result.exception.__context__)

    def test_pypi_network_failure_is_reported_without_secondary_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A namespace lookup failure remains an actionable CLI error."""

        def fail_lookup(*, version: str) -> bool:
            raise RuntimeError("PyPI unavailable")

        monkeypatch.setattr(release_checks, "pypi_version_exists", fail_lookup)

        result = cli_runner.invoke(
            release_checks.app,
            ["pypi", "--expected-version", "0.0.10b1"],
        )

        assert result.exit_code == 1
        assert isinstance(result.exception, release_checks.click.ClickException)
        assert "PyPI unavailable" in str(result.exception)
        assert "AttributeError" not in str(result.exception)


class TestPublishPreflight:
    def test_publish_preflight_accepts_unused_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Publication can proceed only when all three namespaces are unused."""

        monkeypatch.setattr(
            release_checks,
            "find_publish_collisions",
            lambda *, version, remote, repository: [],
        )

        result = cli_runner.invoke(
            release_checks.app,
            ["publish-preflight", "--expected-version", "0.0.10b1"],
        )

        assert result.exit_code == 0
        assert "PyPI, origin, and GitHub: 0.0.10b1" in result.stdout

    def test_publish_preflight_reports_every_existing_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any existing package, tag, or release blocks publication."""

        monkeypatch.setattr(
            release_checks,
            "pypi_version_exists",
            lambda *, version: True,
        )
        monkeypatch.setattr(
            release_checks,
            "remote_tag_exists",
            lambda *, version, remote: True,
        )
        monkeypatch.setattr(
            release_checks,
            "github_release_exists",
            lambda *, version, repository: True,
        )

        result = cli_runner.invoke(
            release_checks.app,
            ["publish-preflight", "--expected-version", "0.0.10b1"],
        )

        assert result.exit_code == 2
        assert result.exception is not None
        assert result.exception.__context__ is not None
        message = str(result.exception.__context__)
        assert "PyPI version 0.0.10b1" in message
        assert "origin tag v0.0.10b1" in message
        assert "GitHub Release v0.0.10b1" in message

    @pytest.mark.parametrize(
        ("returncode", "expected"),
        [(0, True), (2, False)],
    )
    def test_remote_tag_exit_codes_are_classified(
        self,
        returncode: int,
        expected: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Git distinguishes an existing tag from an absent remote ref."""

        monkeypatch.setattr(
            release_checks,
            "_exec_command",
            lambda *, command: release_checks.CommandResult(
                returncode=returncode,
                stdout="",
                stderr="",
            ),
        )

        assert (
            release_checks.remote_tag_exists(version="0.0.10b1", remote="origin")
            is expected
        )

    @pytest.mark.parametrize(
        ("returncode", "stderr", "expected"),
        [(0, "", True), (1, "gh: Not Found (HTTP 404)", False)],
    )
    def test_github_release_exit_codes_are_classified(
        self,
        returncode: int,
        stderr: str,
        expected: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GitHub distinguishes an existing release from an absent tag release."""

        monkeypatch.setattr(
            release_checks,
            "_exec_command",
            lambda *, command: release_checks.CommandResult(
                returncode=returncode,
                stdout="",
                stderr=stderr,
            ),
        )

        assert (
            release_checks.github_release_exists(
                version="0.0.10b1", repository="fzyzcjy/torch_memory_saver"
            )
            is expected
        )


class TestCommandLogging:
    def test_run_command_records_command_output_and_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Evidence logging preserves command output and terminal status."""

        def complete_run(
            command: list[str],
            *,
            check: bool,
            stdout: object,
            stderr: int,
            text: bool,
            timeout: float | None,
        ) -> subprocess.CompletedProcess[str]:
            stdout.write("command output\n")
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(release_checks.subprocess, "run", complete_run)
        log_path = tmp_path / "evidence.log"

        result = cli_runner.invoke(
            release_checks.app,
            ["run", "--log-path", str(log_path), "--", "example", "--flag"],
        )

        assert result.exit_code == 0
        assert log_path.read_text(encoding="utf-8") == (
            "EXEC: example --flag\ncommand output\nRESULT: returncode=0\n"
        )

    def test_run_command_propagates_nonzero_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed evidence command keeps its return code and stops the workflow."""

        def fail_run(
            command: list[str],
            *,
            check: bool,
            stdout: object,
            stderr: int,
            text: bool,
            timeout: float | None,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=command, returncode=19)

        monkeypatch.setattr(release_checks.subprocess, "run", fail_run)
        log_path = tmp_path / "evidence.log"

        result = cli_runner.invoke(
            release_checks.app,
            ["run", "--log-path", str(log_path), "--", "example"],
        )

        assert result.exit_code == 19
        assert log_path.read_text(encoding="utf-8").endswith("RESULT: returncode=19\n")

    def test_run_command_records_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timed-out evidence command leaves a terminal timeout record."""

        def timeout_run(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=["example"], timeout=7)

        monkeypatch.setattr(release_checks.subprocess, "run", timeout_run)
        log_path = tmp_path / "evidence.log"

        result = cli_runner.invoke(
            release_checks.app,
            ["run", "--log-path", str(log_path), "--", "example"],
        )

        assert result.exit_code != 0
        assert log_path.read_text(encoding="utf-8").endswith(
            "RESULT: timeout=7 seconds\n"
        )


class TestPreUpload:
    def test_pre_upload_cli_dispatches_the_complete_gate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The public pre-upload command accepts every evidence boundary input."""

        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        manifest = tmp_path / "artifacts.sha256"
        manifest.touch()
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            release_checks,
            "run_pre_upload_checks",
            lambda **kwargs: calls.append(kwargs),
        )

        result = cli_runner.invoke(
            release_checks.app,
            [
                "pre-upload",
                "--dist-dir",
                str(dist_dir),
                "--manifest",
                str(manifest),
                "--expected-version",
                "0.0.10b1",
                "--log-path",
                str(tmp_path / "publish-recheck.log"),
                "--repo-root",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0]["expected_version"] == "0.0.10b1"
        assert calls[0]["dist_dir"] == dist_dir
        assert calls[0]["manifest"] == manifest

    def test_one_command_runs_every_pre_upload_gate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The combined gate checks source, artifacts, Twine, and namespaces."""

        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _write_release_artifacts(dist_dir=dist_dir, version="0.0.10b1")
        repo_root = tmp_path / "repo"
        _write_repo_source(repo_root=repo_root, version="0.0.10b1")
        manifest = tmp_path / "artifacts.sha256"
        release_checks.write_sha256_manifest(dist_dir=dist_dir, output=manifest)
        commands: list[list[str]] = []

        def complete_command(*, command: list[str], log_path: Path) -> int:
            commands.append(command)
            return 0

        monkeypatch.setattr(release_checks, "run_logged_command", complete_command)
        monkeypatch.setattr(
            release_checks,
            "find_publish_collisions",
            lambda *, version, remote, repository: [],
        )
        log_path = tmp_path / "publish-recheck.log"

        release_checks.run_pre_upload_checks(
            dist_dir=dist_dir,
            manifest=manifest,
            expected_version="0.0.10b1",
            log_path=log_path,
            repo_root=repo_root,
            remote="origin",
            repository="fzyzcjy/torch_memory_saver",
        )

        assert commands[0][:2] == ["bash", "-lc"]
        assert commands[0][2].startswith("set -euxo pipefail\n")
        assert commands[1][:4] == [sys.executable, "-m", "twine", "check"]
        evidence = log_path.read_text(encoding="utf-8")
        assert "release_checks version" in evidence
        assert "release_checks verify-manifest" in evidence
        assert "release_checks artifacts" in evidence
        assert "release_checks publish-preflight" in evidence
        assert evidence.count("RESULT: returncode=0") == 4

    def test_pre_upload_namespace_collision_is_logged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A late namespace collision fails closed with complete evidence."""

        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _write_release_artifacts(dist_dir=dist_dir, version="0.0.10b1")
        repo_root = tmp_path / "repo"
        _write_repo_source(repo_root=repo_root, version="0.0.10b1")
        manifest = tmp_path / "artifacts.sha256"
        release_checks.write_sha256_manifest(dist_dir=dist_dir, output=manifest)
        monkeypatch.setattr(
            release_checks,
            "run_logged_command",
            lambda *, command, log_path: 0,
        )
        monkeypatch.setattr(
            release_checks,
            "find_publish_collisions",
            lambda *, version, remote, repository: ["PyPI version 0.0.10b1"],
        )
        log_path = tmp_path / "publish-recheck.log"

        with pytest.raises(ValueError, match="Release identity already exists"):
            release_checks.run_pre_upload_checks(
                dist_dir=dist_dir,
                manifest=manifest,
                expected_version="0.0.10b1",
                log_path=log_path,
                repo_root=repo_root,
                remote="origin",
                repository="fzyzcjy/torch_memory_saver",
            )

        evidence = log_path.read_text(encoding="utf-8")
        assert "Traceback" in evidence
        assert "RESULT: error=ValueError" in evidence


class TestArtifactValidation:
    def test_complete_release_artifact_set_is_accepted(self, tmp_path: Path) -> None:
        """Both architecture wheels and the sdist satisfy the release gate."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")

        release_checks.validate_release_artifacts(
            dist_dir=tmp_path,
            expected_version="0.0.10b1",
            repo_root=tmp_path,
        )

    def test_missing_architecture_wheel_is_rejected(self, tmp_path: Path) -> None:
        """A release cannot pass without its aarch64 wheel."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_aarch64.whl"
        ).unlink()

        with pytest.raises(ValueError, match="missing=.*aarch64"):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    def test_unexpected_stale_artifact_is_rejected(self, tmp_path: Path) -> None:
        """Stale distributions cannot be included in a release upload."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        (tmp_path / "torch_memory_saver-0.0.9.tar.gz").touch()

        with pytest.raises(ValueError, match="unexpected=.*0.0.9"):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    def test_missing_cuda_binary_is_rejected(self, tmp_path: Path) -> None:
        """A wheel missing a CUDA runtime variant cannot pass validation."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        wheel_path = (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
        )
        _rewrite_wheel_without(
            wheel_path=wheel_path,
            removed_name="torch_memory_saver_hook_mode_torch_cu13.abi3.so",
        )

        with pytest.raises(
            ValueError, match="missing=.*torch_memory_saver_hook_mode_torch_cu13"
        ):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    def test_mislabeled_binary_architecture_is_rejected(self, tmp_path: Path) -> None:
        """A wheel tag cannot disagree with the architecture of its ELF binaries."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        wheel_path = (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_aarch64.whl"
        )
        _rewrite_wheel_file(
            wheel_path=wheel_path,
            target_name="torch_memory_saver_hook_mode_torch_cu13.abi3.so",
            content=_elf_header(architecture="x86_64"),
        )

        with pytest.raises(
            ValueError, match="ELF machine mismatch.*expected 183, found 62"
        ):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    def test_unsuffixed_binary_must_equal_cuda12_binary(self, tmp_path: Path) -> None:
        """The historical preload name remains a byte-identical CUDA 12 alias."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        wheel_path = (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
        )
        _rewrite_wheel_file(
            wheel_path=wheel_path,
            target_name="torch_memory_saver_hook_mode_preload.abi3.so",
            content=_elf_header(architecture="x86_64") + b"different",
        )

        with pytest.raises(ValueError, match="compatibility binary differs"):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    def test_missing_wheel_record_is_rejected(self, tmp_path: Path) -> None:
        """A wheel without RECORD cannot pass the artifact gate."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        wheel_path = (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
        )
        _rewrite_wheel_without_record_update(
            wheel_path=wheel_path,
            removed_suffix=".dist-info/RECORD",
        )

        with pytest.raises(ValueError, match="exactly one file ending.*RECORD"):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    @pytest.mark.parametrize("field", ["hash", "size"])
    def test_corrupt_wheel_record_entry_is_rejected(
        self,
        field: str,
        tmp_path: Path,
    ) -> None:
        """A stale RECORD hash or size cannot pass the artifact gate."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        wheel_path = (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
        )
        _corrupt_wheel_record(wheel_path=wheel_path, field=field)

        with pytest.raises(ValueError, match=f"RECORD {field} mismatch"):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    @pytest.mark.parametrize(
        ("corruption", "message"),
        [
            ("duplicate", "Duplicate RECORD path"),
            ("missing", "RECORD member set mismatch"),
            ("self", "RECORD self-entry must omit hash and size"),
        ],
    )
    def test_invalid_wheel_record_structure_is_rejected(
        self,
        corruption: str,
        message: str,
        tmp_path: Path,
    ) -> None:
        """RECORD must map each wheel member exactly once with an empty self-entry."""

        _write_release_artifacts(dist_dir=tmp_path, version="0.0.10b1")
        wheel_path = (
            tmp_path / "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
        )
        _corrupt_wheel_record(wheel_path=wheel_path, field=corruption)

        with pytest.raises(ValueError, match=message):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )

    def test_sha256_manifest_rejects_artifact_changed_after_review(
        self,
        tmp_path: Path,
    ) -> None:
        """The publish recheck rejects an artifact changed after human review."""

        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        artifact_path = dist_dir / "artifact.whl"
        artifact_path.write_bytes(b"approved")
        manifest = tmp_path / "artifacts.sha256"
        release_checks.write_sha256_manifest(dist_dir=dist_dir, output=manifest)

        release_checks.verify_sha256_manifest(
            dist_dir=dist_dir,
            manifest=manifest,
        )
        artifact_path.write_bytes(b"replaced")

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            release_checks.verify_sha256_manifest(
                dist_dir=dist_dir,
                manifest=manifest,
            )

    @pytest.mark.parametrize("change", ["add", "remove"])
    def test_sha256_manifest_rejects_artifact_set_changes(
        self,
        change: str,
        tmp_path: Path,
    ) -> None:
        """The approved manifest covers the exact final artifact filename set."""

        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        artifact_path = dist_dir / "artifact.whl"
        artifact_path.write_bytes(b"approved")
        manifest = tmp_path / "artifacts.sha256"
        release_checks.write_sha256_manifest(dist_dir=dist_dir, output=manifest)
        if change == "add":
            (dist_dir / "unexpected.tar.gz").write_bytes(b"unexpected")
        else:
            artifact_path.unlink()

        with pytest.raises(ValueError, match="manifest artifact set mismatch"):
            release_checks.verify_sha256_manifest(
                dist_dir=dist_dir,
                manifest=manifest,
            )

    def test_sha256_manifest_cannot_be_overwritten(self, tmp_path: Path) -> None:
        """A reviewed manifest cannot be silently regenerated in place."""

        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        (dist_dir / "artifact.whl").write_bytes(b"approved")
        manifest = tmp_path / "artifacts.sha256"
        release_checks.write_sha256_manifest(dist_dir=dist_dir, output=manifest)

        with pytest.raises(ValueError, match="Refusing to overwrite"):
            release_checks.write_sha256_manifest(dist_dir=dist_dir, output=manifest)

    def test_missing_backend_source_is_rejected(self, tmp_path: Path) -> None:
        """An sdist missing a platform backend source cannot pass validation."""

        _write_release_artifacts(
            dist_dir=tmp_path,
            version="0.0.10b1",
            omitted_source="csrc/hardware_xpu_support.cpp",
        )

        with pytest.raises(
            ValueError, match="missing backend sources:.*hardware_xpu_support.cpp"
        ):
            release_checks.validate_release_artifacts(
                dist_dir=tmp_path,
                expected_version="0.0.10b1",
                repo_root=tmp_path,
            )


def _write_release_artifacts(
    *,
    dist_dir: Path,
    version: str,
    omitted_source: str | None = None,
) -> None:
    csrc_dir = dist_dir / "csrc"
    csrc_dir.mkdir()
    for source_name in ("core.cpp", "core.h", "hardware_xpu_support.cpp"):
        (csrc_dir / source_name).write_text(source_name)

    for architecture in ("x86_64", "aarch64"):
        _write_wheel(
            wheel_path=dist_dir
            / f"torch_memory_saver-{version}-cp39-abi3-manylinux2014_{architecture}.whl",
            version=version,
            architecture=architecture,
        )
    _write_sdist(
        sdist_path=dist_dir / f"torch_memory_saver-{version}.tar.gz",
        version=version,
        repo_root=dist_dir,
        omitted_source=omitted_source,
    )


def _write_repo_source(*, repo_root: Path, version: str) -> None:
    repo_root.mkdir()
    (repo_root / "setup.py").write_text(
        f'from setuptools import setup\nsetup(version="{version}")\n',
        encoding="utf-8",
    )
    csrc_dir = repo_root / "csrc"
    csrc_dir.mkdir()
    for source_name in ("core.cpp", "core.h", "hardware_xpu_support.cpp"):
        (csrc_dir / source_name).write_text(source_name, encoding="utf-8")


def _write_wheel(*, wheel_path: Path, version: str, architecture: str) -> None:
    metadata_root = f"torch_memory_saver-{version}.dist-info"
    contents = {
        f"{metadata_root}/METADATA": (
            f"Name: torch-memory-saver\nVersion: {version}\n"
        ).encode(),
        f"{metadata_root}/WHEEL": (
            f"Wheel-Version: 1.0\nTag: cp39-abi3-manylinux2014_{architecture}\n"
        ).encode(),
        **{
            binary_name: _elf_header(architecture=architecture)
            for binary_name in release_checks._EXPECTED_BINARY_NAMES
        },
    }
    _write_wheel_contents(
        wheel_path=wheel_path,
        contents=contents,
        record_name=f"{metadata_root}/RECORD",
    )


def _write_sdist(
    *,
    sdist_path: Path,
    version: str,
    repo_root: Path,
    omitted_source: str | None,
) -> None:
    package_info = sdist_path.parent / "PKG-INFO"
    package_info.write_text(f"Name: torch-memory-saver\nVersion: {version}\n")
    with tarfile.open(sdist_path, mode="w:gz") as sdist:
        sdist.add(package_info, arcname=f"torch_memory_saver-{version}/PKG-INFO")
        sdist.add(
            package_info,
            arcname=f"torch_memory_saver-{version}/torch_memory_saver.egg-info/PKG-INFO",
        )
        for source_path in sorted((repo_root / "csrc").iterdir()):
            relative_path = source_path.relative_to(repo_root).as_posix()
            if relative_path != omitted_source:
                sdist.add(
                    source_path,
                    arcname=f"torch_memory_saver-{version}/{relative_path}",
                )
    package_info.unlink()


def _rewrite_wheel_without(*, wheel_path: Path, removed_name: str) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        record_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")
        )
        contents = {
            name: wheel.read(name)
            for name in wheel.namelist()
            if name not in {removed_name, record_name}
        }
    _write_wheel_contents(
        wheel_path=wheel_path,
        contents=contents,
        record_name=record_name,
    )


def _rewrite_wheel_file(*, wheel_path: Path, target_name: str, content: bytes) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        record_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")
        )
        contents = {
            name: wheel.read(name) for name in wheel.namelist() if name != record_name
        }
    contents[target_name] = content
    _write_wheel_contents(
        wheel_path=wheel_path,
        contents=contents,
        record_name=record_name,
    )


def _write_wheel_contents(
    *,
    wheel_path: Path,
    contents: dict[str, bytes],
    record_name: str,
) -> None:
    record_output = StringIO()
    writer = csv.writer(record_output, lineterminator="\n")
    for name, content in sorted(contents.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        writer.writerow((name, f"sha256={digest}", str(len(content))))
    writer.writerow((record_name, "", ""))

    with zipfile.ZipFile(wheel_path, mode="w") as wheel:
        for name, content in contents.items():
            wheel.writestr(name, content)
        wheel.writestr(record_name, record_output.getvalue())


def _rewrite_wheel_without_record_update(
    *,
    wheel_path: Path,
    removed_suffix: str,
) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        contents = {
            name: wheel.read(name)
            for name in wheel.namelist()
            if not name.endswith(removed_suffix)
        }
    with zipfile.ZipFile(wheel_path, mode="w") as wheel:
        for name, content in contents.items():
            wheel.writestr(name, content)


def _corrupt_wheel_record(*, wheel_path: Path, field: str) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        contents = {name: wheel.read(name) for name in wheel.namelist()}
        record_name = next(
            name for name in contents if name.endswith(".dist-info/RECORD")
        )
    rows = list(csv.reader(StringIO(contents[record_name].decode())))
    target_row = next(row for row in rows if row[0].endswith(".dist-info/WHEEL"))
    if field in {"hash", "size"}:
        target_row[1 if field == "hash" else 2] = "broken"
    elif field == "duplicate":
        rows.append(target_row.copy())
    elif field == "missing":
        rows.remove(target_row)
    elif field == "self":
        next(row for row in rows if row[0] == record_name)[1] = "sha256=broken"
    else:
        raise ValueError(f"Unknown RECORD corruption: {field}")
    record_output = StringIO()
    csv.writer(record_output, lineterminator="\n").writerows(rows)
    contents[record_name] = record_output.getvalue().encode()

    with zipfile.ZipFile(wheel_path, mode="w") as wheel:
        for name, content in contents.items():
            wheel.writestr(name, content)


def _elf_header(*, architecture: str) -> bytes:
    binary = bytearray(20)
    binary[:7] = b"\x7fELF\x02\x01\x01"
    binary[18:20] = release_checks._ELF_MACHINE_BY_ARCHITECTURE[architecture].to_bytes(
        length=2, byteorder="little"
    )
    return bytes(binary)
