from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner


def _load_gpu_validation_module() -> ModuleType:
    module_path = Path(__file__).with_name("gpu_validation.py")
    spec = importlib.util.spec_from_file_location("gpu_validation", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gpu_validation = _load_gpu_validation_module()
cli_runner = CliRunner()


class TestGpuValidationCommand:
    def test_command_builds_local_workstation_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI builds a local Docker config without a remote option."""

        release_root = tmp_path / "release"
        artifact_root = tmp_path / "artifacts"
        release_root.mkdir()
        artifact_root.mkdir()
        configs: list[gpu_validation.GpuValidationConfig] = []

        def record_config(*, config: gpu_validation.GpuValidationConfig) -> None:
            configs.append(config)

        monkeypatch.setattr(
            gpu_validation,
            "run_gpu_validation",
            record_config,
        )

        result = cli_runner.invoke(
            gpu_validation.app,
            [
                "--release-root",
                str(release_root),
                "--artifact-root",
                str(artifact_root),
                "--expected-version",
                "0.0.10b1",
            ],
        )

        assert result.exit_code == 0
        assert len(configs) == 1
        assert configs[0].release_root == release_root
        assert configs[0].run_dir.parent == artifact_root
        assert "--remote" not in result.output

    def test_matrix_maps_each_architecture_and_cuda_runtime_exactly(self) -> None:
        """Every published binary family has one matching isolated GPU gate."""

        assert [
            (
                case.name,
                case.image,
                case.architecture,
                case.cuda_major,
                case.server_image,
                case.test_environment,
            )
            for case in gpu_validation._GPU_VALIDATION_CASES
        ] == [
            (
                "x86_64-cuda12-gpu",
                "docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
                "x86_64",
                "12",
                None,
                (),
            ),
            (
                "x86_64-cuda13-gpu",
                "docker.io/pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime",
                "x86_64",
                "13",
                None,
                (),
            ),
            (
                "aarch64-cuda12-gpu",
                "ghcr.io/lupinemachines/lupine-pytorch-worker@sha256:f152fbcb3e2eda5661abafdfb4f3024afe9b775eb702015b5df0527ca0b7f556",
                "aarch64",
                "12",
                "ghcr.io/lupinemachines/lupine-server@sha256:e6b1103392165e929ca7f4f910eeec9f0b0f7155c5ff65b7402ca083c6bf9d53",
                (("TMS_TEST_LUPINE", "1"),),
            ),
            (
                "aarch64-cuda13-gpu",
                "ghcr.io/lupinemachines/lupine-pytorch-worker@sha256:a154530bfeed825e8c915be1b6129965f293dbf29ce4764441014dccf9c6c08a",
                "aarch64",
                "13",
                "ghcr.io/lupinemachines/lupine-server@sha256:f1d805e14e0b2da5d5912adeed72f3e1c7d0458082c3cbaf3ba9bb2346b869cd",
                (("TMS_TEST_LUPINE", "1"),),
            ),
        ]

    @pytest.mark.parametrize(
        "script",
        [
            gpu_validation._X86_VALIDATION_SCRIPT,
            gpu_validation._ARM_VALIDATION_SCRIPT,
        ],
    )
    def test_gpu_scripts_run_runtime_suite_from_installed_wheel(
        self, script: str
    ) -> None:
        """Each architecture installs the final wheel and exercises runtime tests."""

        assert "pip install" in script
        assert "--no-deps" in script
        assert "test_configure_subprocess.py" in script
        assert "test_examples.py" in script
        assert "test_utils.py" in script
        assert "pytest test -vv -ra" in script
        assert "timeout --signal=TERM --kill-after=30s 3600s" in script
        assert "torch.cuda.is_available()" in script
        assert "get_device_name(0)" in script
        assert '"4090 D" in name or "4090D" in name' in script
        assert "site-packages" in script
        assert "/workspace/scripts" not in script
        assert "test_merge_cuda_wheels.py" not in script

    def test_gpu_scripts_do_not_deselect_runtime_tests(self) -> None:
        """Runtime-specific skips remain visible in pytest output."""

        assert "--deselect" not in gpu_validation._X86_VALIDATION_SCRIPT
        assert "--deselect" not in gpu_validation._ARM_VALIDATION_SCRIPT

    def test_lupine_runtime_tests_declare_exact_skip_markers(self) -> None:
        """Only the two host-memory tests opt out under Lupine."""

        test_path = Path(__file__).parents[4] / "test" / "test_examples.py"
        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        marked_tests = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(decorator, ast.Name)
                and decorator.id == "_skip_on_lupine"
                for decorator in node.decorator_list
            )
        }

        assert 'os.environ.get("TMS_TEST_LUPINE") == "1"' in source
        assert marked_tests == {
            "test_cpu_backup_preload_backend_from_env",
            "test_disk_backup",
        }

    def test_arm_script_requires_real_aarch64_gpu_execution(self) -> None:
        """The ARM gate proves architecture, CUDA major, ELF type, and GPU access."""

        script = gpu_validation._ARM_VALIDATION_SCRIPT

        assert 'platform.machine() == "aarch64"' in script
        assert '{"site-packages", "dist-packages"}' in script
        assert "pytest==8.3.5" in script
        assert "numpy==2.2.3" in script
        assert "nvidia-ml-py==12.570.86" in script
        assert "--only-binary=:all:" in script
        assert "== 183" in script
        assert "libcuda.so.1" not in script
        assert "cuda-stubs" not in script

    def test_cli_reports_validation_failure_without_secondary_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed GPU command is rendered as one actionable CLI error."""

        release_root = tmp_path / "release"
        artifact_root = tmp_path / "artifacts"
        release_root.mkdir()
        artifact_root.mkdir()

        def fail_validation(*, config: gpu_validation.GpuValidationConfig) -> None:
            raise subprocess.CalledProcessError(returncode=7, cmd=["docker", "run"])

        monkeypatch.setattr(
            gpu_validation,
            "run_gpu_validation",
            fail_validation,
        )

        result = cli_runner.invoke(
            gpu_validation.app,
            [
                "--release-root",
                str(release_root),
                "--artifact-root",
                str(artifact_root),
                "--expected-version",
                "0.0.10b1",
            ],
        )

        assert result.exit_code == 1
        assert isinstance(
            result.exception,
            gpu_validation.click.ClickException,
        )
        message = str(result.exception)
        assert "returned non-zero exit status 7" in message
        assert "logs:" in message
        assert "AttributeError" not in message

    def test_runtime_test_classification_accepts_complete_inventory(
        self, tmp_path: Path
    ) -> None:
        """Every repository test module is explicitly runtime or build tooling."""

        _write_test_inventory(release_root=tmp_path)

        gpu_validation._require_runtime_test_contract(release_root=tmp_path)

    def test_runtime_test_classification_rejects_new_unclassified_module(
        self, tmp_path: Path
    ) -> None:
        """A new test cannot silently escape clean-install GPU validation."""

        _write_test_inventory(release_root=tmp_path)
        (tmp_path / "test" / "test_new_runtime.py").touch()

        with pytest.raises(ValueError, match="unclassified=.*test_new_runtime.py"):
            gpu_validation._require_runtime_test_contract(release_root=tmp_path)

    def test_missing_wheel_fails_before_docker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing final wheel fails before any Docker command runs."""

        release_root = tmp_path / "release"
        artifact_root = tmp_path / "artifacts"
        release_root.mkdir()
        artifact_root.mkdir()
        config = gpu_validation._build_config(
            release_root=release_root,
            artifact_root=artifact_root,
            expected_version="0.0.10b1",
            proxy_url="http://127.0.0.1:7890",
        )
        monkeypatch.setattr(gpu_validation.platform, "machine", lambda: "x86_64")

        with pytest.raises(ValueError, match="Missing release wheels"):
            gpu_validation.run_gpu_validation(config=config)

    def test_validation_dispatches_all_four_gpu_cells(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The harness dispatches both direct and both Lupine GPU cells."""

        config = _write_config_with_wheels(tmp_path=tmp_path)
        dispatched: list[tuple[str, str]] = []

        def complete_command(
            *,
            command: list[str],
            log_path: Path,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=command, returncode=0)

        def record_x86(
            *,
            config: gpu_validation.GpuValidationConfig,
            case: gpu_validation.GpuValidationCase,
        ) -> None:
            dispatched.append(("x86", case.name))

        def record_lupine(
            *,
            config: gpu_validation.GpuValidationConfig,
            case: gpu_validation.GpuValidationCase,
        ) -> None:
            dispatched.append(("lupine", case.name))

        monkeypatch.setattr(gpu_validation.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(gpu_validation, "_exec_command", complete_command)
        monkeypatch.setattr(gpu_validation, "_run_x86_case", record_x86)
        monkeypatch.setattr(gpu_validation, "_run_lupine_case", record_lupine)

        gpu_validation.run_gpu_validation(config=config)

        assert dispatched == [
            ("x86", "x86_64-cuda12-gpu"),
            ("x86", "x86_64-cuda13-gpu"),
            ("lupine", "aarch64-cuda12-gpu"),
            ("lupine", "aarch64-cuda13-gpu"),
        ]

    def test_x86_runner_uses_fresh_foreground_gpu_container(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Direct GPU validation is isolated, read-only, and foreground."""

        config = _write_config_with_wheels(tmp_path=tmp_path)
        case = gpu_validation._GPU_VALIDATION_CASES[0]
        commands = _record_commands(monkeypatch=monkeypatch)

        gpu_validation._run_x86_case(config=config, case=case)

        run_command = commands[0][0]
        assert run_command[:3] == ["docker", "run", "--rm"]
        assert ["--pull", "missing"] == run_command[
            run_command.index("--pull") : run_command.index("--pull") + 2
        ]
        assert "--detach" not in run_command
        assert "--name" in run_command
        assert ["--gpus", "device=0"] == run_command[
            run_command.index("--gpus") : run_command.index("--gpus") + 2
        ]
        assert f"{config.release_root}:/workspace:ro" in run_command
        assert f"TMS_CUDA_MAJOR={case.cuda_major}" in run_command
        assert "TMS_TEST_LUPINE=1" not in run_command
        assert commands[1][0][:4] == ["docker", "image", "inspect", case.image]
        assert "org.opencontainers.image.revision" in commands[1][0][-1]
        assert commands[-1][0][:4] == ["docker", "container", "rm", "--force"]

    def test_x86_runner_cleans_named_container_after_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A timed-out direct GPU gate still removes its named container."""

        config = _write_config_with_wheels(tmp_path=tmp_path)
        case = gpu_validation._GPU_VALIDATION_CASES[1]
        commands: list[tuple[list[str], bool]] = []

        def timeout_gpu_run(
            *,
            command: list[str],
            log_path: Path,
            check: bool = True,
            input_text: str | None = None,
        ) -> subprocess.CompletedProcess[str]:
            commands.append((command, check))
            if command[:3] == ["docker", "run", "--rm"]:
                raise subprocess.TimeoutExpired(
                    cmd=command,
                    timeout=gpu_validation._COMMAND_TIMEOUT_SECONDS,
                )
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(gpu_validation, "_exec_command", timeout_gpu_run)

        with pytest.raises(subprocess.TimeoutExpired):
            gpu_validation._run_x86_case(config=config, case=case)

        resource_prefix = gpu_validation._resource_prefix(
            config=config,
            case=case,
        )
        assert commands[-2][0][:4] == ["docker", "image", "inspect", case.image]
        assert commands[-2][1] is False
        assert commands[-1] == (
            [
                "docker",
                "container",
                "rm",
                "--force",
                f"{resource_prefix}-validation",
            ],
            False,
        )

    def test_lupine_runner_uses_published_worker_and_private_network(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ARM validation connects matched images without exposing the RPC port."""

        config = _write_config_with_wheels(tmp_path=tmp_path)
        case = gpu_validation._GPU_VALIDATION_CASES[2]
        commands = _record_commands(monkeypatch=monkeypatch)

        gpu_validation._run_lupine_case(config=config, case=case)

        command_lists = [command for command, _ in commands]
        server_command = next(
            command
            for command in command_lists
            if command[:3] == ["docker", "run", "--rm"] and "--detach" in command
        )
        client_commands = [
            command
            for command in command_lists
            if command[:3] == ["docker", "run", "--rm"]
            and "--platform" in command
            and "linux/arm64" in command
        ]

        assert command_lists[0][:3] == ["docker", "network", "create"]
        assert command_lists[1] == server_command
        assert command_lists[2][:4] == [
            "docker",
            "image",
            "inspect",
            case.server_image,
        ]
        assert command_lists[3] == client_commands[0]
        assert command_lists[4][:4] == [
            "docker",
            "image",
            "inspect",
            case.image,
        ]
        assert command_lists[5] == client_commands[1]
        assert not any(command[:2] == ["docker", "build"] for command in command_lists)
        assert not any(command[:2] == ["docker", "pull"] for command in command_lists)
        assert case.server_image in server_command
        assert ["--gpus", "device=0"] == server_command[-3:-1]
        assert "-p" not in server_command
        assert "--publish" not in server_command
        assert len(client_commands) == 2
        for command in [server_command, *client_commands]:
            pull_index = command.index("--pull")
            assert command[pull_index : pull_index + 2] == ["--pull", "missing"]
        for command in client_commands:
            add_host_index = command.index("--add-host")
            assert command[add_host_index : add_host_index + 2] == [
                "--add-host",
                "host.docker.internal:host-gateway",
            ]
        assert "--name" in client_commands[0]
        assert "--name" in client_commands[1]
        assert f"{config.release_root}:/workspace:ro" not in client_commands[0]
        assert f"{config.release_root}:/workspace:ro" in client_commands[1]
        assert f"TMS_CUDA_MAJOR={case.cuda_major}" in client_commands[1]
        assert "TMS_TEST_LUPINE=1" in client_commands[1]
        assert "http_proxy=http://host.docker.internal:7890" in client_commands[1]
        assert "https_proxy=http://host.docker.internal:7890" in client_commands[1]
        assert client_commands[0][-4] == case.image
        assert client_commands[1][-4] == case.image
        assert (
            "timeout --signal=TERM --kill-after=5s 20s python3 -c"
            in client_commands[0][-1]
        )

    def test_lupine_runner_records_worker_identity_after_smoke_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed worker smoke keeps its error and records the pulled image."""

        config = _write_config_with_wheels(tmp_path=tmp_path)
        case = gpu_validation._GPU_VALIDATION_CASES[2]
        commands: list[tuple[list[str], bool]] = []

        def fail_smoke(
            *,
            command: list[str],
            log_path: Path,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            commands.append((command, check))
            if command[:3] == ["docker", "run", "--rm"] and command[-1].startswith(
                "for attempt in"
            ):
                raise subprocess.CalledProcessError(returncode=9, cmd=command)
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(gpu_validation, "_exec_command", fail_smoke)

        with pytest.raises(subprocess.CalledProcessError) as error:
            gpu_validation._run_lupine_case(config=config, case=case)

        assert error.value.returncode == 9
        smoke_index = next(
            index
            for index, (command, _) in enumerate(commands)
            if command[:3] == ["docker", "run", "--rm"]
            and command[-1].startswith("for attempt in")
        )
        assert commands[smoke_index + 1][0][:4] == [
            "docker",
            "image",
            "inspect",
            case.image,
        ]
        assert commands[smoke_index + 1][1] is False

    @pytest.mark.parametrize(
        ("proxy_url", "expected"),
        [
            (
                "http://127.0.0.1:7890",
                "http://host.docker.internal:7890",
            ),
            (
                "http://localhost:7890",
                "http://host.docker.internal:7890",
            ),
            (
                "http://proxy.example.com:8080",
                "http://proxy.example.com:8080",
            ),
        ],
    )
    def test_arm_proxy_resolves_host_loopback_from_bridge_network(
        self,
        proxy_url: str,
        expected: str,
    ) -> None:
        """ARM dependency installs can reach the configured host proxy."""

        assert gpu_validation._host_gateway_proxy_url(proxy_url=proxy_url) == expected

    def test_loopback_proxy_without_port_is_rejected(self) -> None:
        """A loopback proxy without a reachable host port fails before Docker."""

        with pytest.raises(ValueError, match="requires a port"):
            gpu_validation._host_gateway_proxy_url(proxy_url="http://127.0.0.1")

    def test_lupine_runner_cleans_resources_after_client_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Server and private network are removed after a failed ARM gate."""

        config = _write_config_with_wheels(tmp_path=tmp_path)
        case = gpu_validation._GPU_VALIDATION_CASES[3]
        commands: list[tuple[list[str], str | None, bool]] = []

        def fail_final_client(
            *,
            command: list[str],
            log_path: Path,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            commands.append((command, None, check))
            if (
                command[:3] == ["docker", "run", "--rm"]
                and command[-1] == gpu_validation._ARM_VALIDATION_SCRIPT
            ):
                raise subprocess.CalledProcessError(returncode=1, cmd=command)
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(gpu_validation, "_exec_command", fail_final_client)

        with pytest.raises(subprocess.CalledProcessError):
            gpu_validation._run_lupine_case(config=config, case=case)

        resource_prefix = gpu_validation._resource_prefix(
            config=config,
            case=case,
        )
        assert [entry[0][:4] for entry in commands[-4:]] == [
            ["docker", "container", "rm", "--force"],
            ["docker", "container", "rm", "--force"],
            ["docker", "container", "rm", "--force"],
            ["docker", "network", "rm", commands[-1][0][3]],
        ]
        assert [entry[0][-1] for entry in commands[-4:]] == [
            f"{resource_prefix}-smoke",
            f"{resource_prefix}-validation",
            f"{resource_prefix}-server",
            f"{resource_prefix}-network",
        ]
        assert [entry[2] for entry in commands[-4:]] == [
            False,
            False,
            False,
            False,
        ]

    def test_exec_command_applies_orchestration_timeout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every external Docker command has a process-level hard timeout."""

        captured_timeouts: list[float | None] = []

        def complete_run(
            command: list[str],
            *,
            check: bool,
            stdout: object,
            stderr: int,
            text: bool,
            timeout: float | None,
        ) -> subprocess.CompletedProcess[str]:
            captured_timeouts.append(timeout)
            return subprocess.CompletedProcess(args=command, returncode=0)

        monkeypatch.setattr(gpu_validation.subprocess, "run", complete_run)

        gpu_validation._exec_command(
            command=["docker", "info"],
            log_path=tmp_path / "command.log",
        )

        assert captured_timeouts == [gpu_validation._COMMAND_TIMEOUT_SECONDS]

    def test_best_effort_image_inspect_suppresses_timeout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Failure-path image inspection cannot replace the primary error."""

        def timeout_inspect(
            *,
            command: list[str],
            log_path: Path,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(command, timeout=1)

        monkeypatch.setattr(gpu_validation, "_exec_command", timeout_inspect)

        gpu_validation._inspect_image(
            image="example.invalid/image:latest",
            log_path=tmp_path / "inspect.log",
            check=False,
        )

        with pytest.raises(subprocess.TimeoutExpired):
            gpu_validation._inspect_image(
                image="example.invalid/image:latest",
                log_path=tmp_path / "inspect.log",
            )


def _write_config_with_wheels(*, tmp_path: Path) -> gpu_validation.GpuValidationConfig:
    release_root = tmp_path / "release"
    artifact_root = tmp_path / "artifacts"
    dist_dir = release_root / "dist"
    dist_dir.mkdir(parents=True)
    artifact_root.mkdir()
    for architecture in ("x86_64", "aarch64"):
        (
            dist_dir
            / f"torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_{architecture}.whl"
        ).touch()
    _write_test_inventory(release_root=release_root)
    return gpu_validation._build_config(
        release_root=release_root,
        artifact_root=artifact_root,
        expected_version="0.0.10b1",
        proxy_url="http://127.0.0.1:7890",
    )


def _write_test_inventory(*, release_root: Path) -> None:
    test_dir = release_root / "test"
    (test_dir / "examples").mkdir(parents=True)
    for name in (
        *gpu_validation._RUNTIME_TEST_MODULES,
        *gpu_validation._BUILD_TOOL_TEST_MODULES,
    ):
        (test_dir / name).touch()


def _record_commands(
    *, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[list[str], str | None]]:
    commands: list[tuple[list[str], str | None]] = []

    def record_command(
        *,
        command: list[str],
        log_path: Path,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        commands.append((command, None))
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(gpu_validation, "_exec_command", record_command)
    return commands
