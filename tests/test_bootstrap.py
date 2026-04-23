"""Bootstrap entry-point tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bootstrap
from core.types import DiscoveryResult, FirewallStatus, PlatformInfo
from setup import ProjectEnvironmentStatus


class BootstrapTests(unittest.TestCase):
    """Validate bootstrap's benchmark auto-run and happy path."""

    def test_build_parser_accepts_retest_and_rebuild_flags(self) -> None:
        args = bootstrap.build_parser().parse_args(["--retest", "--rebuild", "--verbose"])

        self.assertTrue(args.retest)
        self.assertTrue(args.rebuild)
        self.assertTrue(args.verbose)
        self.assertEqual(args.log_start_mode, "normal")

    def test_apply_log_start_mode_archives_previous_logs(self) -> None:
        archive_path = bootstrap.PROJECT_ROOT / "logs" / "logs-archive-20260417-120000.zip"

        with mock.patch("bootstrap.archive_existing_logs", return_value=(archive_path, 3)) as archive_mock:
            result = bootstrap._apply_log_start_mode("clean")

        self.assertEqual(
            result,
            (
                bootstrap.logging.INFO,
                "Log start mode clean archived 3 previous log files into logs/logs-archive-20260417-120000.zip.",
            ),
        )
        archive_mock.assert_called_once_with()

    def test_apply_log_start_mode_cleanses_previous_logs(self) -> None:
        with mock.patch("bootstrap.cleanse_existing_logs", return_value=4) as cleanse_mock:
            result = bootstrap._apply_log_start_mode("cleanse")

        self.assertEqual(
            result,
            (bootstrap.logging.INFO, "Log start mode cleanse removed 4 previous log artifacts."),
        )
        cleanse_mock.assert_called_once_with()

    def test_ensure_bootstrap_runtime_environment_runs_setup_then_relaunches_when_env_missing(self) -> None:
        logger = mock.Mock()

        with (
            mock.patch(
                "bootstrap.inspect_project_environment",
                side_effect=[
                    ProjectEnvironmentStatus(
                        venv_exists=False,
                        requirements_current=False,
                        using_project_python=False,
                    ),
                    ProjectEnvironmentStatus(
                        venv_exists=True,
                        requirements_current=True,
                        using_project_python=False,
                    ),
                ],
            ),
            mock.patch("bootstrap.project_python_path", return_value=Path("C:/venv/python.exe")),
            mock.patch(
                "bootstrap.subprocess.run",
                side_effect=[
                    mock.Mock(returncode=0),
                    mock.Mock(returncode=0),
                ],
            ) as run_mock,
        ):
            result = bootstrap.ensure_bootstrap_runtime_environment(logger, [])

        self.assertEqual(result, 0)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args_list[0].args[0], bootstrap._setup_command())
        self.assertEqual(
            run_mock.call_args_list[1].args[0],
            [
                str(Path("C:/venv/python.exe")),
                "-X",
                "utf8",
                str(bootstrap.PROJECT_ROOT / "bootstrap.py"),
                "--elevate-if-needed",
            ],
        )

    def test_ensure_bootstrap_runtime_environment_stops_when_setup_fails(self) -> None:
        logger = mock.Mock()

        with (
            mock.patch(
                "bootstrap.inspect_project_environment",
                return_value=ProjectEnvironmentStatus(
                    venv_exists=False,
                    requirements_current=False,
                    using_project_python=False,
                ),
            ),
            mock.patch(
                "bootstrap.subprocess.run",
                return_value=mock.Mock(returncode=7),
            ) as run_mock,
        ):
            result = bootstrap.ensure_bootstrap_runtime_environment(logger, [])

        self.assertEqual(result, 7)
        run_mock.assert_called_once_with(
            bootstrap._setup_command(),
            check=False,
            cwd=bootstrap.PROJECT_ROOT,
        )
        logger.error.assert_called()

    def test_ensure_bootstrap_runtime_environment_relaunches_with_project_python(self) -> None:
        logger = mock.Mock()

        with (
            mock.patch(
                "bootstrap.inspect_project_environment",
                return_value=ProjectEnvironmentStatus(
                    venv_exists=True,
                    requirements_current=True,
                    using_project_python=False,
                ),
            ),
            mock.patch("bootstrap.project_python_path", return_value=Path("C:/venv/python.exe")),
            mock.patch("bootstrap.subprocess.run", return_value=mock.Mock(returncode=0)) as run_mock,
        ):
            result = bootstrap.ensure_bootstrap_runtime_environment(logger, ["--role", "discover"])

        self.assertEqual(result, 0)
        run_mock.assert_called_once()
        self.assertEqual(
            run_mock.call_args.args[0],
            [
                str(Path("C:/venv/python.exe")),
                "-X",
                "utf8",
                str(bootstrap.PROJECT_ROOT / "bootstrap.py"),
                "--role",
                "discover",
                "--elevate-if-needed",
            ],
        )

    def test_runtime_relaunch_argv_appends_elevation_flag_once(self) -> None:
        self.assertEqual(
            bootstrap._runtime_relaunch_argv(["--role", "discover"]),
            ["--role", "discover", "--elevate-if-needed"],
        )
        self.assertEqual(
            bootstrap._runtime_relaunch_argv(["--role", "discover", "--retest", "--elevate-if-needed"]),
            ["--role", "discover", "--retest", "--elevate-if-needed"],
        )

    def test_ensure_compute_node_benchmark_ready_runs_benchmark_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result_path = temp_root / "compute_node" / "performance_metrics" / "result.json"
            script_path = temp_root / "compute_node" / "performance_metrics" / "benchmark.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("# placeholder\n", encoding="utf-8")
            logger = mock.Mock()

            def fake_run(*_args, **_kwargs) -> int:
                result_path.write_text("{}", encoding="utf-8")
                return 0

            with (
                mock.patch.object(bootstrap, "PROJECT_ROOT", temp_root),
                mock.patch.object(bootstrap, "BENCHMARK_SCRIPT_PATH", script_path),
                mock.patch.object(bootstrap, "BENCHMARK_RESULT_PATH", result_path),
                mock.patch("bootstrap._run_streaming_command", side_effect=fake_run) as run_mock,
            ):
                ready = bootstrap.ensure_compute_node_benchmark_ready(logger)

        self.assertTrue(ready)
        run_mock.assert_called_once()
        logger.warning.assert_called_once()
        logger.info.assert_called()

    def test_ensure_compute_node_benchmark_ready_retests_datasets_before_rerunning_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result_path = temp_root / "compute_node" / "performance_metrics" / "result.json"
            benchmark_script_path = temp_root / "compute_node" / "performance_metrics" / "benchmark.py"
            dataset_script_path = temp_root / "compute_node" / "input_matrix" / "generate.py"
            benchmark_script_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_script_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark_script_path.write_text("# placeholder\n", encoding="utf-8")
            dataset_script_path.write_text("# placeholder\n", encoding="utf-8")
            logger = mock.Mock()

            def fake_dataset_run(command, **kwargs):
                del kwargs
                self.assertIn(str(dataset_script_path), command)
                return 0

            def fake_benchmark_run(command, **kwargs):
                del kwargs
                self.assertIn(str(benchmark_script_path), command)
                result_path.write_text("{\"ok\": true}", encoding="utf-8")
                return 0

            with (
                mock.patch.object(bootstrap, "PROJECT_ROOT", temp_root),
                mock.patch.object(bootstrap, "BENCHMARK_SCRIPT_PATH", benchmark_script_path),
                mock.patch.object(bootstrap, "BENCHMARK_RESULT_PATH", result_path),
                mock.patch.object(bootstrap, "INPUT_MATRIX_SCRIPT_PATH", dataset_script_path),
                mock.patch("bootstrap._run_passthrough_command", side_effect=fake_dataset_run) as dataset_run_mock,
                mock.patch("bootstrap._run_streaming_command", side_effect=fake_benchmark_run) as benchmark_run_mock,
            ):
                ready = bootstrap.ensure_compute_node_benchmark_ready(logger, force_retest=True)

        self.assertTrue(ready)
        dataset_run_mock.assert_called_once()
        benchmark_run_mock.assert_called_once()
        self.assertEqual(dataset_run_mock.call_args.args[0][3], str(dataset_script_path))
        self.assertEqual(dataset_run_mock.call_args.args[0][-1], "--force")
        self.assertEqual(benchmark_run_mock.call_args.args[0][3], str(benchmark_script_path))
        self.assertNotIn("--rebuild", benchmark_run_mock.call_args.args[0])

    def test_ensure_compute_node_benchmark_ready_rebuilds_benchmark_binaries_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result_path = temp_root / "compute_node" / "performance_metrics" / "result.json"
            benchmark_script_path = temp_root / "compute_node" / "performance_metrics" / "benchmark.py"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{\"cached\": true}", encoding="utf-8")
            benchmark_script_path.write_text("# placeholder\n", encoding="utf-8")
            logger = mock.Mock()

            def fake_run(command, **kwargs):
                del kwargs
                if str(benchmark_script_path) in command:
                    result_path.write_text("{\"rebuilt\": true}", encoding="utf-8")
                return 0

            with (
                mock.patch.object(bootstrap, "PROJECT_ROOT", temp_root),
                mock.patch.object(bootstrap, "BENCHMARK_SCRIPT_PATH", benchmark_script_path),
                mock.patch.object(bootstrap, "BENCHMARK_RESULT_PATH", result_path),
                mock.patch("bootstrap._run_passthrough_command") as dataset_run_mock,
                mock.patch("bootstrap._run_streaming_command", side_effect=fake_run) as run_mock,
            ):
                ready = bootstrap.ensure_compute_node_benchmark_ready(
                    logger,
                    force_rebuild=True,
                )

        self.assertTrue(ready)
        dataset_run_mock.assert_not_called()
        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.args[0][3], str(benchmark_script_path))
        self.assertIn("--rebuild", run_mock.call_args.args[0])

    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=None)
    @mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=False)
    def test_main_stops_when_benchmark_auto_run_fails(
        self,
        benchmark_ready_mock: mock.Mock,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
    ) -> None:
        configure_logging_mock.return_value = mock.Mock()
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
            hostname="mdong-zephy14",
            is_wsl=False,
            is_admin=False,
            can_elevate=True,
        )

        result = bootstrap.main([])

        self.assertEqual(result, 1)
        runtime_environment_mock.assert_called_once()
        benchmark_ready_mock.assert_called_once_with(configure_logging_mock.return_value, force_retest=False, force_rebuild=False, verbose=False)
        detect_os_mock.assert_called_once()

    @mock.patch("bootstrap._load_supervisor_class")
    @mock.patch("bootstrap.ensure_rules")
    @mock.patch("bootstrap.relaunch_as_admin", return_value=False)
    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=None)
    @mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=True)
    def test_main_continues_when_benchmark_result_exists(
        self,
        benchmark_ready_mock: mock.Mock,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
        relaunch_as_admin_mock: mock.Mock,
        ensure_rules_mock: mock.Mock,
        load_supervisor_class_mock: mock.Mock,
    ) -> None:
        del relaunch_as_admin_mock
        logger = mock.Mock()
        configure_logging_mock.return_value = logger
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
            hostname="mdong-zephy14",
            is_wsl=False,
            is_admin=False,
            can_elevate=True,
        )
        ensure_rules_mock.return_value = FirewallStatus(
            supported=True,
            applied=False,
            needs_admin=False,
            backend="windows",
            message="ok",
        )
        supervisor_cls_mock = mock.Mock()
        load_supervisor_class_mock.return_value = supervisor_cls_mock
        supervisor_mock = supervisor_cls_mock.return_value
        supervisor_mock.run.return_value = DiscoveryResult(
            success=True,
            peer_address="10.0.0.5",
            peer_port=52020,
            source="mdns",
            message="ready",
        )

        result = bootstrap.main([])

        self.assertEqual(result, 0)
        runtime_environment_mock.assert_called_once()
        benchmark_ready_mock.assert_called_once_with(logger, force_retest=False, force_rebuild=False, verbose=False)
        detect_os_mock.assert_called_once()
        ensure_rules_mock.assert_called_once()
        load_supervisor_class_mock.assert_called_once()
        supervisor_cls_mock.assert_called_once()
        supervisor_mock.run.assert_called_once()
        supervisor_mock.shutdown.assert_called_once()

    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=1)
    def test_main_stops_and_reports_when_setup_is_needed(
        self,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
    ) -> None:
        configure_logging_mock.return_value = mock.Mock()

        result = bootstrap.main([])

        self.assertEqual(result, 1)
        runtime_environment_mock.assert_called_once()

    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=1)
    def test_main_applies_log_start_mode_before_configuring_logging(
        self,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
    ) -> None:
        events: list[tuple[str, str]] = []

        def fake_apply_log_start_mode(mode: str) -> tuple[int, str] | None:
            events.append(("apply", mode))
            return None

        def fake_configure_logging(*args, **kwargs):
            del args, kwargs
            events.append(("configure", "logger"))
            return mock.Mock()

        configure_logging_mock.side_effect = fake_configure_logging

        with mock.patch("bootstrap._apply_log_start_mode", side_effect=fake_apply_log_start_mode):
            result = bootstrap.main(["--log-start-mode", "clean"])

        self.assertEqual(result, 1)
        self.assertEqual(events, [("apply", "clean"), ("configure", "logger")])
        runtime_environment_mock.assert_called_once()

    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=None)
    def test_main_forwards_retest_and_rebuild_flags_to_benchmark_prep(
        self,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
    ) -> None:
        logger = mock.Mock()
        configure_logging_mock.return_value = logger
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
            hostname="mdong-zephy14",
            is_wsl=False,
            is_admin=False,
            can_elevate=True,
        )

        with mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=False) as benchmark_ready_mock:
            result = bootstrap.main(["--retest", "--rebuild"])

        self.assertEqual(result, 1)
        runtime_environment_mock.assert_called_once()
        benchmark_ready_mock.assert_called_once_with(logger, force_retest=True, force_rebuild=True, verbose=False)


class BootstrapNoCliTests(unittest.TestCase):
    """Validate the ``--no-cli`` detach handoff and config propagation."""

    def test_build_parser_accepts_no_cli_flag(self) -> None:
        args = bootstrap.build_parser().parse_args(["--no-cli"])
        self.assertTrue(args.no_cli)

    def test_build_config_propagates_no_cli(self) -> None:
        args = bootstrap.build_parser().parse_args(["--no-cli"])
        config = bootstrap.build_config(args)
        self.assertTrue(config.no_cli)

    def test_build_config_defaults_no_cli_false(self) -> None:
        args = bootstrap.build_parser().parse_args([])
        config = bootstrap.build_config(args)
        self.assertFalse(config.no_cli)

    def test_build_parser_accepts_peer_process_flag(self) -> None:
        args = bootstrap.build_parser().parse_args(["--peer-process"])
        self.assertTrue(args.peer_process)

    def test_build_config_propagates_peer_process(self) -> None:
        args = bootstrap.build_parser().parse_args(["--peer-process"])
        config = bootstrap.build_config(args)
        self.assertTrue(config.peer_process)

    def test_build_config_defaults_peer_process_false(self) -> None:
        args = bootstrap.build_parser().parse_args([])
        config = bootstrap.build_config(args)
        self.assertFalse(config.peer_process)

    @mock.patch("bootstrap.detach_from_current_console", return_value=True)
    @mock.patch("bootstrap.has_attached_console", return_value=True)
    @mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=False)
    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=None)
    def test_main_detaches_and_exits_zero_when_no_cli_with_console(
        self,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
        benchmark_ready_mock: mock.Mock,
        has_console_mock: mock.Mock,
        detach_mock: mock.Mock,
    ) -> None:
        configure_logging_mock.return_value = mock.Mock()
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
            hostname="host",
            is_wsl=False,
            is_admin=True,
            can_elevate=False,
        )

        result = bootstrap.main(["--no-cli"])

        self.assertEqual(result, 0)
        detach_mock.assert_called_once()
        benchmark_ready_mock.assert_not_called()

    @mock.patch("bootstrap.detach_from_current_console")
    @mock.patch("bootstrap.has_attached_console", return_value=False)
    @mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=False)
    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=None)
    def test_main_skips_detach_when_console_already_absent(
        self,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
        benchmark_ready_mock: mock.Mock,
        has_console_mock: mock.Mock,
        detach_mock: mock.Mock,
    ) -> None:
        configure_logging_mock.return_value = mock.Mock()
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
            hostname="host",
            is_wsl=False,
            is_admin=True,
            can_elevate=False,
        )

        bootstrap.main(["--no-cli"])

        detach_mock.assert_not_called()
        benchmark_ready_mock.assert_called_once()

    @mock.patch("bootstrap.relaunch_as_admin", return_value=True)
    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_bootstrap_runtime_environment", return_value=None)
    def test_main_passes_hidden_true_to_relaunch_when_no_cli_and_elevating(
        self,
        runtime_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
        relaunch_mock: mock.Mock,
    ) -> None:
        configure_logging_mock.return_value = mock.Mock()
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
            hostname="host",
            is_wsl=False,
            is_admin=False,
            can_elevate=True,
        )

        result = bootstrap.main(["--no-cli", "--elevate-if-needed"])

        self.assertEqual(result, 0)
        relaunch_mock.assert_called_once_with(hidden=True)


if __name__ == "__main__":
    unittest.main()
