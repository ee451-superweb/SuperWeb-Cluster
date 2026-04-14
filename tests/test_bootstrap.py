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
from common.types import DiscoveryResult, FirewallStatus, PlatformInfo
from setup import ProjectEnvironmentStatus


class BootstrapTests(unittest.TestCase):
    """Validate bootstrap's benchmark auto-run and happy path."""

    def test_ensure_bootstrap_runtime_environment_prompts_setup_when_env_missing(self) -> None:
        logger = mock.Mock()

        with mock.patch(
            "bootstrap.inspect_project_environment",
            return_value=ProjectEnvironmentStatus(
                venv_exists=False,
                requirements_current=False,
                using_project_python=False,
            ),
        ):
            result = bootstrap.ensure_bootstrap_runtime_environment(logger, [])

        self.assertEqual(result, 1)
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

    def test_ensure_compute_node_benchmark_ready_runs_benchmark_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result_path = temp_root / "compute_node" / "performance_metrics" / "result.json"
            script_path = temp_root / "compute_node" / "performance_metrics" / "benchmark.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("# placeholder\n", encoding="utf-8")
            logger = mock.Mock()

            def fake_run(*_args, **_kwargs) -> mock.Mock:
                result_path.write_text("{}", encoding="utf-8")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(bootstrap, "PROJECT_ROOT", temp_root),
                mock.patch.object(bootstrap, "BENCHMARK_SCRIPT_PATH", script_path),
                mock.patch.object(bootstrap, "BENCHMARK_RESULT_PATH", result_path),
                mock.patch("bootstrap.subprocess.run", side_effect=fake_run) as run_mock,
            ):
                ready = bootstrap.ensure_compute_node_benchmark_ready(logger)

        self.assertTrue(ready)
        run_mock.assert_called_once()
        logger.warning.assert_called_once()
        logger.info.assert_called()

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

        result = bootstrap.main([])

        self.assertEqual(result, 1)
        runtime_environment_mock.assert_called_once()
        benchmark_ready_mock.assert_called_once()
        detect_os_mock.assert_not_called()

    @mock.patch("bootstrap.Supervisor")
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
        supervisor_cls_mock: mock.Mock,
    ) -> None:
        del relaunch_as_admin_mock
        logger = mock.Mock()
        configure_logging_mock.return_value = logger
        detect_os_mock.return_value = PlatformInfo(
            platform_name="windows",
            system="Windows",
            release="11",
            machine="AMD64",
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
        benchmark_ready_mock.assert_called_once()
        detect_os_mock.assert_called_once()
        ensure_rules_mock.assert_called_once()
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


if __name__ == "__main__":
    unittest.main()
