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


class BootstrapTests(unittest.TestCase):
    """Validate bootstrap's benchmark auto-run and happy path."""

    def test_ensure_project_python_environment_creates_venv_and_installs_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            venv_dir = temp_root / ".venv"
            requirements_path = temp_root / "requirements.txt"
            requirements_path.write_text("tqdm>=4.67.0\n", encoding="utf-8")
            stamp_path = venv_dir / ".requirements.sha256"
            logger = mock.Mock()

            def fake_run(command, **_kwargs):
                if command[1:3] == ["-m", "venv"]:
                    python_path = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / (
                        "python.exe" if sys.platform == "win32" else "python"
                    )
                    python_path.parent.mkdir(parents=True, exist_ok=True)
                    python_path.write_text("", encoding="utf-8")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(bootstrap, "PROJECT_ROOT", temp_root),
                mock.patch.object(bootstrap, "VENV_DIR", venv_dir),
                mock.patch.object(bootstrap, "REQUIREMENTS_PATH", requirements_path),
                mock.patch.object(bootstrap, "REQUIREMENTS_STAMP_PATH", stamp_path),
                mock.patch("bootstrap.subprocess.run", side_effect=fake_run) as run_mock,
            ):
                ready = bootstrap.ensure_project_python_environment(logger)
                self.assertTrue(ready)
                self.assertEqual(run_mock.call_count, 2)
                self.assertTrue(stamp_path.exists())

    def test_ensure_project_python_environment_skips_reinstall_when_stamp_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            venv_dir = temp_root / ".venv"
            requirements_path = temp_root / "requirements.txt"
            requirements_path.write_text("tqdm>=4.67.0\n", encoding="utf-8")
            stamp_path = venv_dir / ".requirements.sha256"
            python_path = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / (
                "python.exe" if sys.platform == "win32" else "python"
            )
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            expected_hash = bootstrap.hashlib.sha256(requirements_path.read_bytes()).hexdigest()
            stamp_path.write_text(expected_hash, encoding="utf-8")
            logger = mock.Mock()

            with (
                mock.patch.object(bootstrap, "PROJECT_ROOT", temp_root),
                mock.patch.object(bootstrap, "VENV_DIR", venv_dir),
                mock.patch.object(bootstrap, "REQUIREMENTS_PATH", requirements_path),
                mock.patch.object(bootstrap, "REQUIREMENTS_STAMP_PATH", stamp_path),
                mock.patch("bootstrap.subprocess.run") as run_mock,
            ):
                ready = bootstrap.ensure_project_python_environment(logger)
                self.assertTrue(ready)
                run_mock.assert_not_called()

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
    @mock.patch("bootstrap.ensure_project_python_environment", return_value=True)
    @mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=False)
    def test_main_stops_when_benchmark_auto_run_fails(
        self,
        benchmark_ready_mock: mock.Mock,
        python_environment_mock: mock.Mock,
        configure_logging_mock: mock.Mock,
        detect_os_mock: mock.Mock,
    ) -> None:
        configure_logging_mock.return_value = mock.Mock()

        result = bootstrap.main([])

        self.assertEqual(result, 1)
        python_environment_mock.assert_called_once()
        benchmark_ready_mock.assert_called_once()
        detect_os_mock.assert_not_called()

    @mock.patch("bootstrap.Supervisor")
    @mock.patch("bootstrap.ensure_rules")
    @mock.patch("bootstrap.relaunch_as_admin", return_value=False)
    @mock.patch("bootstrap.detect_os")
    @mock.patch("bootstrap.configure_logging")
    @mock.patch("bootstrap.ensure_project_python_environment", return_value=True)
    @mock.patch("bootstrap.ensure_compute_node_benchmark_ready", return_value=True)
    def test_main_continues_when_benchmark_result_exists(
        self,
        benchmark_ready_mock: mock.Mock,
        python_environment_mock: mock.Mock,
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
        python_environment_mock.assert_called_once()
        benchmark_ready_mock.assert_called_once()
        detect_os_mock.assert_called_once()
        ensure_rules_mock.assert_called_once()
        supervisor_cls_mock.assert_called_once()
        supervisor_mock.run.assert_called_once()
        supervisor_mock.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
