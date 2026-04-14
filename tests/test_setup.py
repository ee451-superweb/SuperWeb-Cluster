"""Project setup entry-point tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import setup


class SetupTests(unittest.TestCase):
    """Validate local venv setup and dependency install separation."""

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
                mock.patch.object(setup, "PROJECT_ROOT", temp_root),
                mock.patch.object(setup, "VENV_DIR", venv_dir),
                mock.patch.object(setup, "REQUIREMENTS_PATH", requirements_path),
                mock.patch.object(setup, "REQUIREMENTS_STAMP_PATH", stamp_path),
                mock.patch("setup.subprocess.run", side_effect=fake_run) as run_mock,
            ):
                ready = setup.ensure_project_python_environment(logger)

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
            expected_hash = setup.hashlib.sha256(requirements_path.read_bytes()).hexdigest()
            stamp_path.write_text(expected_hash, encoding="utf-8")
            logger = mock.Mock()

            with (
                mock.patch.object(setup, "PROJECT_ROOT", temp_root),
                mock.patch.object(setup, "VENV_DIR", venv_dir),
                mock.patch.object(setup, "REQUIREMENTS_PATH", requirements_path),
                mock.patch.object(setup, "REQUIREMENTS_STAMP_PATH", stamp_path),
                mock.patch("setup.subprocess.run") as run_mock,
            ):
                ready = setup.ensure_project_python_environment(logger)

            self.assertTrue(ready)
            run_mock.assert_not_called()

    def test_venv_only_skips_dependency_install(self) -> None:
        logger = mock.Mock()

        with (
            mock.patch("setup.ensure_virtual_environment", return_value=True) as ensure_venv_mock,
            mock.patch("setup.install_project_requirements") as install_mock,
        ):
            ready = setup.ensure_project_python_environment(
                logger,
                install_requirements_flag=False,
            )

        self.assertTrue(ready)
        ensure_venv_mock.assert_called_once_with(logger)
        install_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

