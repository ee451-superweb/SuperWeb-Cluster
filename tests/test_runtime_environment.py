"""Tests for project venv runtime relaunch helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import runtime_environment


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_relaunch_skips_when_project_python_is_missing(self) -> None:
        with mock.patch(
            "app.runtime_environment.project_python_path",
            return_value=Path("C:/missing/.venv/Scripts/python.exe"),
        ):
            result = runtime_environment.relaunch_with_project_python_if_needed(["--help"])

        self.assertIsNone(result)

    def test_relaunch_skips_when_already_using_project_python(self) -> None:
        with (
            mock.patch(
                "app.runtime_environment.project_python_path",
                return_value=Path("C:/repo/.venv/Scripts/python.exe"),
            ),
            mock.patch(
                "app.runtime_environment.current_python_uses_project_venv",
                return_value=True,
            ),
        ):
            result = runtime_environment.relaunch_with_project_python_if_needed(["--help"])

        self.assertIsNone(result)

    def test_relaunch_uses_project_python_when_available(self) -> None:
        with (
            mock.patch(
                "app.runtime_environment.project_python_path",
                return_value=Path("C:/repo/.venv/Scripts/python.exe"),
            ),
            mock.patch(
                "app.runtime_environment.current_python_uses_project_venv",
                return_value=False,
            ),
            mock.patch(
                "pathlib.Path.exists",
                return_value=True,
            ),
            mock.patch(
                "app.runtime_environment.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ) as run_mock,
        ):
            result = runtime_environment.relaunch_with_project_python_if_needed(
                ["--method", "all"],
                script_path=Path("C:/repo/compute_node/input_matrix/generate.py"),
                cwd=Path("C:/repo"),
            )

        self.assertEqual(result, 0)
        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertEqual(Path(command[0]), Path("C:/repo/.venv/Scripts/python.exe"))
        self.assertEqual(Path(command[1]), Path("C:/repo/compute_node/input_matrix/generate.py"))
        self.assertEqual(command[2:], ["--method", "all"])
        self.assertEqual(run_mock.call_args.kwargs, {"check": False, "cwd": Path("C:/repo")})


if __name__ == "__main__":
    unittest.main()
