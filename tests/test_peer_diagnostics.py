"""Unit tests for the py-spy stack-dump helper.

The helper has to degrade gracefully whenever py-spy is missing, the target
already exited, or py-spy itself wedges. Callers (the supervisor) write the
returned string straight into a WARNING log, so any failure path that raises
would produce no record at all — defeating the entire point of the helper.
"""

import subprocess
import unittest
from unittest import mock

from supervision import peer_diagnostics


class ResolvePySpyExecutableTests(unittest.TestCase):
    def test_returns_none_when_neither_venv_nor_path_has_py_spy(self) -> None:
        with mock.patch.object(peer_diagnostics.Path, "is_file", return_value=False), \
             mock.patch.object(peer_diagnostics.shutil, "which", return_value=None):
            self.assertIsNone(peer_diagnostics.resolve_py_spy_executable())


class DumpPythonStackTests(unittest.TestCase):
    """Validate the helper's failure-tolerance and output formatting."""

    def test_returns_unavailable_marker_when_py_spy_missing(self) -> None:
        with mock.patch.object(peer_diagnostics, "resolve_py_spy_executable", return_value=None):
            result = peer_diagnostics.dump_python_stack(1234)
        self.assertTrue(result.startswith("<py-spy unavailable"), result)

    def test_returns_stdout_when_py_spy_succeeds(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["py-spy", "dump", "--pid", "1234"],
            returncode=0,
            stdout="Thread 0x1: blocked at runner.py:42\n",
            stderr="",
        )
        with mock.patch.object(peer_diagnostics, "resolve_py_spy_executable", return_value="/usr/bin/py-spy"), \
             mock.patch.object(peer_diagnostics.subprocess, "run", return_value=completed) as run_mock:
            result = peer_diagnostics.dump_python_stack(1234)
        run_mock.assert_called_once()
        self.assertIn("blocked at runner.py:42", result)

    def test_includes_stderr_when_present(self) -> None:
        # py-spy on Windows prints diagnostic noise to stderr even on success
        # (e.g. "Found native python frames..."); we want it preserved so the
        # operator can see attach errors mixed with the stack.
        completed = subprocess.CompletedProcess(
            args=["py-spy", "dump", "--pid", "1234"],
            returncode=0,
            stdout="Thread 0x1: ok\n",
            stderr="warning: incomplete native frames\n",
        )
        with mock.patch.object(peer_diagnostics, "resolve_py_spy_executable", return_value="/usr/bin/py-spy"), \
             mock.patch.object(peer_diagnostics.subprocess, "run", return_value=completed):
            result = peer_diagnostics.dump_python_stack(1234)
        self.assertIn("Thread 0x1: ok", result)
        self.assertIn("[stderr]", result)
        self.assertIn("incomplete native frames", result)

    def test_returns_failure_marker_when_py_spy_returns_nonzero(self) -> None:
        # Access denied / process gone manifest as nonzero returncode; we must
        # not raise — the supervisor logs the marker and continues.
        completed = subprocess.CompletedProcess(
            args=["py-spy", "dump", "--pid", "1234"],
            returncode=1,
            stdout="",
            stderr="Error: Failed to open process",
        )
        with mock.patch.object(peer_diagnostics, "resolve_py_spy_executable", return_value="/usr/bin/py-spy"), \
             mock.patch.object(peer_diagnostics.subprocess, "run", return_value=completed):
            result = peer_diagnostics.dump_python_stack(1234)
        self.assertTrue(result.startswith("<py-spy failed"), result)
        self.assertIn("Failed to open process", result)

    def test_returns_marker_when_py_spy_times_out(self) -> None:
        with mock.patch.object(peer_diagnostics, "resolve_py_spy_executable", return_value="/usr/bin/py-spy"), \
             mock.patch.object(
                 peer_diagnostics.subprocess,
                 "run",
                 side_effect=subprocess.TimeoutExpired(cmd=["py-spy"], timeout=15.0),
             ):
            result = peer_diagnostics.dump_python_stack(1234, timeout=15.0)
        self.assertIn("<py-spy failed", result)
        self.assertIn("exceeded", result)


if __name__ == "__main__":
    unittest.main()
