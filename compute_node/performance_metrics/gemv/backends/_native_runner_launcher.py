"""Shared helper for invoking a native gemv runner with per-trial streaming.

Each gemv backend launches its native runner the same way: stdout carries the
final JSON report, stderr carries per-trial progress lines when ``--verbose``
is passed. ``subprocess.run(..., capture_output=True)`` buffers both streams
until the process exits, which hides verbose progress from the parent terminal.
``Popen.communicate`` spawns its own drain thread that would race our stderr
pump for the same pipe and silently swallow progress lines. So we launch the
runner with ``Popen`` and pump stderr + stdout in two daemon threads, then
parse the buffered stdout as JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path


def run_native_runner_with_streaming(
    command: list[str],
    *,
    timeout_seconds: float,
    cwd: Path,
) -> dict[str, object]:
    """Run a native gemv runner and stream its stderr to the parent terminal.

    Args:
        command: Full argument vector for the native runner.
        timeout_seconds: Subprocess timeout for the run.
        cwd: Working directory for the subprocess (usually the gemv root).

    Returns:
        The parsed JSON metrics written to stdout by the runner.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    stderr_chunks: list[str] = []
    stdout_chunks: list[str] = []

    def _pump_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    def _pump_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stdout_chunks.append(line)

    stderr_pump = threading.Thread(target=_pump_stderr, daemon=True)
    stdout_pump = threading.Thread(target=_pump_stdout, daemon=True)
    stderr_pump.start()
    stdout_pump.start()
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        stderr_pump.join(timeout=1.0)
        stdout_pump.join(timeout=1.0)
        raise
    stderr_pump.join(timeout=5.0)
    stdout_pump.join(timeout=5.0)
    stdout_data = "".join(stdout_chunks)
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
            output=stdout_data,
            stderr="".join(stderr_chunks),
        )
    return json.loads(stdout_data)
