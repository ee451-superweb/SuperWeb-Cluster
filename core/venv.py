"""Helpers for preferring the project virtual environment at runtime."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from adapters.process import python_utf8_command
from core.constants import LOGGER_NAME
from setup import current_python_uses_project_venv, project_python_path


def relaunch_with_project_python_if_needed(
    argv: list[str] | None = None,
    *,
    script_path: Path | None = None,
    cwd: Path | None = None,
) -> int | None:
    """Relaunch the current CLI with the project venv when available."""

    venv_python = project_python_path()
    if not venv_python.exists() or current_python_uses_project_venv():
        return None

    effective_argv = list(sys.argv[1:] if argv is None else argv)
    target_script = Path(sys.argv[0]).resolve() if script_path is None else Path(script_path)
    launch_cwd = Path.cwd() if cwd is None else Path(cwd)
    command = python_utf8_command(venv_python, target_script, *effective_argv)
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("Relaunching with project virtual environment: %s", venv_python)
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=launch_cwd,
        )
    except OSError as exc:
        logger.error("Failed to relaunch with project virtual environment: %s", exc)
        return 1
    return int(result.returncode)
