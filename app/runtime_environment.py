"""Helpers for preferring the project virtual environment at runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    command = [str(venv_python), str(target_script), *effective_argv]

    print(
        f"relaunching with project virtual environment: {venv_python}",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=launch_cwd,
        )
    except OSError as exc:
        print(
            f"failed to relaunch with project virtual environment: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return int(result.returncode)
