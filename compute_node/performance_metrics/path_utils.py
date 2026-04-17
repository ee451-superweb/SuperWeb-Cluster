"""Helpers for showing project paths without machine-specific absolute prefixes.

The benchmark code still uses absolute `Path` objects internally when that is
the easiest way to find files reliably. What we avoid here is *persisting* or
*printing* those absolute paths into generated files such as `result.json` or
temporary build scripts, because those paths leak machine-specific details and
make outputs less portable.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def to_relative_string(path: Path | str, *, start: Path | None = None) -> str:
    """Render a path relative to the performance-metrics project root.

    Use this when persisting benchmark output or logs so machine-specific
    absolute paths do not leak into generated files.

    Args:
        path: Path that should be rewritten relative to the chosen base.
        start: Optional base directory. Defaults to the performance-metrics root.

    Returns:
        A slash-normalized relative path string.
    """

    base = PROJECT_ROOT if start is None else start
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")

    relative = os.path.relpath(str(candidate), str(base))
    return relative.replace("\\", "/")


def to_relative_cli_path(path: Path | str, *, start: Path | None = None) -> str:
    """Render a relative path for subprocess command lines.

    Use this for subprocess command arguments when you want relative paths
    without breaking the platform's native path-separator expectations.

    Args:
        path: Path that should be rewritten relative to the chosen base.
        start: Optional base directory. Defaults to the performance-metrics root.

    Returns:
        A relative path string using platform-native separators.
    """

    base = PROJECT_ROOT if start is None else start
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)

    return os.path.relpath(str(candidate), str(base))


def to_relative_executable_path(path: Path | str, *, start: Path | None = None) -> str:
    """Render a relative executable path that Windows can launch reliably.

    Use this when a subprocess target should stay relative and Windows may need
    an explicit ``.\\`` prefix to execute the file successfully.

    Args:
        path: Executable path that should be rewritten relative to the base.
        start: Optional base directory. Defaults to the performance-metrics root.

    Returns:
        A launch-ready relative executable path.
    """

    relative = to_relative_cli_path(path, start=start)
    if os.name == "nt" and not relative.startswith("."):
        return f".\\{relative}"
    return relative


def sanitize_text(text: str, *, start: Path | None = None) -> str:
    """Strip machine-specific absolute-path prefixes out of free-form text.

    Use this on backend notes or command output before writing benchmark
    reports so paths remain portable across machines.

    Args:
        text: Free-form text that may contain absolute paths.
        start: Optional project-relative base directory for replacement.

    Returns:
        Sanitized text with absolute-path prefixes removed or shortened.
    """

    base = PROJECT_ROOT if start is None else start
    cleaned = text

    replacements = {
        str(base): ".",
        str(base).replace("\\", "/"): ".",
        str(Path.home()): "~",
        str(Path.home()).replace("\\", "/"): "~",
    }

    for absolute_prefix, replacement in replacements.items():
        if absolute_prefix:
            cleaned = cleaned.replace(absolute_prefix, replacement)

    return cleaned.replace("\\\\?\\", "")
