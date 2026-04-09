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

    If the target path lives outside the project, we still return a relative
    path using `..` segments instead of falling back to an absolute path.
    """

    base = PROJECT_ROOT if start is None else start
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate).replace("\\", "/")

    relative = os.path.relpath(str(candidate), str(base))
    return relative.replace("\\", "/")


def to_relative_cli_path(path: Path | str, *, start: Path | None = None) -> str:
    """Render a relative path for subprocess command lines.

    This keeps the platform-native path separators so Windows can launch
    relative executables correctly.
    """

    base = PROJECT_ROOT if start is None else start
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)

    return os.path.relpath(str(candidate), str(base))


def to_relative_executable_path(path: Path | str, *, start: Path | None = None) -> str:
    """Render a relative executable path that Windows can launch reliably."""

    relative = to_relative_cli_path(path, start=start)
    if os.name == "nt" and not relative.startswith("."):
        return f".\\{relative}"
    return relative


def sanitize_text(text: str, *, start: Path | None = None) -> str:
    """Strip machine-specific absolute-path prefixes out of free-form text."""

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
