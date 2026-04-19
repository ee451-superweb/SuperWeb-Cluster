"""Process-boundary encoding adapter.

Forces UTF-8 across Python process boundaries on every host so the runtime
no longer depends on the OS locale code page (cp936/GBK on Chinese Windows,
shift_jis on Japanese Windows, etc.) or on the Windows "Beta UTF-8 worldwide
language support" toggle. See PEP 540 for the underlying Python UTF-8 mode
that ``python_utf8_command`` activates for child interpreters.
"""

from __future__ import annotations

import os
import sys
from typing import Any


TEXT_ENCODING = "utf-8"
TEXT_ERRORS = "replace"

TEXT_SUBPROCESS_KWARGS: dict[str, Any] = {
    "text": True,
    "encoding": TEXT_ENCODING,
    "errors": TEXT_ERRORS,
}


def enable_utf8_mode() -> None:
    """Force UTF-8 stdio for this process and Python UTF-8 mode for children.

    Call once at the very top of every process entry point, before any other
    import that may capture ``sys.stdout`` or read environment variables.

    The current Python interpreter cannot enter PEP 540 UTF-8 mode after
    startup, so this only:

    - reconfigures the live ``sys.stdout``/``sys.stderr`` text streams to
      UTF-8 with ``replace`` so redirected stdio also escapes the locale code
      page even though the in-console path already uses ``WriteConsoleW``
    - exports ``PYTHONUTF8=1`` and ``PYTHONIOENCODING=utf-8`` so child Python
      interpreters spawned from this process start in UTF-8 mode

    Non-Python child processes still need ``encoding="utf-8"`` passed to
    ``subprocess.run``/``Popen`` explicitly, which is what
    ``TEXT_SUBPROCESS_KWARGS`` is for.
    """

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding=TEXT_ENCODING, errors=TEXT_ERRORS)
        except (OSError, ValueError):
            pass


def python_utf8_command(executable: str | os.PathLike, *args: str | os.PathLike) -> list[str]:
    """Build a Python child command that starts in PEP 540 UTF-8 mode.

    Prepends ``-X utf8`` so the child interpreter forces UTF-8 even if the
    parent did not export ``PYTHONUTF8`` or the launcher stripped it.
    """

    return [str(executable), "-X", "utf8", *(str(arg) for arg in args)]
