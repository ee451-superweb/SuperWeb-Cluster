"""Capture a Python stack dump from a hung peer subprocess.

Use this when the cluster registry has evicted a peer (heartbeat timeout) but
the OS process is still alive. The wait-based ``Supervisor._watch_peer`` thread
cannot see this case because the process never exits. This helper invokes
``py-spy dump`` so the operator gets a real frame stack instead of guessing why
the peer hung.

py-spy is invoked as a subprocess of the supervisor; on Windows the parent
already holds an inheritable handle to the child, so no ``SeDebugPrivilege`` is
needed. The helper degrades gracefully (returns ``None``) if py-spy is not
installed in the venv.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_PY_SPY_TIMEOUT_SECONDS = 15.0


def resolve_py_spy_executable() -> str | None:
    """Return the path to the venv-local py-spy binary, or ``None`` if absent.

    Why prefer the venv copy: the project pins py-spy in ``requirements.txt`` so
    every dev environment has the same version; an arbitrary system py-spy on
    PATH may be the wrong major.
    """
    venv_dir = Path(sys.executable).parent
    candidate = venv_dir / ("py-spy.exe" if sys.platform == "win32" else "py-spy")
    if candidate.is_file():
        return str(candidate)
    fallback = shutil.which("py-spy")
    return fallback


def dump_python_stack(pid: int, *, timeout: float = _PY_SPY_TIMEOUT_SECONDS) -> str:
    """Return a textual Python stack dump for ``pid`` via ``py-spy dump``.

    Args:
        pid: Operating-system process id whose stack should be captured.
        timeout: Hard cap on the py-spy invocation; stack-dumps on a healthy
            process complete in well under a second, so a long timeout almost
            always means py-spy itself is wedged.

    Returns:
        The combined stdout/stderr py-spy produced. On any failure
        (py-spy missing, access denied, target already exited, timeout) the
        returned string starts with ``"<py-spy unavailable: "`` or
        ``"<py-spy failed: "`` so callers can log it verbatim without raising.
    """
    executable = resolve_py_spy_executable()
    if executable is None:
        return "<py-spy unavailable: not installed in venv and not on PATH>"
    try:
        completed = subprocess.run(
            [executable, "dump", "--pid", str(pid)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"<py-spy unavailable: {exc}>"
    except subprocess.TimeoutExpired:
        return f"<py-spy failed: dump --pid {pid} exceeded {timeout:.1f}s>"
    except OSError as exc:
        return f"<py-spy failed: {exc}>"
    output_parts: list[str] = []
    if completed.stdout:
        output_parts.append(completed.stdout.rstrip())
    if completed.stderr:
        output_parts.append(f"[stderr]\n{completed.stderr.rstrip()}")
    body = "\n".join(output_parts) if output_parts else "<empty>"
    if completed.returncode != 0:
        return f"<py-spy failed: returncode={completed.returncode}>\n{body}"
    return body
