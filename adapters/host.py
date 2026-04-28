"""Platform and privilege detection helpers."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys

from core.types import PlatformInfo
from core.tracing import trace_function


@trace_function
def is_wsl() -> bool:
    """Return True when running under Windows Subsystem for Linux."""

    if platform.system().lower() != "linux":
        return False

    # WSL can be identified either from the kernel release string or from
    # `/proc/version`, depending on the exact distro/kernel packaging.
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True

    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


@trace_function
def is_admin() -> bool:
    """Return True when the current process has elevated privileges."""

    if platform.system().lower() == "windows":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError):
            return False

    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


@trace_function
def detect_os() -> PlatformInfo:
    """Detect the current platform and privilege state."""

    system = platform.system()
    normalized = system.lower()

    if normalized == "windows":
        platform_name = "windows"
    elif normalized == "darwin":
        platform_name = "macos"
    elif normalized == "linux" and is_wsl():
        platform_name = "wsl"
    elif normalized == "linux":
        platform_name = "linux"
    else:
        platform_name = "unknown"

    admin = is_admin()
    hostname = os.environ.get("COMPUTERNAME") or platform.node() or "unknown-host"

    return PlatformInfo(
        platform_name=platform_name,
        system=system,
        release=platform.release(),
        machine=platform.machine(),
        hostname=hostname,
        is_wsl=platform_name == "wsl",
        is_admin=admin,
        can_elevate=platform_name == "windows" and not admin,
    )


@trace_function
def relaunch_as_admin(argv: list[str] | None = None, *, hidden: bool = False) -> bool:
    """Attempt to relaunch the current process with Windows elevation.

    Args:
        argv: Optional explicit argv override for the elevated child.
        hidden: When True, pass ``SW_HIDE`` so the elevated process does not
            show a console window. Use this when the caller already decided
            the runtime should be headless (e.g. ``--no-cli``) so the UAC
            handoff and headless decision collapse into one step.
    """

    if platform.system().lower() != "windows":
        return False

    # ShellExecuteW with the "runas" verb is the standard Windows elevation
    # hook for reopening the same Python entry point as administrator. argv[0]
    # is the script path — dropping it leaves the elevated python.exe with
    # only flags and no entry point, so the new admin window would exit
    # immediately. Keep the full argv instead.
    argv = list(sys.argv if argv is None else argv)
    params = subprocess.list2cmdline(["-X", "utf8", *argv])
    show_command = 0 if hidden else 1

    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            params,
            None,
            show_command,
        )
    except (AttributeError, OSError):
        return False

    return result > 32


@trace_function
def has_attached_console() -> bool:
    """Return True when the current process owns a console / TTY.

    Use this before the ``--no-cli`` detach step to skip re-detaching a
    process that was already started without a console (e.g. the already
    detached child of a prior ``--no-cli`` handoff, or a ``pythonw.exe`` run).
    """

    if platform.system().lower() == "windows":
        try:
            return bool(ctypes.windll.kernel32.GetConsoleWindow())
        except (AttributeError, OSError):
            return False

    try:
        return os.isatty(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return False


@trace_function
def detach_from_current_console(argv: list[str] | None = None) -> bool:
    """Spawn a copy of this process detached from the current console.

    Use this to honor ``--no-cli``: the new process has no console window and
    its stdout/stderr go to ``DEVNULL``; all runtime logs continue to hit the
    role-specific log file. The caller should return immediately after this
    function returns True so the original console-attached parent exits.

    Args:
        argv: Optional explicit argv override for the detached child. Pass
            the exact bootstrap argv so the detached child reaches the same
            runtime configuration.

    Returns:
        True when the detached child was spawned successfully; False when
        the platform is unsupported or ``Popen`` failed (the caller should
        continue running in-place in that case so the user is not stranded
        without any process).
    """

    argv = list(sys.argv if argv is None else argv)
    command = [sys.executable, "-X", "utf8", *argv]

    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }

    if platform.system().lower() == "windows":
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        popen_kwargs["creationflags"] = detached | no_window | new_group
    else:
        # On POSIX, detach via setsid so the child survives the controlling
        # terminal and does not receive SIGHUP when the parent exits.
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(command, **popen_kwargs)
    except OSError:
        return False
    return True

