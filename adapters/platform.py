"""Platform and privilege detection helpers."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys

from common.types import PlatformInfo
from trace_utils import trace_function


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

    return PlatformInfo(
        platform_name=platform_name,
        system=system,
        release=platform.release(),
        machine=platform.machine(),
        is_wsl=platform_name == "wsl",
        is_admin=admin,
        can_elevate=platform_name == "windows" and not admin,
    )


@trace_function
def relaunch_as_admin(argv: list[str] | None = None) -> bool:
    """Attempt to relaunch the current process with Windows elevation."""

    if platform.system().lower() != "windows":
        return False

    # ShellExecuteW with the "runas" verb is the standard Windows elevation
    # hook for reopening the same Python entry point as administrator.
    argv = list(sys.argv if argv is None else argv)
    params = subprocess.list2cmdline(argv[1:])

    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            params,
            None,
            1,
        )
    except (AttributeError, OSError):
        return False

    return result > 32
