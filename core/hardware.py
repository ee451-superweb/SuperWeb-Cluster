"""Hardware discovery helpers for compute-node registration."""

from __future__ import annotations

import ctypes
import os
import platform
import socket
import subprocess

from adapters import network
from core.types import HardwareProfile
from core.tracing import trace_function


def _detect_total_memory_bytes() -> int:
    """Return best-effort physical memory size."""

    if os.name == "nt":
        memory_kb = ctypes.c_ulonglong(0)
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(memory_kb)):
            return int(memory_kb.value) * 1024
        return 0

    page_size_name = "SC_PAGE_SIZE"
    phys_pages_name = "SC_PHYS_PAGES"
    if page_size_name in os.sysconf_names and phys_pages_name in os.sysconf_names:
        try:
            return int(os.sysconf(page_size_name)) * int(os.sysconf(phys_pages_name))
        except (OSError, ValueError):
            return 0

    return 0


def detect_processor_name() -> str:
    """Return the friendliest processor label available on this host.

    Use this for registration and report headers so Apple Silicon machines do
    not collapse down to generic labels such as ``arm`` or ``arm64``.
    """

    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            detected = completed.stdout.strip()
            if detected:
                return detected

    return platform.processor() or platform.machine()


@trace_function
def collect_hardware_profile(remote_host: str = "", remote_port: int = 0) -> HardwareProfile:
    """Collect best-effort host information for runtime registration."""

    if remote_host:
        local_ip = network.resolve_local_ip(remote_host=remote_host, remote_port=max(remote_port, 1))
    else:
        local_ip = network.resolve_local_ip()

    return HardwareProfile(
        hostname=socket.gethostname(),
        local_ip=local_ip,
        mac_address=network.get_local_mac_address(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        processor=detect_processor_name(),
        logical_cpu_count=os.cpu_count() or 0,
        memory_bytes=_detect_total_memory_bytes(),
    )
