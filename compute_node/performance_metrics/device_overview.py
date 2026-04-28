"""Collect a lightweight host and accelerator overview for benchmark reports.

Use this module when benchmark output should include a human-readable snapshot
of the machine that produced the measurements without exposing deep internals.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any

from core.hardware import _detect_total_memory_bytes, detect_processor_name
from compute_node.performance_metrics.gemv.backends.windows_gpu_inventory import (
    list_windows_display_adapters,
)


def _run_json_command(command: list[str], *, timeout: float = 5.0) -> Any | None:
    """Execute one command that emits JSON and parse its stdout payload."""

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    payload_text = (completed.stdout or "").strip()
    if not payload_text:
        return None
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError:
        return None


def _run_powershell_json(script: str) -> Any | None:
    """Execute a short PowerShell query and parse JSON output.

    Use this helper on Windows when CIM or GPU inventory data is easiest to
    obtain through PowerShell instead of reimplementing the query in Python.

    Args:
        script: PowerShell command that prints JSON to stdout.

    Returns:
        The parsed JSON payload, or ``None`` when the command fails.
    """
    if os.name != "nt":
        return None
    return _run_json_command(["powershell", "-NoProfile", "-Command", script])


def _run_system_profiler_json(data_type: str) -> Any | None:
    """Query one macOS System Profiler data source and parse its JSON payload."""

    if platform.system() != "Darwin":
        return None
    return _run_json_command(["system_profiler", data_type, "-json"])


def detect_cpu_name() -> str:
    """Return the best-effort CPU model name for the current host.

    Use this in benchmark reports so readers can quickly identify which machine
    produced the measurements without inspecting raw OS inventory output.

    Args:
        None.

    Returns:
        A CPU model string, or a generic platform fallback.
    """
    payload = _run_powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    if isinstance(payload, list):
        names = [str(value).strip() for value in payload if str(value).strip()]
        if names:
            return names[0]
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    detected = detect_processor_name().strip()
    return detected or platform.machine()


def detect_memory_modules() -> list[dict[str, object]]:
    """Return best-effort physical memory-module information.

    Use this on Windows benchmark hosts when reports should include DIMM size
    and clock information in addition to total system memory.

    Args:
        None.

    Returns:
        A list of dictionaries describing detected memory modules.
    """
    payload = _run_powershell_json(
        "Get-CimInstance Win32_PhysicalMemory | "
        "Select-Object Manufacturer,PartNumber,Capacity,ConfiguredClockSpeed | "
        "ConvertTo-Json -Compress"
    )
    rows = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else [])
    modules: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        modules.append(
            {
                "capacity_bytes": int(row.get("Capacity") or 0),
                "configured_clock_mhz": int(row.get("ConfiguredClockSpeed") or 0),
                "manufacturer": str(row.get("Manufacturer") or "").strip(),
                "part_number": str(row.get("PartNumber") or "").strip(),
            }
        )
    return modules


def detect_gpu_devices() -> list[dict[str, object]]:
    """Return display-adapter inventory for the local host.

    Use this in benchmark reports when a lightweight GPU list is enough and the
    benchmark does not need to expose backend-specific internal details.

    Args:
        None.

    Returns:
        A list of detected GPU or display-adapter descriptors.
    """
    adapters, _message = list_windows_display_adapters()
    devices: list[dict[str, object]] = []
    for adapter in adapters:
        devices.append(
            {
                "name": str(adapter.get("Name") or "").strip(),
                "vendor": str(adapter.get("AdapterCompatibility") or "").strip(),
                "pnp_device_id": str(adapter.get("PNPDeviceID") or "").strip(),
            }
        )
    if devices:
        return [device for device in devices if device["name"]]

    payload = _run_system_profiler_json("SPDisplaysDataType")
    rows = payload.get("SPDisplaysDataType") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []

    macos_devices: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("sppci_model") or row.get("_name") or "").strip()
        vendor = str(row.get("spdisplays_vendor") or "").strip()
        if vendor.startswith("sppci_vendor_"):
            vendor = vendor.removeprefix("sppci_vendor_")
        device: dict[str, object] = {
            "name": name,
            "vendor": vendor,
            "pnp_device_id": "",
        }
        core_count = str(row.get("sppci_cores") or "").strip()
        if core_count.isdigit():
            device["core_count"] = int(core_count)
        metal_family = str(row.get("spdisplays_mtlgpufamilysupport") or "").strip()
        if metal_family:
            device["metal_family"] = metal_family
        bus = str(row.get("sppci_bus") or "").strip()
        if bus:
            device["bus"] = bus
        macos_devices.append(device)
    return [device for device in macos_devices if device["name"]]


def collect_device_overview() -> dict[str, object]:
    """Assemble the host summary stored in benchmark report headers.

    Use this once per benchmark run so normalized reports include CPU, memory,
    GPU, and Python-version context for later comparison.

    Args:
        None.

    Returns:
        A dictionary describing the current host at a high level.
    """
    memory_bytes = _detect_total_memory_bytes()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu": {
            "name": detect_cpu_name(),
            "logical_cpu_count": os.cpu_count() or 0,
        },
        "memory": {
            "total_bytes": memory_bytes,
            "modules": detect_memory_modules(),
        },
        "gpus": detect_gpu_devices(),
    }
