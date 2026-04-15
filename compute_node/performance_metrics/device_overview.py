"""Best-effort host and accelerator inventory for benchmark reports."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any

from common.hardware import _detect_total_memory_bytes
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.windows_gpu_inventory import (
    list_windows_display_adapters,
)


def _run_powershell_json(script: str) -> Any | None:
    if os.name != "nt":
        return None
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0:
        return None
    payload_text = (completed.stdout or "").strip()
    if not payload_text:
        return None
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError:
        return None


def detect_cpu_name() -> str:
    payload = _run_powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    if isinstance(payload, list):
        names = [str(value).strip() for value in payload if str(value).strip()]
        if names:
            return names[0]
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return platform.processor() or platform.machine()


def detect_memory_modules() -> list[dict[str, object]]:
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
    return [device for device in devices if device["name"]]


def collect_device_overview() -> dict[str, object]:
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
