"""Windows display-adapter helpers for backend routing.

The benchmark should make one coherent routing decision on Windows:

- NVIDIA display adapters prefer the CUDA backend
- non-NVIDIA display adapters prefer the DX12 backend

This module keeps that adapter inspection in one place so individual backends
do not each reinvent the same PowerShell / WMI probe.
"""

from __future__ import annotations

import json
import os
import subprocess


def _adapter_identity(adapter: dict[str, str]) -> str:
    parts = (
        str(adapter.get("Name") or "").strip(),
        str(adapter.get("AdapterCompatibility") or "").strip(),
        str(adapter.get("PNPDeviceID") or "").strip(),
    )
    return " ".join(part for part in parts if part).lower()


def _is_software_adapter(adapter: dict[str, str]) -> bool:
    identity = _adapter_identity(adapter)
    return "microsoft basic" in identity or "software" in identity


def list_windows_display_adapters() -> tuple[list[dict[str, str]], str]:
    """Return visible Windows display adapters plus a human-readable status."""

    if os.name != "nt":
        return [], "display-adapter routing is only available on Windows"

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterCompatibility,PNPDeviceID | "
                "ConvertTo-Json -Compress"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return [], f"unable to inspect Windows video adapters: {detail or 'unknown error'}"

    payload_text = (completed.stdout or "").strip()
    if not payload_text:
        return [], "unable to inspect Windows video adapters: PowerShell returned no adapters"

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return [], f"unable to parse Windows video adapter inventory: {exc}"

    adapters = payload if isinstance(payload, list) else [payload]
    normalized: list[dict[str, str]] = []
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        normalized_adapter = {
            "Name": str(adapter.get("Name") or "").strip(),
            "AdapterCompatibility": str(adapter.get("AdapterCompatibility") or "").strip(),
            "PNPDeviceID": str(adapter.get("PNPDeviceID") or "").strip(),
        }
        if _adapter_identity(normalized_adapter):
            normalized.append(normalized_adapter)

    return normalized, ""


def detect_nvidia_windows_adapter() -> tuple[str | None, str]:
    """Return one NVIDIA display adapter name when Windows reports one."""

    adapters, message = list_windows_display_adapters()
    if not adapters:
        return None, message

    for adapter in adapters:
        identity = _adapter_identity(adapter)
        if _is_software_adapter(adapter):
            continue
        if "nvidia" in identity:
            return adapter["Name"] or adapter["AdapterCompatibility"] or adapter["PNPDeviceID"], ""

    return None, "CUDA backend is reserved for NVIDIA GPUs; no NVIDIA display adapter was detected."


def detect_non_nvidia_windows_adapter() -> tuple[str | None, str]:
    """Return one AMD/Intel-style display adapter name when Windows reports one."""

    adapters, message = list_windows_display_adapters()
    if not adapters:
        return None, message

    for adapter in adapters:
        identity = _adapter_identity(adapter)
        if _is_software_adapter(adapter):
            continue
        if "nvidia" in identity:
            continue
        return adapter["Name"] or adapter["AdapterCompatibility"] or adapter["PNPDeviceID"], ""

    return None, "DX12 backend is reserved for non-NVIDIA GPUs; no AMD or Intel adapter was detected."
