"""Shared helpers for standalone TCP throughput tests."""

from __future__ import annotations

import json
import re
import socket
from dataclasses import dataclass
from typing import Any

DEFAULT_PORT = 52021
DEFAULT_CHUNK_SIZE = 256 * 1024
DEFAULT_DURATION = 10.0
DEFAULT_PROGRESS_INTERVAL = 1.0
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_IDLE_TIMEOUT = 15.0
MAX_CONTROL_LINE = 64 * 1024

SIZE_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)?)(?P<unit>[a-zA-Z]*)$")
SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1000,
    "ki": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1000**2,
    "mi": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1000**3,
    "gi": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1000**4,
    "ti": 1024**4,
    "tib": 1024**4,
}


@dataclass(slots=True)
class ThroughputResult:
    total_bytes: int
    elapsed_seconds: float

    @property
    def mib_per_s(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_bytes / self.elapsed_seconds / (1024**2)

    @property
    def mbit_per_s(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return (self.total_bytes * 8.0) / self.elapsed_seconds / 1_000_000


def parse_byte_count(text: str) -> int:
    """Parse sizes like `1048576`, `256K`, `64MiB`, or `1.5G`."""

    normalized = text.strip().replace("_", "")
    match = SIZE_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(f"invalid size: {text!r}")

    unit = match.group("unit").lower()
    multiplier = SIZE_UNITS.get(unit)
    if multiplier is None:
        raise ValueError(f"unknown size suffix: {unit!r}")

    value = int(float(match.group("number")) * multiplier)
    if value <= 0:
        raise ValueError("size must be positive")
    return value


def format_bytes(value: int) -> str:
    """Format a byte count using binary units."""

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


def summarize_result(result: ThroughputResult) -> str:
    """Return a compact human-readable throughput summary."""

    return (
        f"{format_bytes(result.total_bytes)} in {result.elapsed_seconds:.3f}s "
        f"({result.mib_per_s:.2f} MiB/s, {result.mbit_per_s:.2f} Mbit/s)"
    )


def send_json_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    """Send a newline-delimited JSON control message."""

    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    sock.sendall(data)


def recv_json_line(sock: socket.socket, *, limit: int = MAX_CONTROL_LINE) -> dict[str, Any]:
    """Receive one newline-delimited JSON control message."""

    buffer = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("connection closed while waiting for control message")
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise ValueError("control message exceeds maximum length")

        newline_index = buffer.find(b"\n")
        if newline_index == -1:
            continue

        trailing = buffer[newline_index + 1 :]
        if trailing:
            raise ValueError("unexpected extra bytes after control message")

        line = buffer[:newline_index].decode("utf-8")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("control message must be a JSON object")
        return payload
