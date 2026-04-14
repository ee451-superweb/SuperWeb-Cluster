"""Shared helpers for standalone TCP throughput tests."""

from __future__ import annotations

import asyncio
import json
import re
import signal
import socket
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

DEFAULT_PORT = 52021
DEFAULT_CHUNK_SIZE = 256 * 1024
DEFAULT_DURATION = 10.0
DEFAULT_PROGRESS_INTERVAL = 1.0
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_IDLE_TIMEOUT = 15.0
DEFAULT_STREAMS = 1
INTERRUPT_POLL_INTERVAL = 0.25
MAX_CONTROL_LINE = 64 * 1024
T = TypeVar("T")

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


class UserInterrupted(Exception):
    """Raised when the user requests shutdown with Ctrl-C or SIGTERM."""


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


async def async_send_json_line(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    """Send a newline-delimited JSON control message over asyncio streams."""

    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    writer.write(data)
    await writer.drain()


async def async_recv_json_line(
    reader: asyncio.StreamReader,
    *,
    limit: int = MAX_CONTROL_LINE,
) -> dict[str, Any]:
    """Receive one newline-delimited JSON control message over asyncio streams."""

    try:
        line = await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError as exc:
        if exc.partial:
            raise ConnectionError("connection closed while waiting for control message") from exc
        raise ConnectionError("connection closed while waiting for control message") from exc
    except asyncio.LimitOverrunError as exc:
        raise ValueError("control message exceeds maximum length") from exc

    if len(line) > limit:
        raise ValueError("control message exceeds maximum length")

    payload = json.loads(line[:-1].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("control message must be a JSON object")
    return payload


def install_interrupt_event(
    loop: asyncio.AbstractEventLoop,
    *,
    announce: bool = True,
) -> tuple[asyncio.Event, Callable[[], None]]:
    """Install SIGINT/SIGTERM handlers that flip an asyncio Event."""

    stop_event = asyncio.Event()
    installed: dict[int, Any] = {}
    announced = False

    def handler(signum: int, _frame: object) -> None:
        nonlocal announced
        if announce and not announced:
            print("stopping...")
            announced = True
        loop.call_soon_threadsafe(stop_event.set)

    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)

    for current in signals:
        installed[current] = signal.getsignal(current)
        signal.signal(current, handler)

    def restore() -> None:
        for current, previous in installed.items():
            signal.signal(current, previous)

    return stop_event, restore


async def wait_for_interruptible(
    awaitable_factory: Callable[[], Awaitable[T]],
    stop_event: asyncio.Event,
    *,
    timeout: float | None = None,
    poll_interval: float = INTERRUPT_POLL_INTERVAL,
) -> T:
    """Repeatedly await an operation while checking for Ctrl-C."""

    deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout

    while True:
        if stop_event.is_set():
            raise UserInterrupted()

        current_timeout = poll_interval
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            current_timeout = min(current_timeout, remaining)

        try:
            return await asyncio.wait_for(awaitable_factory(), timeout=current_timeout)
        except asyncio.TimeoutError:
            if stop_event.is_set():
                raise UserInterrupted() from None
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise


async def sleep_interruptible(
    delay: float,
    stop_event: asyncio.Event,
    *,
    poll_interval: float = INTERRUPT_POLL_INTERVAL,
) -> None:
    """Sleep in small slices so Ctrl-C stays responsive."""

    deadline = asyncio.get_running_loop().time() + delay
    while True:
        if stop_event.is_set():
            raise UserInterrupted()

        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(poll_interval, remaining))
