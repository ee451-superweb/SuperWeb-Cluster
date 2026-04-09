#!/usr/bin/env python3
"""Minimal ZeroMQ throughput receiver for standalone LAN testing."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings

from tcp_speed import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PROGRESS_INTERVAL,
    ThroughputResult,
    UserInterrupted,
    format_bytes,
    install_interrupt_event,
    parse_byte_count,
    summarize_result,
    wait_for_interruptible,
)

try:
    import zmq
    import zmq.asyncio
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "pyzmq is not installed. Activate .venv or run .\\.venv\\Scripts\\python -m pip install pyzmq"
    ) from exc


DEFAULT_PORT = 52041


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for a ZeroMQ PAIR throughput test and report received throughput."
    )
    parser.add_argument("--bind", default="0.0.0.0", help="IPv4 address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port to listen on.")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity before aborting the test.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Seconds between progress prints. Use 0 to disable.",
    )
    parser.add_argument(
        "--rcvbuf",
        type=parse_byte_count,
        default=0,
        help="Optional underlying OS receive buffer size such as 4MiB.",
    )
    parser.add_argument(
        "--rcvhwm",
        type=int,
        default=0,
        help="Optional ZeroMQ receive high-water mark in messages. Default keeps library value.",
    )
    args = parser.parse_args()
    if args.idle_timeout <= 0:
        parser.error("--idle-timeout must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval cannot be negative")
    if args.rcvhwm < 0:
        parser.error("--rcvhwm cannot be negative")
    return args


def encode_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_json(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("control message must be a JSON object")
    return payload


async def async_main(args: argparse.Namespace) -> int:
    stop_event, restore_handlers = install_interrupt_event(asyncio.get_running_loop())
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.PAIR)
    socket.linger = 0
    if args.rcvbuf > 0:
        socket.setsockopt(zmq.RCVBUF, args.rcvbuf)
    if args.rcvhwm > 0:
        socket.setsockopt(zmq.RCVHWM, args.rcvhwm)

    endpoint = f"tcp://{args.bind}:{args.port}"
    socket.bind(endpoint)
    actual_rcvbuf = socket.getsockopt(zmq.RCVBUF)
    actual_rcvhwm = socket.getsockopt(zmq.RCVHWM)

    try:
        print(f"listening on {endpoint}")
        print(f"ZeroMQ receive buffer {format_bytes(actual_rcvbuf)}")
        print(f"ZeroMQ receive HWM {actual_rcvhwm} message(s)")

        header = decode_json(
            bytes(
                await wait_for_interruptible(
                    lambda: socket.recv(copy=False),
                    stop_event,
                    timeout=args.idle_timeout,
                )
            )
        )
        mode = header.get("mode")
        if mode not in {"duration", "bytes"}:
            raise ValueError(f"unsupported mode: {mode!r}")

        chunk_size = header.get("chunk_size")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        duration = header.get("duration")
        if duration is not None and not isinstance(duration, (int, float)):
            raise ValueError("duration must be numeric when present")

        total_bytes = header.get("total_bytes")
        if total_bytes is not None and (not isinstance(total_bytes, int) or total_bytes <= 0):
            raise ValueError("total_bytes must be a positive integer when present")

        await socket.send(encode_json({"status": "ready"}))
        expected = f"{float(duration):.3f}s" if mode == "duration" else format_bytes(int(total_bytes or 0))
        print(f"test mode {mode}, target {expected}, sender chunk {format_bytes(chunk_size)}")

        start_time: float | None = None
        total_received = 0
        next_progress = 0.0

        while True:
            frame = await wait_for_interruptible(
                lambda: socket.recv(copy=False),
                stop_event,
                timeout=args.idle_timeout,
            )
            if len(frame) == 0:
                break

            now = time.perf_counter()
            if start_time is None:
                start_time = now
                next_progress = start_time + args.progress_interval

            total_received += len(frame)

            if args.progress_interval > 0 and now >= next_progress:
                progress = ThroughputResult(total_received, max(now - start_time, 1e-9))
                print(f"progress {summarize_result(progress)}")
                next_progress = now + args.progress_interval

        end_time = time.perf_counter()
        elapsed = 0.0 if start_time is None else max(end_time - start_time, 0.0)
        result = ThroughputResult(total_received, elapsed)
        await socket.send(
            encode_json(
                {
                    "status": "ok",
                    "total_bytes": result.total_bytes,
                    "elapsed_seconds": result.elapsed_seconds,
                    "mib_per_s": result.mib_per_s,
                    "mbit_per_s": result.mbit_per_s,
                }
            )
        )
        print(f"completed {summarize_result(result)}")
        return 0
    except UserInterrupted:
        return 130
    finally:
        socket.close()
        restore_handlers()


def main() -> int:
    args = parse_args()
    selector_policy = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if sys.platform.startswith("win") and selector_policy is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(selector_policy())
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("stopped by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
