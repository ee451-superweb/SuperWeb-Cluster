#!/usr/bin/env python3
"""Minimal ZeroMQ throughput sender for standalone LAN testing."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings

from tcp_speed import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DURATION,
    DEFAULT_PROGRESS_INTERVAL,
    INTERRUPT_POLL_INTERVAL,
    ThroughputResult,
    UserInterrupted,
    format_bytes,
    install_interrupt_event,
    parse_byte_count,
    sleep_interruptible,
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
        description="Connect to a ZeroMQ PAIR receiver and measure sender/receiver throughput."
    )
    parser.add_argument("host", help="Receiver IPv4 or hostname.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Receiver TCP port.")
    parser.add_argument(
        "--chunk-size",
        type=parse_byte_count,
        default=DEFAULT_CHUNK_SIZE,
        help="Bytes per ZeroMQ message, for example 256KiB or 1MiB.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Seconds between progress prints. Use 0 to disable.",
    )
    parser.add_argument(
        "--sndbuf",
        type=parse_byte_count,
        default=0,
        help="Optional underlying OS send buffer size such as 4MiB.",
    )
    parser.add_argument(
        "--sndhwm",
        type=int,
        default=0,
        help="Optional ZeroMQ send high-water mark in messages. Default keeps library value.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--duration", type=float, help="Seconds to keep sending data.")
    group.add_argument("--bytes", dest="total_bytes", type=parse_byte_count, help="Exact total bytes to send.")

    args = parser.parse_args()
    if args.duration is None and args.total_bytes is None:
        args.duration = DEFAULT_DURATION
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval cannot be negative")
    if args.sndhwm < 0:
        parser.error("--sndhwm cannot be negative")
    return args


def encode_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_json(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("control message must be a JSON object")
    return payload


def print_progress(total_sent: int, start_time: float, now: float) -> None:
    result = ThroughputResult(total_sent, max(now - start_time, 1e-9))
    print(f"progress {summarize_result(result)}")


async def run_send_loop(
    socket: zmq.asyncio.Socket,
    args: argparse.Namespace,
    stop_event: asyncio.Event,
) -> ThroughputResult:
    payload = b"\0" * args.chunk_size
    total_sent = 0
    start_time = time.perf_counter()
    next_progress = start_time + args.progress_interval

    if args.total_bytes is not None:
        remaining = args.total_bytes
        while remaining > 0:
            if stop_event.is_set():
                raise UserInterrupted()
            block_size = min(args.chunk_size, remaining)
            block = payload if block_size == args.chunk_size else memoryview(payload)[:block_size]
            await socket.send(block, copy=False)
            total_sent += block_size
            remaining -= block_size

            now = time.perf_counter()
            if args.progress_interval > 0 and now >= next_progress:
                print_progress(total_sent, start_time, now)
                next_progress = now + args.progress_interval
    else:
        deadline = start_time + args.duration
        while True:
            if stop_event.is_set():
                raise UserInterrupted()
            await socket.send(payload, copy=False)
            total_sent += args.chunk_size

            now = time.perf_counter()
            if args.progress_interval > 0 and now >= next_progress:
                print_progress(total_sent, start_time, now)
                next_progress = now + args.progress_interval
            if now >= deadline:
                break

    end_time = time.perf_counter()
    return ThroughputResult(total_sent, max(end_time - start_time, 0.0))


async def async_main(args: argparse.Namespace) -> int:
    stop_event, restore_handlers = install_interrupt_event(asyncio.get_running_loop())
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.PAIR)
    socket.linger = 0
    if args.sndbuf > 0:
        socket.setsockopt(zmq.SNDBUF, args.sndbuf)
    if args.sndhwm > 0:
        socket.setsockopt(zmq.SNDHWM, args.sndhwm)

    endpoint = f"tcp://{args.host}:{args.port}"
    socket.connect(endpoint)
    actual_sndbuf = socket.getsockopt(zmq.SNDBUF)
    actual_sndhwm = socket.getsockopt(zmq.SNDHWM)

    try:
        print(f"connected to {endpoint}")
        print(f"ZeroMQ send buffer {format_bytes(actual_sndbuf)}")
        print(f"ZeroMQ send HWM {actual_sndhwm} message(s)")

        header = {
            "mode": "bytes" if args.total_bytes is not None else "duration",
            "chunk_size": args.chunk_size,
            "duration": args.duration,
            "total_bytes": args.total_bytes,
            "transport": "zmq-pair",
        }
        await socket.send(encode_json(header))

        ready = decode_json(await wait_for_interruptible(lambda: socket.recv(), stop_event))
        if ready.get("status") != "ready":
            message = ready.get("message", "receiver did not acknowledge the test")
            raise RuntimeError(str(message))

        target = f"{args.duration:.3f}s" if args.total_bytes is None else format_bytes(args.total_bytes)
        print(f"starting test, target {target}, chunk {format_bytes(args.chunk_size)}")

        sender_result = await run_send_loop(socket, args, stop_event)
        await socket.send(b"")

        summary = decode_json(await wait_for_interruptible(lambda: socket.recv(), stop_event))
        if summary.get("status") != "ok":
            message = summary.get("message", "receiver reported an error")
            raise RuntimeError(str(message))

        receiver_result = ThroughputResult(
            total_bytes=int(summary["total_bytes"]),
            elapsed_seconds=float(summary["elapsed_seconds"]),
        )

        print(f"sender   {summarize_result(sender_result)}")
        print(f"receiver {summarize_result(receiver_result)}")
        print("note: ZeroMQ can queue on the sender side, so receiver throughput is the more trustworthy number")
        if receiver_result.total_bytes != sender_result.total_bytes:
            print(
                "warning: sender and receiver byte counts differ "
                f"({sender_result.total_bytes} sent vs {receiver_result.total_bytes} received)"
            )
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
