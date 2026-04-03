#!/usr/bin/env python3
"""Minimal TCP throughput receiver for standalone LAN testing."""

from __future__ import annotations

import argparse
import socket
import time

from tcp_speed import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PORT,
    DEFAULT_PROGRESS_INTERVAL,
    ThroughputResult,
    format_bytes,
    parse_byte_count,
    recv_json_line,
    send_json_line,
    summarize_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for a TCP throughput test and report received throughput."
    )
    parser.add_argument("--bind", default="0.0.0.0", help="IPv4 address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port to listen on.")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity before aborting a client connection.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Seconds between progress prints. Use 0 to disable.",
    )
    parser.add_argument(
        "--recv-buffer",
        type=parse_byte_count,
        default=0,
        help="Optional SO_RCVBUF override such as 4MiB. Default keeps the OS value.",
    )
    parser.add_argument("--once", action="store_true", help="Exit after the first completed test.")
    args = parser.parse_args()
    if args.idle_timeout <= 0:
        parser.error("--idle-timeout must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval cannot be negative")
    return args


def validate_control_message(payload: dict[str, object]) -> tuple[str, float | None, int | None, int]:
    mode = payload.get("mode")
    if mode not in {"duration", "bytes"}:
        raise ValueError(f"unsupported mode: {mode!r}")

    chunk_size = payload.get("chunk_size")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    duration = payload.get("duration")
    if duration is not None and not isinstance(duration, (int, float)):
        raise ValueError("duration must be numeric when present")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")

    total_bytes = payload.get("total_bytes")
    if total_bytes is not None and (not isinstance(total_bytes, int) or total_bytes <= 0):
        raise ValueError("total_bytes must be a positive integer when present")

    if mode == "duration" and duration is None:
        raise ValueError("duration mode requires duration")
    if mode == "bytes" and total_bytes is None:
        raise ValueError("bytes mode requires total_bytes")

    return mode, (float(duration) if duration is not None else None), total_bytes, chunk_size


def handle_client(conn: socket.socket, addr: tuple[str, int], args: argparse.Namespace) -> bool:
    conn.settimeout(args.idle_timeout)
    if args.recv_buffer > 0:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.recv_buffer)

    actual_recv_buffer = conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"accepted connection from {addr[0]}:{addr[1]}")
    print(f"socket receive buffer {format_bytes(actual_recv_buffer)}")

    try:
        control = recv_json_line(conn)
        mode, duration, total_bytes, chunk_size = validate_control_message(control)
        read_size = min(max(chunk_size, 64 * 1024), 1024 * 1024)

        send_json_line(
            conn,
            {
                "status": "ready",
                "read_size": read_size,
            },
        )

        expected = f"{duration:.3f}s" if mode == "duration" else format_bytes(total_bytes or 0)
        print(f"test mode {mode}, target {expected}, sender chunk {format_bytes(chunk_size)}")

        start_time: float | None = None
        total_received = 0
        next_progress = 0.0

        while True:
            chunk = conn.recv(read_size)
            if not chunk:
                break

            now = time.perf_counter()
            if start_time is None:
                start_time = now
                next_progress = start_time + args.progress_interval

            total_received += len(chunk)

            if args.progress_interval > 0 and now >= next_progress:
                progress = ThroughputResult(total_received, max(now - start_time, 1e-9))
                print(f"progress {summarize_result(progress)}")
                next_progress = now + args.progress_interval

        end_time = time.perf_counter()
        elapsed = 0.0 if start_time is None else max(end_time - start_time, 0.0)
        result = ThroughputResult(total_received, elapsed)

        send_json_line(
            conn,
            {
                "status": "ok",
                "total_bytes": result.total_bytes,
                "elapsed_seconds": result.elapsed_seconds,
                "mib_per_s": result.mib_per_s,
                "mbit_per_s": result.mbit_per_s,
            },
        )
        print(f"completed {summarize_result(result)}")
        return True
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"client {addr[0]}:{addr[1]} failed: {exc}")
        try:
            send_json_line(conn, {"status": "error", "message": str(exc)})
        except OSError:
            pass
        return False
    finally:
        conn.close()


def main() -> int:
    args = parse_args()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen()

    actual_port = listener.getsockname()[1]
    print(f"listening on {args.bind}:{actual_port}")

    try:
        while True:
            conn, addr = listener.accept()
            completed = handle_client(conn, addr, args)
            if args.once and completed:
                return 0
    except KeyboardInterrupt:
        print("stopped by user")
        return 130
    finally:
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
