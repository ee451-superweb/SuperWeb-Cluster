#!/usr/bin/env python3
"""Minimal TCP throughput sender for standalone LAN testing."""

from __future__ import annotations

import argparse
import socket
import time

from tcp_speed import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_DURATION,
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
        description="Connect to a TCP throughput receiver and measure sender/receiver throughput."
    )
    parser.add_argument("host", help="Receiver IPv4 or hostname.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Receiver TCP port.")
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help="Seconds allowed for TCP connect.",
    )
    parser.add_argument(
        "--chunk-size",
        type=parse_byte_count,
        default=DEFAULT_CHUNK_SIZE,
        help="Bytes per send call, for example 256KiB or 1MiB.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Seconds between progress prints. Use 0 to disable.",
    )
    parser.add_argument(
        "--send-buffer",
        type=parse_byte_count,
        default=0,
        help="Optional SO_SNDBUF override such as 4MiB. Default keeps the OS value.",
    )
    parser.add_argument("--tcp-nodelay", action="store_true", help="Enable TCP_NODELAY on the socket.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--duration", type=float, help="Seconds to keep sending data.")
    group.add_argument("--bytes", dest="total_bytes", type=parse_byte_count, help="Exact total bytes to send.")

    args = parser.parse_args()
    if args.duration is None and args.total_bytes is None:
        args.duration = DEFAULT_DURATION
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval cannot be negative")
    return args


def print_progress(total_sent: int, start_time: float, now: float) -> None:
    result = ThroughputResult(total_sent, max(now - start_time, 1e-9))
    print(f"progress {summarize_result(result)}")


def run_send_loop(sock: socket.socket, args: argparse.Namespace) -> ThroughputResult:
    payload = b"\0" * args.chunk_size
    total_sent = 0
    start_time = time.perf_counter()
    next_progress = start_time + args.progress_interval

    if args.total_bytes is not None:
        remaining = args.total_bytes
        while remaining > 0:
            block_size = min(args.chunk_size, remaining)
            sock.sendall(payload[:block_size])
            total_sent += block_size
            remaining -= block_size

            now = time.perf_counter()
            if args.progress_interval > 0 and now >= next_progress:
                print_progress(total_sent, start_time, now)
                next_progress = now + args.progress_interval
    else:
        deadline = start_time + args.duration
        while True:
            sock.sendall(payload)
            total_sent += args.chunk_size

            now = time.perf_counter()
            if args.progress_interval > 0 and now >= next_progress:
                print_progress(total_sent, start_time, now)
                next_progress = now + args.progress_interval
            if now >= deadline:
                break

    end_time = time.perf_counter()
    return ThroughputResult(total_sent, max(end_time - start_time, 0.0))


def main() -> int:
    args = parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=args.connect_timeout)
    try:
        sock.settimeout(None)
        if args.send_buffer > 0:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.send_buffer)
        if args.tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        actual_send_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print(f"connected to {args.host}:{args.port}")
        print(f"socket send buffer {format_bytes(actual_send_buffer)}")

        control = {
            "mode": "bytes" if args.total_bytes is not None else "duration",
            "chunk_size": args.chunk_size,
            "duration": args.duration,
            "total_bytes": args.total_bytes,
        }
        send_json_line(sock, control)

        ready = recv_json_line(sock)
        if ready.get("status") != "ready":
            message = ready.get("message", "receiver did not acknowledge the test")
            raise RuntimeError(str(message))

        target = f"{args.duration:.3f}s" if args.total_bytes is None else format_bytes(args.total_bytes)
        print(f"starting test, target {target}, chunk {format_bytes(args.chunk_size)}")

        sender_result = run_send_loop(sock, args)
        sock.shutdown(socket.SHUT_WR)

        summary = recv_json_line(sock)
        if summary.get("status") != "ok":
            message = summary.get("message", "receiver reported an error")
            raise RuntimeError(str(message))

        receiver_result = ThroughputResult(
            total_bytes=int(summary["total_bytes"]),
            elapsed_seconds=float(summary["elapsed_seconds"]),
        )

        print(f"sender   {summarize_result(sender_result)}")
        print(f"receiver {summarize_result(receiver_result)}")
        if receiver_result.total_bytes != sender_result.total_bytes:
            print(
                "warning: sender and receiver byte counts differ "
                f"({sender_result.total_bytes} sent vs {receiver_result.total_bytes} received)"
            )
        return 0
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
