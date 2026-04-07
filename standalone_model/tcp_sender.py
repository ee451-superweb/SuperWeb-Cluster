#!/usr/bin/env python3
"""Minimal TCP throughput sender for standalone LAN testing."""

from __future__ import annotations

import argparse
import asyncio
import socket
import time
import uuid
from dataclasses import dataclass, field

from tcp_speed import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_DURATION,
    DEFAULT_PORT,
    DEFAULT_PROGRESS_INTERVAL,
    DEFAULT_STREAMS,
    INTERRUPT_POLL_INTERVAL,
    ThroughputResult,
    UserInterrupted,
    async_recv_json_line,
    async_send_json_line,
    format_bytes,
    install_interrupt_event,
    parse_byte_count,
    sleep_interruptible,
    summarize_result,
    wait_for_interruptible,
)


@dataclass(slots=True)
class SenderSessionState:
    streams: int
    bytes_sent: list[int] = field(init=False)
    sender_end_times: list[float | None] = field(init=False)
    start_time: float | None = None
    ready_count: int = 0
    start_event: asyncio.Event = field(default_factory=asyncio.Event)
    receiver_result: ThroughputResult | None = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bytes_sent = [0] * self.streams
        self.sender_end_times = [None] * self.streams

    def record_error(self, message: str) -> None:
        self.errors.append(message)
        self.start_event.set()

    def mark_ready(self) -> None:
        self.ready_count += 1
        if self.ready_count == self.streams:
            self.start_time = time.perf_counter()
            self.start_event.set()

    def add_bytes(self, stream_index: int, amount: int) -> None:
        self.bytes_sent[stream_index] += amount

    def mark_stream_end(self, stream_index: int, end_time: float) -> None:
        self.sender_end_times[stream_index] = end_time

    def snapshot(self) -> tuple[int, float | None]:
        return sum(self.bytes_sent), self.start_time

    def set_receiver_result(self, result: ThroughputResult) -> None:
        if self.receiver_result is None:
            self.receiver_result = result


@dataclass(slots=True)
class StreamOutcome:
    stream_index: int
    send_buffer: int
    sender_result: ThroughputResult


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
    parser.add_argument(
        "--streams",
        type=int,
        default=DEFAULT_STREAMS,
        help="Number of parallel TCP streams to open.",
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
    if args.streams <= 0:
        parser.error("--streams must be positive")
    if args.total_bytes is not None and args.total_bytes < args.streams:
        parser.error("--bytes must be at least as large as --streams")
    return args


def print_progress(total_sent: int, start_time: float, now: float) -> None:
    result = ThroughputResult(total_sent, max(now - start_time, 1e-9))
    print(f"progress {summarize_result(result)}")


def split_total_bytes(total_bytes: int, streams: int, stream_index: int) -> int:
    base = total_bytes // streams
    remainder = total_bytes % streams
    return base + (1 if stream_index < remainder else 0)


def stream_target_label(args: argparse.Namespace, stream_index: int) -> str:
    if args.total_bytes is None:
        return f"{args.duration:.3f}s"
    return format_bytes(split_total_bytes(args.total_bytes, args.streams, stream_index))


async def run_send_loop(
    writer: asyncio.StreamWriter,
    *,
    chunk_size: int,
    duration: float | None,
    total_bytes: int | None,
    stream_index: int,
    session_state: SenderSessionState,
    stop_event: asyncio.Event,
) -> ThroughputResult:
    payload = b"\0" * chunk_size
    total_sent = 0
    start_time = time.perf_counter()

    if total_bytes is not None:
        remaining = total_bytes
        while remaining > 0:
            if stop_event.is_set():
                raise UserInterrupted()
            block_size = min(chunk_size, remaining)
            writer.write(payload[:block_size])
            await writer.drain()
            total_sent += block_size
            remaining -= block_size
            session_state.add_bytes(stream_index, block_size)
    else:
        assert duration is not None
        deadline = start_time + duration
        while True:
            if stop_event.is_set():
                raise UserInterrupted()
            writer.write(payload)
            await writer.drain()
            total_sent += chunk_size
            session_state.add_bytes(stream_index, chunk_size)
            if time.perf_counter() >= deadline:
                break

    end_time = time.perf_counter()
    session_state.mark_stream_end(stream_index, end_time)
    return ThroughputResult(total_sent, max(end_time - start_time, 0.0))


async def open_stream(
    args: argparse.Namespace,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, socket.socket, int]:
    connect_task = asyncio.open_connection(args.host, args.port)
    reader, writer = await asyncio.wait_for(connect_task, timeout=args.connect_timeout)
    sock = writer.get_extra_info("socket")
    if sock is None:
        raise RuntimeError("socket not available from asyncio transport")

    if args.send_buffer > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.send_buffer)
    if args.tcp_nodelay:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    actual_send_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    return reader, writer, sock, actual_send_buffer


async def run_stream(
    stream_index: int,
    args: argparse.Namespace,
    *,
    session_id: str,
    session_state: SenderSessionState,
    outcomes: list[StreamOutcome | None],
    stop_event: asyncio.Event,
) -> None:
    stream_total_bytes = None
    if args.total_bytes is not None:
        stream_total_bytes = split_total_bytes(args.total_bytes, args.streams, stream_index)

    sock: socket.socket | None = None
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer, sock, actual_send_buffer = await open_stream(args)
        print(f"stream {stream_index} connected to {args.host}:{args.port}")
        print(f"stream {stream_index} socket send buffer {format_bytes(actual_send_buffer)}")

        control = {
            "mode": "bytes" if stream_total_bytes is not None else "duration",
            "chunk_size": args.chunk_size,
            "duration": args.duration if stream_total_bytes is None else None,
            "total_bytes": stream_total_bytes,
            "session_id": session_id,
            "stream_index": stream_index,
            "streams": args.streams,
        }
        await async_send_json_line(writer, control)

        ready = await wait_for_interruptible(lambda: async_recv_json_line(reader), stop_event)
        if ready.get("status") != "ready":
            message = ready.get("message", "receiver did not acknowledge the test")
            raise RuntimeError(str(message))

        print(f"stream {stream_index} ready, target {stream_target_label(args, stream_index)}")
        session_state.mark_ready()
        await wait_for_interruptible(lambda: session_state.start_event.wait(), stop_event)
        if session_state.errors:
            return

        sender_result = await run_send_loop(
            writer,
            chunk_size=args.chunk_size,
            duration=args.duration if stream_total_bytes is None else None,
            total_bytes=stream_total_bytes,
            stream_index=stream_index,
            session_state=session_state,
            stop_event=stop_event,
        )
        try:
            if sock is None:
                raise RuntimeError("socket missing for half-close")
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            writer.close()

        summary = await wait_for_interruptible(lambda: async_recv_json_line(reader), stop_event)
        if summary.get("status") != "ok":
            message = summary.get("message", "receiver reported an error")
            raise RuntimeError(str(message))

        session_state.set_receiver_result(
            ThroughputResult(
                total_bytes=int(summary["session_total_bytes"]),
                elapsed_seconds=float(summary["session_elapsed_seconds"]),
            )
        )
        outcomes[stream_index] = StreamOutcome(
            stream_index=stream_index,
            send_buffer=actual_send_buffer,
            sender_result=sender_result,
        )
    except UserInterrupted:
        session_state.record_error("stopped by user")
    except Exception as exc:
        session_state.record_error(f"stream {stream_index}: {exc}")
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()


def build_sender_result(session_state: SenderSessionState) -> ThroughputResult:
    total_sent, start_time = session_state.snapshot()
    if start_time is None:
        return ThroughputResult(total_sent, 0.0)

    end_times = [value for value in session_state.sender_end_times if value is not None]
    if not end_times:
        return ThroughputResult(total_sent, 0.0)
    return ThroughputResult(total_sent, max(max(end_times) - start_time, 0.0))


async def progress_worker(
    args: argparse.Namespace,
    session_state: SenderSessionState,
    stop_event: asyncio.Event,
) -> None:
    if args.progress_interval <= 0:
        return

    try:
        while True:
            await sleep_interruptible(args.progress_interval, stop_event)
            total_sent, start_time = session_state.snapshot()
            if start_time is not None:
                print_progress(total_sent, start_time, time.perf_counter())
            if session_state.errors or stop_event.is_set():
                return
            if all(end_time is not None for end_time in session_state.sender_end_times):
                return
    except UserInterrupted:
        return


def report_stream_results(outcomes: list[StreamOutcome | None]) -> None:
    for outcome in outcomes:
        if outcome is None:
            continue
        print(
            f"stream {outcome.stream_index} sender {summarize_result(outcome.sender_result)} "
            f"with socket buffer {format_bytes(outcome.send_buffer)}"
        )


async def async_main(args: argparse.Namespace) -> int:
    stop_event, restore_handlers = install_interrupt_event(asyncio.get_running_loop())
    session_id = uuid.uuid4().hex
    session_state = SenderSessionState(streams=args.streams)
    outcomes: list[StreamOutcome | None] = [None] * args.streams

    target = f"{args.duration:.3f}s" if args.total_bytes is None else format_bytes(args.total_bytes)
    print(f"starting test to {args.host}:{args.port} with {args.streams} stream(s), total target {target}")
    print(f"stream chunk size {format_bytes(args.chunk_size)}")

    tasks = [
        asyncio.create_task(
            run_stream(
                stream_index,
                args,
                session_id=session_id,
                session_state=session_state,
                outcomes=outcomes,
                stop_event=stop_event,
            )
        )
        for stream_index in range(args.streams)
    ]
    progress_task = asyncio.create_task(progress_worker(args, session_state, stop_event))

    try:
        while True:
            if stop_event.is_set():
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                return 130
            if all(task.done() for task in tasks):
                break
            await asyncio.sleep(INTERRUPT_POLL_INTERVAL)

        await asyncio.gather(*tasks)
        if not progress_task.done():
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        if session_state.errors:
            messages = [message for message in session_state.errors if message != "stopped by user"]
            if not messages and stop_event.is_set():
                return 130
            for message in messages or session_state.errors:
                print(f"error {message}")
            return 1

        sender_result = build_sender_result(session_state)
        receiver_result = session_state.receiver_result
        report_stream_results(outcomes)
        print(f"sender   {summarize_result(sender_result)} across {args.streams} stream(s)")
        if receiver_result is not None:
            print(f"receiver {summarize_result(receiver_result)} across {args.streams} stream(s)")
            if receiver_result.total_bytes != sender_result.total_bytes:
                print(
                    "warning: sender and receiver byte counts differ "
                    f"({sender_result.total_bytes} sent vs {receiver_result.total_bytes} received)"
                )
        return 0
    finally:
        if not progress_task.done():
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        restore_handlers()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("stopped by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
