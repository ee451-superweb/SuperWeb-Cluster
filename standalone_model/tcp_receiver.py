#!/usr/bin/env python3
"""Minimal TCP throughput receiver for standalone LAN testing."""

from __future__ import annotations

import argparse
import asyncio
import socket
import time
from dataclasses import dataclass, field

from tcp_speed import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PORT,
    DEFAULT_PROGRESS_INTERVAL,
    INTERRUPT_POLL_INTERVAL,
    MAX_CONTROL_LINE,
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
class SessionState:
    session_id: str
    expected_streams: int
    mode: str
    duration: float | None
    connected_streams: set[int] = field(default_factory=set)
    completed_streams: set[int] = field(default_factory=set)
    total_bytes: int = 0
    start_time: float | None = None
    end_time: float | None = None
    active_handlers: int = 0
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    failure_message: str | None = None


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def register_stream(
        self,
        *,
        session_id: str,
        streams: int,
        stream_index: int,
        mode: str,
        duration: float | None,
    ) -> SessionState:
        session = self._sessions.get(session_id)
        if session is None:
            session = SessionState(
                session_id=session_id,
                expected_streams=streams,
                mode=mode,
                duration=duration,
            )
            self._sessions[session_id] = session
        elif session.expected_streams != streams or session.mode != mode:
            raise ValueError("session parameters do not match earlier streams")

        if stream_index in session.connected_streams:
            raise ValueError(f"duplicate stream index {stream_index}")

        session.connected_streams.add(stream_index)
        session.active_handlers += 1
        if len(session.connected_streams) == session.expected_streams:
            session.ready_event.set()
        return session

    async def wait_until_ready(self, session: SessionState, timeout: float) -> None:
        if not session.ready_event.is_set():
            try:
                await asyncio.wait_for(session.ready_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.fail_session(session, "timed out waiting for all streams to connect")

        if session.failure_message is not None:
            raise RuntimeError(session.failure_message)

    def add_stream_result(
        self,
        session: SessionState,
        *,
        stream_index: int,
        total_bytes: int,
        start_time: float | None,
        end_time: float,
    ) -> None:
        if stream_index in session.completed_streams:
            raise ValueError(f"duplicate result for stream {stream_index}")

        session.completed_streams.add(stream_index)
        session.total_bytes += total_bytes
        if start_time is not None:
            session.start_time = start_time if session.start_time is None else min(session.start_time, start_time)
        session.end_time = end_time if session.end_time is None else max(session.end_time, end_time)

        if len(session.completed_streams) == session.expected_streams:
            session.done_event.set()

    def fail_session(self, session: SessionState, message: str) -> None:
        if session.failure_message is None:
            session.failure_message = message
        session.ready_event.set()
        session.done_event.set()

    async def wait_for_session_result(self, session: SessionState, timeout: float) -> ThroughputResult:
        if not session.done_event.is_set():
            try:
                await asyncio.wait_for(session.done_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.fail_session(session, "timed out waiting for all streams to finish")

        if session.failure_message is not None:
            raise RuntimeError(session.failure_message)

        elapsed = 0.0
        if session.start_time is not None and session.end_time is not None:
            elapsed = max(session.end_time - session.start_time, 0.0)
        return ThroughputResult(session.total_bytes, elapsed)

    def release_stream(self, session: SessionState) -> None:
        session.active_handlers -= 1
        if session.active_handlers <= 0:
            existing = self._sessions.get(session.session_id)
            if existing is session:
                self._sessions.pop(session.session_id, None)


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


def validate_control_message(
    payload: dict[str, object],
) -> tuple[str, float | None, int | None, int, str, int, int]:
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

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")

    stream_index = payload.get("stream_index")
    streams = payload.get("streams")
    if not isinstance(stream_index, int) or stream_index < 0:
        raise ValueError("stream_index must be a non-negative integer")
    if not isinstance(streams, int) or streams <= 0:
        raise ValueError("streams must be a positive integer")
    if stream_index >= streams:
        raise ValueError("stream_index must be smaller than streams")

    return mode, (float(duration) if duration is not None else None), total_bytes, chunk_size, session_id, stream_index, streams


def session_wait_timeout(mode: str, duration: float | None, idle_timeout: float) -> float:
    if mode == "duration" and duration is not None:
        return duration + idle_timeout + 5.0
    return idle_timeout + 5.0


async def read_payload(reader: asyncio.StreamReader, size: int, idle_timeout: float) -> bytes:
    return await asyncio.wait_for(reader.read(size), timeout=idle_timeout)


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    args: argparse.Namespace,
    registry: SessionRegistry,
    completed_session: asyncio.Event,
    stop_event: asyncio.Event,
) -> None:
    session: SessionState | None = None
    sock = writer.get_extra_info("socket")
    if sock is None:
        writer.close()
        await writer.wait_closed()
        return

    if args.recv_buffer > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.recv_buffer)

    actual_recv_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    peername = writer.get_extra_info("peername")
    peer_host, peer_port = peername[0], peername[1]
    print(f"accepted connection from {peer_host}:{peer_port}")
    print(f"socket receive buffer {format_bytes(actual_recv_buffer)}")

    try:
        control = await wait_for_interruptible(
            lambda: async_recv_json_line(reader),
            stop_event,
            timeout=args.idle_timeout,
        )
        mode, duration, total_bytes, chunk_size, session_id, stream_index, streams = validate_control_message(control)
        read_size = min(max(chunk_size, 64 * 1024), 1024 * 1024)

        session = registry.register_stream(
            session_id=session_id,
            streams=streams,
            stream_index=stream_index,
            mode=mode,
            duration=duration,
        )
        await wait_for_interruptible(
            lambda: registry.wait_until_ready(session, args.idle_timeout),
            stop_event,
        )

        await async_send_json_line(
            writer,
            {
                "status": "ready",
                "read_size": read_size,
            },
        )

        expected = f"{duration:.3f}s" if mode == "duration" else format_bytes(total_bytes or 0)
        print(
            f"session {session_id} stream {stream_index}/{streams - 1} "
            f"mode {mode}, target {expected}, sender chunk {format_bytes(chunk_size)}"
        )

        start_time: float | None = None
        total_received = 0
        next_progress = 0.0

        while True:
            chunk = await wait_for_interruptible(
                lambda: read_payload(reader, read_size, args.idle_timeout),
                stop_event,
            )
            if not chunk:
                break

            now = time.perf_counter()
            if start_time is None:
                start_time = now
                next_progress = start_time + args.progress_interval

            total_received += len(chunk)

            if args.progress_interval > 0 and now >= next_progress:
                progress = ThroughputResult(total_received, max(now - start_time, 1e-9))
                print(f"session {session_id} stream {stream_index} progress {summarize_result(progress)}")
                next_progress = now + args.progress_interval

        end_time = time.perf_counter()
        stream_result = ThroughputResult(
            total_received,
            0.0 if start_time is None else max(end_time - start_time, 0.0),
        )
        registry.add_stream_result(
            session,
            stream_index=stream_index,
            total_bytes=stream_result.total_bytes,
            start_time=start_time,
            end_time=end_time,
        )
        session_result = await wait_for_interruptible(
            lambda: registry.wait_for_session_result(
                session,
                session_wait_timeout(mode, duration, args.idle_timeout),
            ),
            stop_event,
        )

        await async_send_json_line(
            writer,
            {
                "status": "ok",
                "stream_total_bytes": stream_result.total_bytes,
                "stream_elapsed_seconds": stream_result.elapsed_seconds,
                "stream_mib_per_s": stream_result.mib_per_s,
                "stream_mbit_per_s": stream_result.mbit_per_s,
                "session_total_bytes": session_result.total_bytes,
                "session_elapsed_seconds": session_result.elapsed_seconds,
                "session_mib_per_s": session_result.mib_per_s,
                "session_mbit_per_s": session_result.mbit_per_s,
                "streams": streams,
            },
        )
        print(f"session {session_id} stream {stream_index} completed {summarize_result(stream_result)}")
        if stream_index == 0:
            print(f"session {session_id} aggregate {summarize_result(session_result)} across {streams} stream(s)")
            if args.once:
                completed_session.set()
    except UserInterrupted:
        if session is not None:
            registry.fail_session(session, "stopped by user")
    except (ConnectionError, OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
        print(f"client {peer_host}:{peer_port} failed: {exc}")
        if session is not None:
            registry.fail_session(session, str(exc))
        try:
            await async_send_json_line(writer, {"status": "error", "message": str(exc)})
        except (ConnectionError, OSError, ValueError):
            pass
    finally:
        if session is not None:
            registry.release_stream(session)
        writer.close()
        await writer.wait_closed()


async def async_main(args: argparse.Namespace) -> int:
    stop_event, restore_handlers = install_interrupt_event(asyncio.get_running_loop())
    registry = SessionRegistry()
    completed_session = asyncio.Event()

    server = await asyncio.start_server(
        lambda reader, writer: handle_client(reader, writer, args, registry, completed_session, stop_event),
        host=args.bind,
        port=args.port,
        limit=MAX_CONTROL_LINE,
    )

    actual_port = server.sockets[0].getsockname()[1]
    print(f"listening on {args.bind}:{actual_port}")

    try:
        async with server:
            await server.start_serving()
            while True:
                if stop_event.is_set():
                    return 130
                if args.once and completed_session.is_set():
                    return 0
                await sleep_interruptible(INTERRUPT_POLL_INTERVAL, stop_event)
    except UserInterrupted:
        return 130
    finally:
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
