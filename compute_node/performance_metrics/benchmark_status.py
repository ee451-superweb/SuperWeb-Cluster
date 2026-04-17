"""Persist crash-survivable benchmark progress snapshots.

Use this module when benchmark entrypoints need to emit a durable current-step
JSON file plus an append-only trace so operators can recover context after a
crash, reset, or forced reboot.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_PATH_ENV = "SUPERWEB_BENCHMARK_STATUS_PATH"
TRACE_PATH_ENV = "SUPERWEB_BENCHMARK_TRACE_PATH"
RUN_ID_ENV = "SUPERWEB_BENCHMARK_RUN_ID"


def _json_default(value: Any) -> Any:
    """Convert non-JSON-native helper types into serializable values.

    Use this as the ``default`` hook for ``json.dumps`` whenever benchmark
    status payloads include ``Path`` objects or sets.

    Args:
        value: Arbitrary object that JSON could not serialize directly.

    Returns:
        A JSON-serializable replacement value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one status payload through JSON round-tripping.

    Use this before writing benchmark status files so unsupported helper types
    are converted into plain JSON-compatible structures.

    Args:
        payload: Status payload assembled by a caller.

    Returns:
        A JSON-safe dictionary ready to write to disk.
    """
    return json.loads(json.dumps(payload, default=_json_default))


def default_status_paths(output_path: Path) -> tuple[Path, Path]:
    """Derive default status and trace file paths from one output path.

    Use this when the caller did not provide explicit status destinations and
    the benchmark should write companion files next to the main report.

    Args:
        output_path: Final benchmark-report JSON path.

    Returns:
        The default ``(status_json_path, trace_jsonl_path)`` pair.
    """
    stem = output_path.stem if output_path.suffix else output_path.name
    return (
        output_path.with_name(f"{stem}_status.json"),
        output_path.with_name(f"{stem}_trace.jsonl"),
    )


def resolve_status_paths(
    *,
    output_path: Path,
    status_path: Path | None = None,
    trace_path: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve explicit or default benchmark status-output destinations.

    Use this in CLI entrypoints so optional overrides still fall back to the
    standard status and trace file names beside the main output file.

    Args:
        output_path: Main benchmark-report JSON path.
        status_path: Optional override for the current-status JSON file.
        trace_path: Optional override for the append-only trace file.

    Returns:
        The resolved ``(status_json_path, trace_jsonl_path)`` pair.
    """
    default_status_path, default_trace_path = default_status_paths(output_path)
    return (
        default_status_path if status_path is None else Path(status_path),
        default_trace_path if trace_path is None else Path(trace_path),
    )


def configure_status_environment(
    *,
    status_path: Path,
    trace_path: Path,
    run_id: str | None = None,
) -> str:
    """Publish status-file targets through environment variables.

    Use this before spawning helper subprocesses so every benchmark stage can
    append to the same status and trace outputs.

    Args:
        status_path: Current-status JSON path.
        trace_path: Append-only JSONL trace path.
        run_id: Optional explicit benchmark run identifier.

    Returns:
        The run id that was stored in the environment.
    """
    resolved_run_id = run_id or uuid.uuid4().hex
    os.environ[STATUS_PATH_ENV] = str(status_path)
    os.environ[TRACE_PATH_ENV] = str(trace_path)
    os.environ[RUN_ID_ENV] = resolved_run_id
    return resolved_run_id


def status_logging_enabled() -> bool:
    """Return whether status logging has been configured for this process.

    Use this in helpers that may run outside the top-level CLI so they can skip
    status writes when no output locations were configured.

    Args:
        None.

    Returns:
        ``True`` when both status and trace paths exist in the environment.
    """
    return bool(os.environ.get(STATUS_PATH_ENV) and os.environ.get(TRACE_PATH_ENV))


def _status_targets() -> tuple[Path, Path, str] | None:
    """Read the configured status-output targets from the environment.

    Use this inside low-level emit helpers so all benchmark stages share the
    same destination files and run id without threading parameters everywhere.

    Args:
        None.

    Returns:
        The resolved ``(status_path, trace_path, run_id)`` triple, or ``None``.
    """
    status_text = os.environ.get(STATUS_PATH_ENV)
    trace_text = os.environ.get(TRACE_PATH_ENV)
    if not status_text or not trace_text:
        return None
    run_id = os.environ.get(RUN_ID_ENV) or uuid.uuid4().hex
    return Path(status_text), Path(trace_text), run_id


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one JSON file with a new payload.

    Use this for the current-status snapshot so readers never observe a partial
    file if the process crashes in the middle of a write.

    Args:
        path: Destination JSON path.
        payload: Status payload that should replace the file contents.

    Returns:
        ``None`` after the file has been atomically replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one status event to the trace JSONL file.

    Use this for the historical trace so each benchmark step leaves behind a
    durable append-only record in addition to the latest snapshot.

    Args:
        path: Destination JSONL path.
        payload: Event payload to append.

    Returns:
        ``None`` after the line has been flushed to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def emit_status(event: str, **fields: Any) -> None:
    """Write one benchmark status snapshot plus one trace event.

    Use this from benchmark entrypoints, dataset generation, and backend runs to
    expose the current step to operators and crash recovery tools.

    Args:
        event: Stable event name describing the current benchmark step.
        **fields: Additional JSON-serializable fields for the event payload.

    Returns:
        ``None`` after the snapshot and trace entry have been written.
    """
    targets = _status_targets()
    if targets is None:
        return

    status_path, trace_path, run_id = targets
    now = time.time()
    payload = _normalize_payload(
        {
            "run_id": run_id,
            "event": event,
            "timestamp_unix": now,
            "timestamp_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "pid": os.getpid(),
            **fields,
        }
    )
    _write_json_atomic(status_path, payload)
    _append_jsonl(trace_path, payload)


def mark_benchmark_started(*, argv: list[str], cwd: Path, output_path: Path, methods: list[str]) -> None:
    """Emit the standard benchmark-started status event.

    Args:
        argv: CLI arguments used to launch the benchmark.
        cwd: Working directory for the benchmark run.
        output_path: Final report destination.
        methods: Methods scheduled for this benchmark invocation.

    Returns:
        ``None`` after the event has been written.
    """
    emit_status(
        "benchmark.started",
        status="running",
        argv=argv,
        cwd=str(cwd),
        output_path=str(output_path),
        methods=methods,
    )


def mark_benchmark_finished(*, output_path: Path, methods_completed: list[str], elapsed_seconds: float) -> None:
    """Emit the standard benchmark-finished status event.

    Args:
        output_path: Final report destination.
        methods_completed: Methods that completed successfully.
        elapsed_seconds: Total wall-clock time for the benchmark run.

    Returns:
        ``None`` after the event has been written.
    """
    emit_status(
        "benchmark.finished",
        status="completed",
        output_path=str(output_path),
        methods_completed=methods_completed,
        elapsed_seconds=elapsed_seconds,
    )


def mark_benchmark_failed(*, output_path: Path, error: str, methods_started: list[str]) -> None:
    """Emit the standard benchmark-failed status event.

    Args:
        output_path: Intended final report destination.
        error: Human-readable failure message.
        methods_started: Methods that had started before the failure.

    Returns:
        ``None`` after the event has been written.
    """
    emit_status(
        "benchmark.failed",
        status="failed",
        output_path=str(output_path),
        error=error,
        methods_started=methods_started,
    )
