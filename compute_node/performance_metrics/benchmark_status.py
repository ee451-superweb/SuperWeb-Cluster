"""Persistent benchmark status logging that survives crashes or hard resets."""

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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=_json_default))


def default_status_paths(output_path: Path) -> tuple[Path, Path]:
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
    resolved_run_id = run_id or uuid.uuid4().hex
    os.environ[STATUS_PATH_ENV] = str(status_path)
    os.environ[TRACE_PATH_ENV] = str(trace_path)
    os.environ[RUN_ID_ENV] = resolved_run_id
    return resolved_run_id


def status_logging_enabled() -> bool:
    return bool(os.environ.get(STATUS_PATH_ENV) and os.environ.get(TRACE_PATH_ENV))


def _status_targets() -> tuple[Path, Path, str] | None:
    status_text = os.environ.get(STATUS_PATH_ENV)
    trace_text = os.environ.get(TRACE_PATH_ENV)
    if not status_text or not trace_text:
        return None
    run_id = os.environ.get(RUN_ID_ENV) or uuid.uuid4().hex
    return Path(status_text), Path(trace_text), run_id


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def emit_status(event: str, **fields: Any) -> None:
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
    emit_status(
        "benchmark.started",
        status="running",
        argv=argv,
        cwd=str(cwd),
        output_path=str(output_path),
        methods=methods,
    )


def mark_benchmark_finished(*, output_path: Path, methods_completed: list[str], elapsed_seconds: float) -> None:
    emit_status(
        "benchmark.finished",
        status="completed",
        output_path=str(output_path),
        methods_completed=methods_completed,
        elapsed_seconds=elapsed_seconds,
    )


def mark_benchmark_failed(*, output_path: Path, error: str, methods_started: list[str]) -> None:
    emit_status(
        "benchmark.failed",
        status="failed",
        output_path=str(output_path),
        error=error,
        methods_started=methods_started,
    )
