"""Shared streaming float32 generation helpers for `compute_node/input_matrix/`."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import io
from pathlib import Path
from typing import Callable

from .spec import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_MATRIX_SEED,
    DEFAULT_VECTOR_SEED,
    GENERATOR_ALGORITHM,
    DatasetLayout,
    InputMatrixSpec,
)
from .splitmix import np, float32_chunk_from_counter

WRITE_SLICE_BYTES = 4 * 1024 * 1024
AUTO_GENERATOR_WORKER_CAP = 16


def _write_generated_chunk(
    handle: io.BufferedWriter,
    sha256: hashlib._Hash,
    chunk: bytes | bytearray,
    *,
    label: str,
    completed_bytes: int,
    total_bytes: int,
    progress: Callable[[str, int, int], None] | None = None,
) -> int:
    """Write one generated chunk in smaller slices to smooth I/O and progress updates."""

    chunk_view = memoryview(chunk)
    try:
        for offset in range(0, len(chunk_view), WRITE_SLICE_BYTES):
            piece = chunk_view[offset:offset + WRITE_SLICE_BYTES]
            handle.write(piece)
            sha256.update(piece)
            completed_bytes += len(piece)
            if progress:
                progress(label, completed_bytes, total_bytes)
    finally:
        chunk_view.release()
    return completed_bytes


def _generator_worker_count(requested_workers: int | None, total_values: int, chunk_values: int) -> int:
    """Choose a conservative worker count for dataset generation."""

    resolved_workers = max(1, int(requested_workers)) if requested_workers is not None else 0
    if total_values <= 0:
        return 1

    cpu_count = os.cpu_count() or 1
    automatic_workers = max(1, min(cpu_count, AUTO_GENERATOR_WORKER_CAP))
    return resolved_workers or automatic_workers


def _build_chunk_plan(total_values: int, chunk_values: int) -> list[tuple[int, int, int]]:
    """Return `(offset_bytes, value_count, start_index)` for each file chunk."""

    plan: list[tuple[int, int, int]] = []
    written_values = 0
    while written_values < total_values:
        current_chunk_values = min(chunk_values, total_values - written_values)
        plan.append((written_values * 4, current_chunk_values, written_values))
        written_values += current_chunk_values
    return plan


def _parallel_chunk_values(total_values: int, chunk_values: int, worker_count: int) -> int:
    """Split large logical chunks further so multiple workers can actually help."""

    if worker_count <= 1 or total_values <= 1:
        return max(1, chunk_values)

    min_parallel_chunks = max(2, worker_count * 2)
    current_chunks = max(1, math.ceil(total_values / chunk_values))
    if current_chunks >= min_parallel_chunks:
        return max(1, chunk_values)

    adjusted_chunk_values = math.ceil(total_values / min_parallel_chunks)
    min_reasonable_chunk_values = max(1, (1024 * 1024) // 4)
    if total_values > min_reasonable_chunk_values:
        adjusted_chunk_values = max(min_reasonable_chunk_values, adjusted_chunk_values)
    return max(1, min(chunk_values, adjusted_chunk_values))


def _generate_chunk_bytes(task: tuple[int, int, int]) -> bytes | bytearray:
    """Worker entrypoint that returns one deterministic float32 byte chunk."""

    count, start_index, seed = task
    return float32_chunk_from_counter(count, start_index, seed)


def _remove_if_exists(path: Path) -> None:
    """Delete a temporary file without failing if it is already gone."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_float32_file_parallel(
    path: Path,
    total_values: int,
    seed: int,
    chunk_values: int,
    *,
    label: str,
    progress: Callable[[str, int, int], None] | None = None,
    worker_count: int,
) -> str:
    """Generate one float32 file by filling independent chunks in parallel."""

    total_bytes = total_values * 4
    temp_path = path.with_suffix(path.suffix + ".tmp")
    completed_bytes = 0
    effective_chunk_values = _parallel_chunk_values(total_values, chunk_values, worker_count)
    plan = _build_chunk_plan(total_values, effective_chunk_values)
    sha256 = hashlib.sha256()
    executor_cls = concurrent.futures.ThreadPoolExecutor if np is not None else concurrent.futures.ProcessPoolExecutor

    _remove_if_exists(temp_path)
    try:
        with temp_path.open("wb") as handle, executor_cls(max_workers=worker_count) as executor:
            chunk_iter = executor.map(
                _generate_chunk_bytes,
                ((current_chunk_values, start_index, seed) for _offset_bytes, current_chunk_values, start_index in plan),
            )
            for (_offset_bytes, current_chunk_values, _start_index), chunk in zip(plan, chunk_iter):
                chunk_bytes = bytes(chunk)
                expected_bytes = current_chunk_values * 4
                if len(chunk_bytes) != expected_bytes:
                    raise ValueError(
                        f"{label} generated {len(chunk_bytes)} bytes for a {current_chunk_values}-value chunk; "
                        f"expected {expected_bytes}"
                    )
                completed_bytes = _write_generated_chunk(
                    handle,
                    sha256,
                    chunk_bytes,
                    label=label,
                    completed_bytes=completed_bytes,
                    total_bytes=total_bytes,
                    progress=progress,
                )

        sha256_hex = sha256.hexdigest()
        temp_path.replace(path)
        return sha256_hex
    except Exception:
        _remove_if_exists(temp_path)
        raise


def write_float32_file(
    path: Path,
    total_values: int,
    seed: int,
    chunk_values: int,
    *,
    label: str,
    progress: Callable[[str, int, int], None] | None = None,
    worker_count: int | None = None,
) -> str:
    """Write one deterministic float32 file without holding the whole file in RAM."""

    total_bytes = total_values * 4
    resolved_worker_count = _generator_worker_count(worker_count, total_values, chunk_values)
    if resolved_worker_count > 1:
        return _write_float32_file_parallel(
            path,
            total_values,
            seed,
            chunk_values,
            label=label,
            progress=progress,
            worker_count=resolved_worker_count,
        )

    temp_path = path.with_suffix(path.suffix + ".tmp")
    sha256 = hashlib.sha256()
    written_values = 0

    _remove_if_exists(temp_path)
    try:
        with temp_path.open("wb") as handle:
            while written_values < total_values:
                current_chunk_values = min(chunk_values, total_values - written_values)
                chunk = float32_chunk_from_counter(current_chunk_values, written_values, seed)
                written_bytes = written_values * 4
                written_bytes = _write_generated_chunk(
                    handle,
                    sha256,
                    chunk,
                    label=label,
                    completed_bytes=written_bytes,
                    total_bytes=total_bytes,
                    progress=progress,
                )
                written_values += current_chunk_values

        temp_path.replace(path)
        return sha256.hexdigest()
    except Exception:
        _remove_if_exists(temp_path)
        raise


def generate_dataset(
    layout: DatasetLayout,
    spec: InputMatrixSpec,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    generator_workers: int | None = None,
    chunk_values: int | None = None,
) -> None:
    """Generate deterministic `A.bin`, `x.bin`, and generic dataset metadata."""

    layout.root_dir.mkdir(parents=True, exist_ok=True)
    chunk_values = max(chunk_values or DEFAULT_CHUNK_VALUES, spec.cols)

    matrix_sha256 = write_float32_file(
        layout.matrix_path,
        total_values=spec.rows * spec.cols,
        seed=DEFAULT_MATRIX_SEED,
        chunk_values=chunk_values,
        label="A.bin",
        progress=progress,
        worker_count=generator_workers,
    )
    vector_sha256 = write_float32_file(
        layout.vector_path,
        total_values=spec.cols,
        seed=DEFAULT_VECTOR_SEED,
        chunk_values=chunk_values,
        label="x.bin",
        progress=progress,
        worker_count=1,
    )

    metadata = {
        "dataset": {
            "rows": spec.rows,
            "cols": spec.cols,
            "dtype": "float32",
            "endianness": "little",
            "matrix_layout": "row-major",
            "vector_shape": [spec.cols],
        },
        "files": {
            "matrix": {
                "path": layout.matrix_path.name,
                "bytes": spec.matrix_bytes,
                "sha256": matrix_sha256,
                "seed": f"0x{DEFAULT_MATRIX_SEED:016X}",
                "algorithm": GENERATOR_ALGORITHM,
                "order": "row-major",
            },
            "vector": {
                "path": layout.vector_path.name,
                "bytes": spec.vector_bytes,
                "sha256": vector_sha256,
                "seed": f"0x{DEFAULT_VECTOR_SEED:016X}",
                "algorithm": GENERATOR_ALGORITHM,
                "shape": [spec.cols],
            },
        },
    }
    layout.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = [
    "generate_dataset",
    "write_float32_file",
]
