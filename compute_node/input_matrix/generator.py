"""Shared streaming float32 generation helpers for `compute_node/input_matrix/`."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .spec import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_MATRIX_SEED,
    DEFAULT_VECTOR_SEED,
    GENERATOR_ALGORITHM,
    HASH_READ_BYTES,
    PROGRESS_STEP_BYTES,
    DatasetLayout,
    InputMatrixSpec,
)
from .splitmix import np, float32_chunk_from_counter


def _hash_file_sha256(path: Path) -> str:
    """Hash one generated file in a streaming pass."""

    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_READ_BYTES)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _generator_worker_count(requested_workers: int | None, total_values: int, chunk_values: int) -> int:
    """Choose a conservative worker count for dataset generation."""

    resolved_workers = max(1, int(requested_workers)) if requested_workers is not None else 0
    if total_values <= chunk_values:
        return 1
    if np is None:
        return resolved_workers or 1

    cpu_count = os.cpu_count() or 1
    automatic_workers = max(1, min(cpu_count, 8))
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


def _write_chunk_to_existing_file(path: str, offset_bytes: int, count: int, start_index: int, seed: int) -> int:
    """Worker entrypoint that fills one byte range inside an already-open file."""

    chunk = float32_chunk_from_counter(count, start_index, seed)
    with Path(path).open("r+b", buffering=0) as handle:
        handle.seek(offset_bytes)
        handle.write(chunk)
    return len(chunk)


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
    plan = _build_chunk_plan(total_values, chunk_values)

    _remove_if_exists(temp_path)
    try:
        with temp_path.open("wb") as handle:
            handle.truncate(total_bytes)

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _write_chunk_to_existing_file,
                    str(temp_path),
                    offset_bytes,
                    current_chunk_values,
                    start_index,
                    seed,
                )
                for offset_bytes, current_chunk_values, start_index in plan
            ]
            for future in concurrent.futures.as_completed(futures):
                completed_bytes += int(future.result())
                if progress:
                    progress(label, completed_bytes, total_bytes)

        sha256_hex = _hash_file_sha256(temp_path)
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
    next_progress_bytes = PROGRESS_STEP_BYTES

    _remove_if_exists(temp_path)
    try:
        with temp_path.open("wb") as handle:
            while written_values < total_values:
                current_chunk_values = min(chunk_values, total_values - written_values)
                chunk = float32_chunk_from_counter(current_chunk_values, written_values, seed)
                handle.write(chunk)
                sha256.update(chunk)
                written_values += current_chunk_values
                written_bytes = written_values * 4
                handle.flush()

                if progress and (written_bytes >= next_progress_bytes or written_values == total_values):
                    progress(label, written_bytes, total_bytes)
                    while next_progress_bytes <= written_bytes:
                        next_progress_bytes += PROGRESS_STEP_BYTES

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
