"""Deterministic binary dataset generation for fixed matrix-vector workloads."""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from array import array
from pathlib import Path

from models import DatasetPaths, WorkloadSpec

MASK64 = 0xFFFFFFFFFFFFFFFF
MASK32 = 0xFFFFFFFF
DEFAULT_MATRIX_SEED = 0x123456789ABCDEF0
DEFAULT_VECTOR_SEED = 0x0FEDCBA987654321
F32 = struct.Struct("<f")


def _splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = (z ^ (z >> 30)) & MASK64
    z = (z * 0xBF58476D1CE4E5B9) & MASK64
    z = (z ^ (z >> 27)) & MASK64
    z = (z * 0x94D049BB133111EB) & MASK64
    z = (z ^ (z >> 31)) & MASK64
    return state, z


def _u32_to_unit_float(u: int) -> float:
    return (u / 2**31) - 1.0


def _f32_bytes_from_prng(count: int, state: int) -> tuple[bytes, int]:
    out = bytearray(count * 4)
    offset = 0
    for _ in range(count):
        state, rnd64 = _splitmix64_next(state)
        value = _u32_to_unit_float(rnd64 & MASK32)
        out[offset : offset + 4] = F32.pack(value)
        offset += 4
    return bytes(out), state


def _write_float32_file(path: Path, total_count: int, seed: int, chunk_count: int) -> str:
    sha = hashlib.sha256()
    state = seed
    written = 0

    with path.open("wb") as handle:
        while written < total_count:
            count = min(chunk_count, total_count - written)
            payload, state = _f32_bytes_from_prng(count, state)
            handle.write(payload)
            sha.update(payload)
            written += count

    return sha.hexdigest()


def ensure_dataset(root_dir: Path, workload: WorkloadSpec) -> DatasetPaths:
    """Generate the row-major matrix and vector files when missing."""

    dataset_dir = root_dir / workload.preset / f"{workload.rows}x{workload.cols}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = dataset_dir / f"A_f32_{workload.rows}x{workload.cols}.bin"
    vector_path = dataset_dir / f"x_f32_{workload.cols}.bin"
    meta_path = dataset_dir / "Ax_meta.json"
    transposed_matrix_path = dataset_dir / f"AT_f32_{workload.cols}x{workload.rows}.bin"

    if not matrix_path.exists():
        _write_float32_file(
            matrix_path,
            total_count=workload.rows * workload.cols,
            seed=DEFAULT_MATRIX_SEED,
            chunk_count=max(workload.cols * 32, workload.cols),
        )

    if not vector_path.exists():
        _write_float32_file(
            vector_path,
            total_count=workload.cols,
            seed=DEFAULT_VECTOR_SEED,
            chunk_count=workload.cols,
        )

    meta = {
        "workload": {
            "name": workload.name,
            "preset": workload.preset,
            "rows": workload.rows,
            "cols": workload.cols,
            "flops_per_run": workload.flops_per_run,
            "ideal_seconds": workload.ideal_seconds,
            "zero_score_seconds": workload.zero_score_seconds,
        },
        "A": {
            "file": matrix_path.name,
            "dtype": "float32",
            "endianness": "little",
            "order": "row-major",
            "bytes": workload.rows * workload.cols * 4,
            "seed": f"0x{DEFAULT_MATRIX_SEED:016X}",
        },
        "x": {
            "file": vector_path.name,
            "dtype": "float32",
            "endianness": "little",
            "shape": [workload.cols],
            "bytes": workload.cols * 4,
            "seed": f"0x{DEFAULT_VECTOR_SEED:016X}",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return DatasetPaths(
        root_dir=dataset_dir,
        matrix_path=matrix_path,
        vector_path=vector_path,
        meta_path=meta_path,
        transposed_matrix_path=transposed_matrix_path,
    )


def ensure_transposed_matrix(dataset: DatasetPaths, workload: WorkloadSpec) -> Path:
    """Create and cache the transposed matrix layout."""

    if dataset.transposed_matrix_path.exists():
        return dataset.transposed_matrix_path

    with dataset.matrix_path.open("rb") as source_handle, dataset.transposed_matrix_path.open("wb") as target_handle:
        source_map = mmap.mmap(source_handle.fileno(), 0, access=mmap.ACCESS_READ)
        source_values = memoryview(source_map).cast("f")
        try:
            row_buffer = array("f", [0.0]) * workload.rows
            for col in range(workload.cols):
                for row in range(workload.rows):
                    row_buffer[row] = source_values[row * workload.cols + col]
                row_buffer.tofile(target_handle)
        finally:
            source_values.release()
            source_map.close()

    return dataset.transposed_matrix_path


def load_vector(dataset: DatasetPaths) -> list[float]:
    """Load the input vector into Python memory once."""

    return load_float32_file(dataset.vector_path)


def load_float32_file(path: Path) -> list[float]:
    """Load one little-endian float32 binary file into Python memory."""

    with path.open("rb") as handle:
        raw = handle.read()
    return list(memoryview(raw).cast("f"))


def compute_reference_output(dataset: DatasetPaths, workload: WorkloadSpec) -> list[float]:
    """Compute a deterministic single-process reference output."""

    vector_values = load_vector(dataset)
    with dataset.matrix_path.open("rb") as handle:
        matrix_map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        matrix_values = memoryview(matrix_map).cast("f")
        try:
            output = [0.0] * workload.rows
            for row in range(workload.rows):
                base = row * workload.cols
                acc = 0.0
                for col in range(workload.cols):
                    acc += matrix_values[base + col] * vector_values[col]
                output[row] = acc
            return output
        finally:
            matrix_values.release()
            matrix_map.close()


def compare_vectors(
    reference: list[float],
    candidate: list[float],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    """Return verification status plus max absolute/relative errors."""

    if len(reference) != len(candidate):
        raise ValueError("vector lengths do not match")

    max_abs_error = 0.0
    max_rel_error = 0.0
    verified = True

    for expected, observed in zip(reference, candidate):
        abs_error = abs(expected - observed)
        scale = abs(expected) * rtol + atol
        rel_error = abs_error / max(abs(expected), atol)
        if abs_error > max_abs_error:
            max_abs_error = abs_error
        if rel_error > max_rel_error:
            max_rel_error = rel_error
        if abs_error > scale:
            verified = False

    return verified, max_abs_error, max_rel_error


def checksum_float32(values: list[float]) -> str:
    """Return a stable checksum for one output vector."""

    sha = hashlib.sha256()
    for value in values:
        sha.update(F32.pack(float(value)))
    return sha.hexdigest()
