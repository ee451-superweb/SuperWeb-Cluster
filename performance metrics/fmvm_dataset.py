"""Dataset helpers for the fixed matrix-vector benchmark.

This module does *not* run the benchmark itself.

Its only responsibilities are:

1. describe where the generated input files live
2. check whether those files already exist
3. generate deterministic `A.bin` and `x.bin` when asked

The key implementation detail here is that generation must be *streaming*.
For the default dataset, `A.bin` is 2 GiB, so building the whole matrix in RAM
would be wasteful. We therefore generate a small batch, write it immediately,
flush periodically, and then continue with the next batch.

If `benchmark.py` is the "orchestrator" and `backends/` are the "hardware
adapters", then this file is simply the "dataset manager".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from models import BenchmarkSpec, DatasetLayout

MASK32 = 0xFFFFFFFF
DEFAULT_MATRIX_SEED = 0x123456789ABCDEF0
DEFAULT_VECTOR_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 1_048_576  # 4 MiB per batch
PROGRESS_STEP_BYTES = 32 * 1024 * 1024


def build_dataset_layout(root_dir: Path) -> DatasetLayout:
    """Return the canonical paths for the generated benchmark inputs.

    We intentionally keep the filenames short and stable so the compute
    programs can simply read `A.bin` and `x.bin`.
    """

    return DatasetLayout(
        root_dir=root_dir,
        matrix_path=root_dir / "A.bin",
        vector_path=root_dir / "x.bin",
        meta_path=root_dir / "dataset_meta.json",
    )


def dataset_is_generated(layout: DatasetLayout, spec: BenchmarkSpec) -> bool:
    """Check whether the expected generated files already exist.

    This is the gate used by `benchmark.py` before it decides whether it needs
    to call `generate.py`.
    """

    if not layout.matrix_path.exists() or not layout.vector_path.exists() or not layout.meta_path.exists():
        return False
    if layout.matrix_path.stat().st_size != spec.matrix_bytes:
        return False
    if layout.vector_path.stat().st_size != spec.vector_bytes:
        return False
    return True


def _xorshift32_next(state: int) -> int:
    """Advance a tiny deterministic PRNG.

    The earlier generator built Python float objects one by one before writing
    anything, which made large datasets feel stuck. This xorshift32-based path
    is intentionally simpler and faster: we keep the state in a 32-bit integer,
    derive one float32 bit-pattern per step, and write batches immediately.
    """

    state ^= (state << 13) & MASK32
    state ^= (state >> 17) & MASK32
    state ^= (state << 5) & MASK32
    return state & MASK32


def _float32_word_from_state(state: int) -> int:
    """Map one PRNG state to one finite float32 bit-pattern.

    We store random-looking values directly as IEEE-754 float32 words so the
    generator can stay in integer arithmetic while producing legal float data.
    The sign bit is random and the exponent is fixed to 126, which keeps the
    magnitude in `[0.5, 1.0)`.
    """

    sign = (state & 0x1) << 31
    mantissa = (state >> 1) & 0x007FFFFF
    return sign | 0x3F000000 | mantissa


def _float32_chunk_from_prng(count: int, state: int) -> tuple[bytearray, int]:
    """Generate one small batch of deterministic float32 bytes.

    The returned `bytearray` is written to disk immediately by the caller. This
    is what makes the generator visibly stream data to the filesystem instead of
    appearing idle for a long time before the first write.
    """

    payload = bytearray(count * 4)
    words = memoryview(payload).cast("I")
    try:
        current_state = state & MASK32
        if current_state == 0:
            current_state = 0x6D2B79F5

        for index in range(count):
            current_state = _xorshift32_next(current_state)
            words[index] = _float32_word_from_state(current_state)
    finally:
        words.release()
    return payload, current_state


def _write_float32_file(
    path: Path,
    total_values: int,
    seed: int,
    chunk_values: int,
    *,
    label: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> str:
    """Write one deterministic float32 file without holding the whole file in RAM.

    `progress`, when provided, receives `(label, written_bytes, total_bytes)`.
    """

    sha256 = hashlib.sha256()
    state = seed & MASK32
    written_values = 0
    total_bytes = total_values * 4
    next_progress_bytes = PROGRESS_STEP_BYTES

    with path.open("wb") as handle:
        while written_values < total_values:
            current_chunk_values = min(chunk_values, total_values - written_values)
            chunk, state = _float32_chunk_from_prng(current_chunk_values, state)
            handle.write(chunk)
            sha256.update(chunk)
            written_values += current_chunk_values
            written_bytes = written_values * 4

            # Flush after each batch so the file size grows on disk while the
            # generator is still running. This makes long runs much less opaque.
            handle.flush()

            if progress and (written_bytes >= next_progress_bytes or written_values == total_values):
                progress(label, written_bytes, total_bytes)
                while next_progress_bytes <= written_bytes:
                    next_progress_bytes += PROGRESS_STEP_BYTES

    return sha256.hexdigest()


def generate_dataset(
    layout: DatasetLayout,
    spec: BenchmarkSpec,
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Generate deterministic `A.bin`, `x.bin`, and metadata for the benchmark."""

    layout.root_dir.mkdir(parents=True, exist_ok=True)

    # The matrix is much larger than the vector, so we keep the batch size
    # modest and write every batch immediately. This keeps RAM bounded and
    # makes progress visible on disk.
    matrix_sha256 = _write_float32_file(
        layout.matrix_path,
        total_values=spec.rows * spec.cols,
        seed=DEFAULT_MATRIX_SEED,
        chunk_values=max(DEFAULT_CHUNK_VALUES, spec.cols),
        label="A.bin",
        progress=progress,
    )
    vector_sha256 = _write_float32_file(
        layout.vector_path,
        total_values=spec.cols,
        seed=DEFAULT_VECTOR_SEED,
        chunk_values=max(DEFAULT_CHUNK_VALUES, spec.cols),
        label="x.bin",
        progress=progress,
    )

    metadata = {
        "benchmark": {
            "name": spec.name,
            "rows": spec.rows,
            "cols": spec.cols,
            "dtype": "float32",
            "endianness": "little",
            "operation": "y = A x",
        },
        "files": {
            "matrix": {
                "path": layout.matrix_path.name,
                "bytes": spec.matrix_bytes,
                "sha256": matrix_sha256,
                "seed": f"0x{DEFAULT_MATRIX_SEED:016X}",
                "algorithm": "xorshift32_to_float32_words",
                "order": "row-major",
            },
            "vector": {
                "path": layout.vector_path.name,
                "bytes": spec.vector_bytes,
                "sha256": vector_sha256,
                "seed": f"0x{DEFAULT_VECTOR_SEED:016X}",
                "algorithm": "xorshift32_to_float32_words",
                "shape": [spec.cols],
            },
        },
    }
    layout.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_float32_file(path: Path) -> list[float]:
    """Load one little-endian float32 binary file into Python memory.

    This helper is mostly for tests and inspection. The real benchmark work is
    done by the hardware-specific compute programs.
    """

    with path.open("rb") as handle:
        raw = handle.read()
    return list(memoryview(raw).cast("f"))


def compare_float32_vectors(reference: list[float], candidate: list[float]) -> tuple[float, float, int, int]:
    """Return max absolute/relative error plus their indices.

    We use this helper for cross-implementation checks where exact checksum
    matches are too strict, especially once different backends legitimately use
    different reduction orders while still staying within FP32 expectations.
    """

    if len(reference) != len(candidate):
        raise ValueError("vector lengths do not match")

    max_abs_error = 0.0
    max_rel_error = 0.0
    max_abs_index = -1
    max_rel_index = -1

    for index, (expected, observed) in enumerate(zip(reference, candidate)):
        abs_error = abs(expected - observed)
        rel_error = abs_error / max(abs(expected), 1.0e-12)

        if abs_error > max_abs_error:
            max_abs_error = abs_error
            max_abs_index = index
        if rel_error > max_rel_error:
            max_rel_error = rel_error
            max_rel_index = index

    return max_abs_error, max_rel_error, max_abs_index, max_rel_index
