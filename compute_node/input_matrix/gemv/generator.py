"""GEMV dataset generation helpers."""

from __future__ import annotations

import json
from typing import Callable

from compute_node.input_matrix.generator import write_float32_file

from .spec import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_MATRIX_SEED,
    DEFAULT_VECTOR_SEED,
    GENERATOR_ALGORITHM,
    DatasetLayout,
    InputMatrixSpec,
)


def generate_dataset(
    layout: DatasetLayout,
    spec: InputMatrixSpec,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    generator_workers: int | None = None,
    chunk_values: int | None = None,
) -> None:
    """Generate one GEMV dataset variant and write its metadata.

    Args:
        layout: Dataset layout that should receive the generated files.
        spec: GEMV dataset specification to generate.
        progress: Optional progress callback for streaming writes.
        generator_workers: Optional worker count for parallel generation.
        chunk_values: Optional chunk size in float32 values.
    """
    layout.root_dir.mkdir(parents=True, exist_ok=True)
    chunk_values = max(chunk_values or DEFAULT_CHUNK_VALUES, spec.cols)

    matrix_sha256 = write_float32_file(
        layout.matrix_path,
        total_values=spec.rows * spec.cols,
        seed=DEFAULT_MATRIX_SEED,
        chunk_values=chunk_values,
        label=layout.matrix_path.name,
        progress=progress,
        worker_count=generator_workers,
    )
    vector_sha256 = write_float32_file(
        layout.vector_path,
        total_values=spec.cols,
        seed=DEFAULT_VECTOR_SEED,
        chunk_values=chunk_values,
        label=layout.vector_path.name,
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
