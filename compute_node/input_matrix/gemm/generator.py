"""GEMM dataset generation helpers."""

from __future__ import annotations

import json
from typing import Callable

from compute_node.input_matrix.generator import write_float32_file

from .spec import (
    DEFAULT_A_SEED,
    DEFAULT_B_SEED,
    DEFAULT_CHUNK_VALUES,
    DatasetLayout,
    GemmSpec,
    GENERATOR_ALGORITHM,
)


def generate_dataset(
    layout: DatasetLayout,
    spec: GemmSpec,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    generator_workers: int | None = None,
    chunk_values: int | None = None,
) -> None:
    """Generate A and B files for one GEMM dataset variant, plus metadata."""

    layout.root_dir.mkdir(parents=True, exist_ok=True)
    chunk_values = max(chunk_values or DEFAULT_CHUNK_VALUES, spec.k)

    a_sha256 = write_float32_file(
        layout.a_path,
        total_values=spec.m * spec.k,
        seed=DEFAULT_A_SEED,
        chunk_values=chunk_values,
        label=layout.a_path.name,
        progress=progress,
        worker_count=generator_workers,
    )
    b_sha256 = write_float32_file(
        layout.b_path,
        total_values=spec.k * spec.n,
        seed=DEFAULT_B_SEED,
        chunk_values=chunk_values,
        label=layout.b_path.name,
        progress=progress,
        worker_count=generator_workers,
    )

    metadata = {
        "dataset": {
            "m": spec.m,
            "n": spec.n,
            "k": spec.k,
            "dtype": "float32",
            "endianness": "little",
            "layout": "row-major",
        },
        "files": {
            "a": {
                "path": layout.a_path.name,
                "bytes": spec.a_bytes,
                "sha256": a_sha256,
                "seed": f"0x{DEFAULT_A_SEED:016X}",
                "algorithm": GENERATOR_ALGORITHM,
                "shape": [spec.m, spec.k],
            },
            "b": {
                "path": layout.b_path.name,
                "bytes": spec.b_bytes,
                "sha256": b_sha256,
                "seed": f"0x{DEFAULT_B_SEED:016X}",
                "algorithm": GENERATOR_ALGORITHM,
                "shape": [spec.k, spec.n],
            },
        },
    }
    layout.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
