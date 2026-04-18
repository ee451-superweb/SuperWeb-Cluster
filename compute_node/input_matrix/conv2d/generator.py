"""Conv2d dataset generation helpers."""

from __future__ import annotations

import json
from typing import Callable

from compute_node.input_matrix.generator import write_float32_file

from .spec import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_INPUT_SEED,
    DEFAULT_WEIGHT_SEED,
    GENERATOR_ALGORITHM,
    DatasetLayout,
    Conv2dSpec,
)


def generate_dataset(
    layout: DatasetLayout,
    spec: Conv2dSpec,
    *,
    skip_weight: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
    generator_workers: int | None = None,
    chunk_values: int | None = None,
) -> None:
    """Generate one conv2d dataset variant and write its metadata.

    Args:
        layout: Dataset layout that should receive the generated files.
        spec: Conv2d dataset specification to generate.
        skip_weight: Whether the weight file may be omitted intentionally.
        progress: Optional progress callback for streaming writes.
        generator_workers: Optional worker count for parallel generation.
        chunk_values: Optional chunk size in float32 values.
    """
    layout.root_dir.mkdir(parents=True, exist_ok=True)
    chunk_values = max(1, chunk_values or DEFAULT_CHUNK_VALUES)

    input_sha256 = write_float32_file(
        layout.input_path,
        total_values=spec.h * spec.w * spec.c_in,
        seed=DEFAULT_INPUT_SEED,
        chunk_values=chunk_values,
        label=layout.input_path.name,
        progress=progress,
        worker_count=generator_workers,
    )

    weight_sha256 = "skipped"
    if not skip_weight:
        weight_sha256 = write_float32_file(
            layout.weight_path,
            total_values=spec.k * spec.k * spec.c_in * spec.c_out,
            seed=DEFAULT_WEIGHT_SEED,
            chunk_values=chunk_values,
            label=layout.weight_path.name,
            progress=progress,
            worker_count=generator_workers,
        )

    metadata = {
        "benchmark": {
            "name": spec.name,
            "h": spec.h,
            "w": spec.w,
            "c_in": spec.c_in,
            "c_out": spec.c_out,
            "k": spec.k,
            "pad": spec.pad,
            "stride": spec.stride,
            "dtype": "float32",
            "endianness": "little",
            "operation": "Conv2D",
        },
        "files": {
            "input_feature_map": {
                "path": layout.input_path.name,
                "bytes": spec.input_bytes,
                "sha256": input_sha256,
                "seed": f"0x{DEFAULT_INPUT_SEED:016X}",
                "algorithm": GENERATOR_ALGORITHM,
                "shape": [spec.h, spec.w, spec.c_in],
            },
            "weights": {
                "path": layout.weight_path.name,
                "bytes": 0 if skip_weight else spec.weight_bytes,
                "sha256": weight_sha256,
                "seed": f"0x{DEFAULT_WEIGHT_SEED:016X}",
                "algorithm": GENERATOR_ALGORITHM,
                "skipped": skip_weight,
                "shape": [spec.c_out, spec.k, spec.k, spec.c_in],
            },
        },
    }
    layout.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
