"""Dataset layout validation and inspection helpers for GEMV input data."""

from __future__ import annotations

import json
from pathlib import Path

from .spec import DatasetLayout, GENERATOR_ALGORITHM, InputMatrixSpec, build_dataset_layout


def dataset_is_generated(layout: DatasetLayout, spec: InputMatrixSpec) -> bool:
    """Return whether the GEMV dataset layout already matches the requested spec.

    Args:
        layout: Dataset layout to validate.
        spec: GEMV spec that the dataset is expected to match.

    Returns:
        ``True`` when all files and metadata match the requested spec.
    """
    if not layout.matrix_path.exists() or not layout.vector_path.exists() or not layout.meta_path.exists():
        return False
    if layout.matrix_path.stat().st_size != spec.matrix_bytes:
        return False
    if layout.vector_path.stat().st_size != spec.vector_bytes:
        return False
    try:
        metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    dataset_metadata = metadata.get("dataset", {})
    if dataset_metadata.get("rows") != spec.rows:
        return False
    if dataset_metadata.get("cols") != spec.cols:
        return False
    if dataset_metadata.get("dtype") != "float32":
        return False

    files_metadata = metadata.get("files", {})
    matrix_metadata = files_metadata.get("matrix", {})
    vector_metadata = files_metadata.get("vector", {})
    if matrix_metadata.get("algorithm") != GENERATOR_ALGORITHM:
        return False
    if vector_metadata.get("algorithm") != GENERATOR_ALGORITHM:
        return False
    return True


def load_float32_file(path: Path) -> list[float]:
    """Load a float32 binary file into a Python float list."""
    with path.open("rb") as handle:
        raw = handle.read()
    return list(memoryview(raw).cast("f"))


def compare_float32_vectors(reference: list[float], candidate: list[float]) -> tuple[float, float, int, int]:
    """Compare two float vectors and report max absolute and relative errors.

    Args:
        reference: Expected float values.
        candidate: Observed float values to compare.

    Returns:
        ``(max_abs_error, max_rel_error, max_abs_index, max_rel_index)``.
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


__all__ = [
    "DatasetLayout",
    "InputMatrixSpec",
    "build_dataset_layout",
    "compare_float32_vectors",
    "dataset_is_generated",
    "load_float32_file",
]
