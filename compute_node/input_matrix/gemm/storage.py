"""Dataset validation helpers for GEMM input data."""

from __future__ import annotations

import json

from .spec import DatasetLayout, GENERATOR_ALGORITHM, GemmSpec


def dataset_is_generated(layout: DatasetLayout, spec: GemmSpec) -> bool:
    """Return whether the GEMM dataset matches the requested spec."""

    if not layout.a_path.exists() or not layout.b_path.exists() or not layout.meta_path.exists():
        return False
    if layout.a_path.stat().st_size != spec.a_bytes:
        return False
    if layout.b_path.stat().st_size != spec.b_bytes:
        return False
    try:
        metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    dataset_metadata = metadata.get("dataset", {})
    if dataset_metadata.get("m") != spec.m:
        return False
    if dataset_metadata.get("n") != spec.n:
        return False
    if dataset_metadata.get("k") != spec.k:
        return False
    files_metadata = metadata.get("files", {})
    a_metadata = files_metadata.get("a", {})
    b_metadata = files_metadata.get("b", {})
    if a_metadata.get("algorithm") != GENERATOR_ALGORITHM:
        return False
    if b_metadata.get("algorithm") != GENERATOR_ALGORITHM:
        return False
    return True


__all__ = ["dataset_is_generated"]
