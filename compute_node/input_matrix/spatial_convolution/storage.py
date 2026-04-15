"""Dataset layout validation helpers for spatial-convolution input data."""

from __future__ import annotations

import json

from .spec import DatasetLayout, GENERATOR_ALGORITHM, SpatialConvolutionSpec


def dataset_is_generated(
    layout: DatasetLayout,
    spec: SpatialConvolutionSpec,
    *,
    skip_weight: bool = False,
) -> bool:
    if not layout.input_path.exists() or not layout.meta_path.exists():
        return False
    if not skip_weight and not layout.weight_path.exists():
        return False

    try:
        metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    benchmark = metadata.get("benchmark", {})
    if benchmark.get("h") != spec.h or benchmark.get("w") != spec.w:
        return False
    if benchmark.get("c_in") != spec.c_in or benchmark.get("c_out") != spec.c_out:
        return False
    if benchmark.get("k") != spec.k or benchmark.get("pad") != spec.pad:
        return False
    if benchmark.get("stride", 1) != spec.stride:
        return False

    files = metadata.get("files", {})
    input_metadata = files.get("input_feature_map", {})
    weights_metadata = files.get("weights", {})
    if input_metadata.get("algorithm") != GENERATOR_ALGORITHM:
        return False
    if not skip_weight and weights_metadata.get("algorithm") != GENERATOR_ALGORITHM:
        return False
    return True
