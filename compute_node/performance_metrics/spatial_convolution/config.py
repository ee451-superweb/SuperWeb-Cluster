"""Stable paths for the spatial-convolution benchmark workspace."""

from __future__ import annotations

from pathlib import Path

from app.constants import METHOD_SPATIAL_CONVOLUTION

METHOD_NAME = METHOD_SPATIAL_CONVOLUTION
DISPLAY_NAME = "Spatial Convolution"
METHOD_DIR = Path(__file__).resolve().parent
RESULT_PATH = METHOD_DIR / "result.json"
DATASET_DIR = METHOD_DIR.parents[1] / "input_matrix" / METHOD_NAME / "generated"
GENERATE_SCRIPT_PATH = METHOD_DIR.parents[1] / "input_matrix" / METHOD_NAME / "generate.py"
RAW_BENCHMARK_PATH = (
    METHOD_DIR.parents[1]
    / "compute_methods"
    / METHOD_NAME
    / "performance_metrics"
    / "benchmark.py"
)
