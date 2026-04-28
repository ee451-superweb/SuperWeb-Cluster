"""Define stable workspace paths for the conv2d benchmark.

Use this module whenever conv2d benchmark code needs the canonical dataset,
result, benchmark-entry, or dataset-generation paths.
"""

from __future__ import annotations

from pathlib import Path

from core.constants import METHOD_CONV2D

METHOD_NAME = METHOD_CONV2D
DISPLAY_NAME = "Spatial Convolution"
METHOD_DIR = Path(__file__).resolve().parent
RESULT_PATH = METHOD_DIR / "result.json"
DATASET_DIR = METHOD_DIR.parents[1] / "input_matrix" / METHOD_NAME / "generated"
GENERATE_SCRIPT_PATH = METHOD_DIR.parents[1] / "input_matrix" / METHOD_NAME / "generate.py"
RAW_BENCHMARK_PATH = METHOD_DIR / "benchmark.py"
