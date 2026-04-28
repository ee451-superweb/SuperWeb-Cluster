"""Define stable workspace paths for the GEMV benchmark.

Use this module whenever benchmark code needs the canonical method-local
dataset, result, or dataset-generation paths for GEMV.
"""

from __future__ import annotations

from pathlib import Path

from core.constants import METHOD_GEMV

METHOD_NAME = METHOD_GEMV
DISPLAY_NAME = "Fixed Matrix-Vector Multiplication"
METHOD_DIR = Path(__file__).resolve().parent
RESULT_PATH = METHOD_DIR / "result.json"
DATASET_DIR = METHOD_DIR.parents[1] / "input_matrix" / METHOD_NAME / "generated"
GENERATE_SCRIPT_PATH = METHOD_DIR.parents[1] / "input_matrix" / METHOD_NAME / "generate.py"
