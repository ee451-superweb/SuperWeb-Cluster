"""Input matrix dataset package.

Top-level exports remain GEMV-compatible for existing callers, while method-local
packages under `gemv/` and `conv2d/`
provide symmetric dataset generation flows.
"""

from core.constants import METHOD_GEMV, METHOD_CONV2D

from .gemv import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_COLS,
    DEFAULT_MATRIX_SEED,
    DEFAULT_ROWS,
    DEFAULT_VECTOR_SEED,
    DatasetLayout,
    InputMatrixSpec,
    build_dataset_layout,
    build_input_matrix_spec,
    compare_float32_vectors,
    dataset_is_generated,
    generate_dataset,
    load_float32_file,
)

__all__ = [
    "DEFAULT_CHUNK_VALUES",
    "DEFAULT_COLS",
    "DEFAULT_MATRIX_SEED",
    "DEFAULT_ROWS",
    "DEFAULT_VECTOR_SEED",
    "DatasetLayout",
    "InputMatrixSpec",
    "build_dataset_layout",
    "build_input_matrix_spec",
    "compare_float32_vectors",
    "dataset_is_generated",
    "generate_dataset",
    "load_float32_file",
    "METHOD_GEMV",
    "METHOD_CONV2D",
]
