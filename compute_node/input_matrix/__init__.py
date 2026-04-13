"""Input matrix dataset package."""

from .generator import generate_dataset
from .spec import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_COLS,
    DEFAULT_MATRIX_SEED,
    DEFAULT_ROWS,
    DEFAULT_VECTOR_SEED,
    DatasetLayout,
    InputMatrixSpec,
    build_dataset_layout,
    build_input_matrix_spec,
)
from .storage import compare_float32_vectors, dataset_is_generated, load_float32_file

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
]
