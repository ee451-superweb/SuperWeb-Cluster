"""FMVM input-dataset package."""

from .generator import generate_dataset
from .spec import (
    DEFAULT_CHUNK_VALUES,
    DEFAULT_COLS,
    DEFAULT_MATRIX_SEED,
    DEFAULT_ROWS,
    DEFAULT_VECTOR_SEED,
    TEST_COLS,
    TEST_ROWS,
    DatasetLayout,
    InputMatrixSpec,
    build_dataset_layout,
    build_input_matrix_spec,
    get_runtime_input_matrix_spec,
    get_test_input_matrix_spec,
)
from .storage import compare_float32_vectors, dataset_is_generated, load_float32_file

__all__ = [
    "DEFAULT_CHUNK_VALUES",
    "DEFAULT_COLS",
    "DEFAULT_MATRIX_SEED",
    "DEFAULT_ROWS",
    "DEFAULT_VECTOR_SEED",
    "TEST_COLS",
    "TEST_ROWS",
    "DatasetLayout",
    "InputMatrixSpec",
    "build_dataset_layout",
    "build_input_matrix_spec",
    "compare_float32_vectors",
    "dataset_is_generated",
    "generate_dataset",
    "get_runtime_input_matrix_spec",
    "get_test_input_matrix_spec",
    "load_float32_file",
]
