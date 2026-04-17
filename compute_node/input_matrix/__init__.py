"""Input matrix dataset package.

Top-level exports remain FMVM-compatible for existing callers, while method-local
packages under `fixed_matrix_vector_multiplication/` and `spatial_convolution/`
provide symmetric dataset generation flows.
"""

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION

from .fixed_matrix_vector_multiplication import (
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
    "METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION",
    "METHOD_SPATIAL_CONVOLUTION",
]
