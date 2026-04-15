"""Compatibility re-exports for spatial-convolution dataset helpers."""

from compute_node.input_matrix.spatial_convolution import (  # noqa: F401
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    generate_dataset,
    get_runtime_input_matrix_spec,
    get_test_input_matrix_spec,
)
