"""Re-export the spatial-convolution dataset helpers used by benchmarks.

Use this module when benchmark code wants a stable import path for dataset
layout and generation helpers without reaching into the input-matrix package
directly.
"""

from compute_node.input_matrix.spatial_convolution import (  # noqa: F401
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_is_generated,
    generate_dataset,
    get_runtime_input_matrix_spec,
    get_test_input_matrix_spec,
)
