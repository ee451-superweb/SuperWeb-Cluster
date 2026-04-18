"""Re-export the conv2d dataset helpers used by benchmarks.

Use this module when benchmark code wants a stable import path for dataset
layout and generation helpers without reaching into the input-matrix package
directly.
"""

from compute_node.input_matrix.conv2d import (  # noqa: F401
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_prefix_for_size,
    dataset_is_generated,
    generate_dataset,
    get_large_input_matrix_spec,
    get_small_input_matrix_spec,
)
