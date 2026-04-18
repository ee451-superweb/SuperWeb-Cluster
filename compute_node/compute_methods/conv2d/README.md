# Spatial Convolution

This directory owns the native runner files for the
`conv2d` method.

It is shared by:

- `compute_node/performance_metrics/conv2d/`
  - benchmark orchestration, autotuning, and result reporting
- `compute_node/compute_methods/conv2d/executor.py`
  - runtime task execution on compute nodes

The important structure is:

- `cpu/`
  - platform-specific C++ CPU runners plus their local build directories
- `cuda/`
  - CUDA runner source plus its local build directory
- `dx12/`
  - Windows DirectX 12 compute runner source plus its local build directory
- `metal/`
  - Metal host/kernel sources plus their local build directory
- `paths.py`
  - one shared place to resolve method-specific source and executable paths

Datasets now live under `compute_node/input_matrix/conv2d/`, and
benchmark Python code lives under `compute_node/performance_metrics/conv2d/`.
