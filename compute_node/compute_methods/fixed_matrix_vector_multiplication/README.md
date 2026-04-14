# Fixed Matrix Vector Multiplication

This directory owns the actual implementation files for the
`fixed_matrix_vector_multiplication` method.

It is shared by:

- `compute_node/performance_metrics/`
  - benchmark orchestration, autotuning, and result reporting
- `compute_node/task_executor.py`
  - runtime task execution on locally registered processors

The important structure is:

- `cpu/`
  - platform-specific C++ CPU runners plus their local build directories
- `cuda/`
  - CUDA runner source plus its local build directory
- `dx12/`
  - Windows DirectX 12 compute runner source plus its local build directory
  - reserved for non-NVIDIA adapters so it acts as the Windows-native path for AMD/Intel GPUs
  - uses a thread-group reduction kernel so adjacent lanes walk adjacent matrix
    columns instead of striding across rows
  - compute-node runtime can keep one resident DX12 worker process alive so the
    fixed matrix is uploaded once at startup and reused until shutdown
- `metal/`
  - Metal host/kernel sources plus their local build directory
- `paths.py`
  - one shared place to resolve method-specific source and executable paths

This keeps method code discoverable without burying it inside the benchmark
workspace.
