# Compute Methods

`compute_node/compute_methods/` owns the actual method implementations that a
compute node can execute.

Right now this folder contains one method:

- `gemv`

That method directory is shared by two higher-level workflows:

- `compute_node/performance_metrics/`
  - benchmarks each hardware backend
  - autotunes the best config for the method
- `compute_node/task_executor.py`
  - reuses the same CPU/CUDA/Metal runners in task mode during real runtime work

The DX12 source tree is still present for debugging, but its benchmark and
runtime entry points are disabled in this build because the module can trigger
fatal system instability on some Windows machines.

This split keeps the file structure easier to read:

- `compute_methods/`
  - method-specific source code and build outputs
- `performance_metrics/`
  - benchmarking orchestration, scoring, and reporting
