# Performance Metrics

This workspace is the local compute-characterization stage for
`superweb-cluster` Sprint 1.

Its job is to answer:

- what hardware backends are available on this machine
- what their best measured GFLOPS are for the fixed workload
- how those backends should be ranked for later scheduling decisions

`bootstrap.py` is still the top-level project entrypoint. In normal use, this
benchmark workspace is invoked automatically when
`compute_node/performance metrics/result.json` is missing.

## How It Fits Into The Project

The current flow is:

1. `bootstrap.py` prepares `.venv` and dependencies.
2. `bootstrap.py` ensures this benchmark has produced `result.json`.
3. A compute node later loads that result summary during runtime registration.
4. It sends only the compact summary upstream:
   - hardware backend count
   - backend ranking
   - best effective GFLOPS per backend
5. The main node stores those reported backends as worker-hardware capability objects.
6. Sprint 2 scheduling will use that capability inventory to distribute load.

## Fixed Workload

The benchmark currently measures one method:

- `fixed_matrix_vector_multiplication`

The default fixed problem is:

- `A`: `16384 x 32768` float32, exactly `2 GiB`
- `x`: `32768` float32
- operation: `y = A x`

The production benchmark keeps this shape fixed so different machines compare
the same work.

## Control Flow

`benchmark.py` acts as the orchestrator:

1. resolve the benchmark shape
2. check whether `A.bin` and `x.bin` already exist
3. call `generate.py` only if the dataset is missing or mismatched
4. run all enabled backend adapters
5. keep the best result for each backend
6. rank backends and write `result.json`

## Main Files

- `benchmark.py`
  - top-level benchmark runner
  - handles dataset checks, backend execution, ranking, and JSON output
- `fmvm_dataset.py`
  - dataset layout and validation
  - deterministic streamed generation of `A.bin`, `x.bin`, and metadata
- `backends/`
  - one adapter per hardware family
  - keeps benchmark orchestration hardware-agnostic
- `backends/cpu_backend.py`
  - Windows CPU backend adapter
- `backends/cuda_backend.py`
  - CUDA backend adapter
- `fixed_matrix_vector_multiplication/cpu/windows/fmvm_cpu_windows.cpp`
  - CPU compute runner
- `fixed_matrix_vector_multiplication/cuda/fmvm_cuda_runner.cu`
  - CUDA compute runner

## Backends

- `cpu`
  - sweeps worker counts in binary-tree order such as `16 -> 8 -> 32 -> 4 -> 64`
  - sweeps multiple tile sizes
  - reads the dataset once and benchmarks candidate configs in memory
- `cuda`
  - enabled when `nvcc` and a CUDA-capable GPU are available
  - sweeps `transpose`, `block_size`, and `tile_size`
  - compiles for the detected compute capability such as `sm_89`
  - keeps the matrix-vector arithmetic in FP32

## Output Schema

`result.json` stores:

- `backend_results`
  - one best entry per backend such as `cpu` or `cuda`
- `ranking`
  - fastest to slowest among successful backends
- `best_backend`
  - the first item in `ranking`

That file is also the source for the compact runtime registration summary sent
by compute nodes to the main node.

## Usage

Run the benchmark with the default fixed dataset:

```bash
python "compute_node/performance metrics/benchmark.py"
```

Run only the CPU backend:

```bash
python "compute_node/performance metrics/benchmark.py" --backend cpu
```

Run only the CUDA backend:

```bash
python "compute_node/performance metrics/benchmark.py" --backend cuda
```

Generate the dataset without running the benchmark:

```bash
python "compute_node/input matrix/generate.py"
```

Write the report to a different path:

```bash
python "compute_node/performance metrics/benchmark.py" --output my_result.json
```

`--rows` and `--cols` remain available for tiny tests. When you use those
overrides without changing `--dataset-dir`, the generated files are written to
`compute_node/input matrix/generated/overrides/<rows>x<cols>/` so the main
production dataset does not get overwritten.

## Notes

- Generated datasets and benchmark reports are local-only and git-ignored.
- The checked-in exception is the pair of Windows benchmark executables so a
  fresh Windows checkout can run CPU/CUDA backends without rebuilding first.
- Cross-implementation validation uses FP32 tolerances rather than exact
  checksum equality.
- This workspace finished its Sprint 1 role: benchmark, rank, and summarize
  local compute hardware for runtime registration. Sprint 2 will consume that
  summary for actual scheduling decisions.
