# Performance Metrics

This workspace benchmarks one method for now:

- `fixed_matrix_vector_multiplication`

The current simplified flow is:

1. make sure `A.bin` and `x.bin` exist
2. if they do not exist, call `generate.py`
3. detect which backend compute programs should run
4. run those programs and keep the best result for each backend
5. write `result.json`

## Fixed Dataset

By default the benchmark uses one fixed problem:

- `A`: `16384 x 32768` float32, exactly `2 GiB`
- `x`: `32768` float32
- operation: `y = A x`

The production benchmark uses this fixed shape so different machines are
comparing the same work.

## What Each File Does

- `benchmark.py`
  - top-level orchestrator
  - checks whether generated inputs exist
  - calls `generate.py` when needed
  - runs the backend adapters, ranks their best results, and writes `result.json`
- `fmvm_dataset.py`
  - dataset manager only
  - knows where `A.bin`, `x.bin`, and `dataset_meta.json` live
  - knows how to generate them deterministically
  - does not run any compute benchmark itself
- `backends/`
  - one Python adapter per hardware family
  - each backend knows how to build and launch its matching compute program
  - `benchmark.py` stays hardware-agnostic by delegating to this folder
- `backends/cpu_backend.py`
  - Windows CPU adapter
  - compiles and runs `fixed_matrix_vector_multiplication/cpu/windows/fmvm_cpu_windows.cpp`
- `backends/cuda_backend.py`
  - CUDA adapter
  - compiles and runs `fixed_matrix_vector_multiplication/cuda/fmvm_cuda_runner.cu`
- `fixed_matrix_vector_multiplication/cpu/windows/fmvm_cpu_windows.cpp`
  - the actual CPU compute program
  - reads `A.bin` and `x.bin`
  - tries multiple worker counts and tile sizes
  - outputs best latency, effective GFLOPS, checksum, and best config
- `fixed_matrix_vector_multiplication/cuda/fmvm_cuda_runner.cu`
  - the actual CUDA compute program
  - reads `A.bin` and `x.bin` once
  - tries multiple `transpose`, `block_size`, and `tile_size` combinations
  - outputs the fastest configuration plus latency, effective GFLOPS, and checksum

## Backends

- `cpu`
  - implemented today
  - searches worker counts in the requested binary-tree order, such as
    `16 -> 8 -> 32 -> 4 -> 64`
  - sweeps multiple tile sizes for each worker candidate
  - reads the dataset once, then benchmarks all candidate configurations in memory
- `cuda`
  - implemented when `nvcc` and a CUDA-capable GPU are available
  - searches `transpose`, `block_size`, and `tile_size`
  - compiles for the detected GPU compute capability, such as `sm_89`
  - keeps the matrix-vector arithmetic in FP32
  - is checked against the CPU path with FP32-friendly error tolerances

## Scoring

The benchmark still maps latency to a linear score:

```text
score = clamp(
    max_score * (zero_score_seconds - wall_clock_latency_seconds)
    / (zero_score_seconds - ideal_seconds),
    0,
    max_score
)
```

Shorter runtime means higher score.

## Usage

Run the benchmark with the default fixed dataset:

```bash
python "performance metrics/benchmark.py"
```

By default this tries every backend that the workspace currently knows about,
which is CPU first and CUDA when the local machine/toolchain supports it.

Run only the CPU backend:

```bash
python "performance metrics/benchmark.py" --backend cpu
```

Run only the CUDA backend:

```bash
python "performance metrics/benchmark.py" --backend cuda
```

Generate the dataset without running the benchmark:

```bash
python "performance metrics/fixed_matrix_vector_multiplication/input matrix/generate.py"
```

When `tqdm` is installed, dataset generation shows a live progress bar for
`A.bin` and `x.bin`.

Write the benchmark report somewhere other than the default `result.json`:

```bash
python "performance metrics/benchmark.py" --output my_result.json
```

`--rows` and `--cols` still exist for tests and tiny local experiments, but the
normal benchmark path is the fixed 2 GiB matrix above.

## Outputs

- Generated input files:
  - `performance metrics/fixed_matrix_vector_multiplication/input matrix/generated/A.bin`
  - `performance metrics/fixed_matrix_vector_multiplication/input matrix/generated/x.bin`
  - `performance metrics/fixed_matrix_vector_multiplication/input matrix/generated/dataset_meta.json`
- Benchmark summary:
  - `performance metrics/result.json`
- CPU build artifacts:
  - `performance metrics/fixed_matrix_vector_multiplication/cpu/windows/build/`

## Notes

- Generated datasets, backend build outputs, and `result.json` are git-ignored
  because they are machine-local artifacts.
- The checked-in exception is the Windows CPU/CUDA benchmark executables, so a
  fresh Windows checkout can run those backends without rebuilding first.
- The CPU compute program reports:
  - wall-clock latency
  - effective GFLOPS
  - checksum
- The CUDA compute program reports the same metrics and also returns the best
  CUDA launch/layout configuration it found.
- Cross-implementation checks use FP32 error tolerances instead of exact
  checksum matches, because different reduction orders can still be correct
  while producing slightly different FP32 results.
- `result.json` now keeps one best entry per backend plus a `ranking` list, so
  other programs can select the top one or top two backends directly.

## Result Schema

`result.json` is organized around two ideas:

- `backend_results`
  - one entry per backend such as `cpu` or `cuda`
  - stores whether that backend was available
  - stores its rank among successful backends
  - stores the best configuration and best measured result for that backend
  - keeps backend-level notes plus best-trial notes such as CUDA device and `sm`
- `ranking`
  - ordered list of backend names from fastest to slowest among successful
    backends
  - downstream code can read `ranking[0]` for the single best backend or
    `ranking[:2]` for the top two choices
- `dataset`
  - stores dataset file locations as paths relative to `performance metrics/`
    instead of machine-specific absolute paths

Example shape:

```json
{
  "best_backend": "cuda",
  "ranking": ["cuda", "cpu"],
  "backend_results": {
    "cuda": {
      "available": true,
      "rank": 1,
      "best_config": {
        "transpose": false,
        "block_size": 256,
        "tile_size": 8
      },
      "best_result": {
        "wall_clock_latency_seconds": 0.0085,
        "effective_gflops": 125.2,
        "checksum": "fnv1a64:...",
        "score": 1000.0
      },
      "trial_notes": ["device=RTX-class GPU", "sm=89"]
    },
    "cpu": {
      "available": true,
      "rank": 2,
      "best_config": {
        "workers": 64,
        "tile_size": 4096
      },
      "best_result": {
        "wall_clock_latency_seconds": 0.044,
        "effective_gflops": 24.1,
        "checksum": "fnv1a64:...",
        "score": 1000.0
      },
      "trial_notes": []
    }
  }
}
```
