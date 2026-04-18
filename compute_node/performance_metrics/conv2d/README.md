# Spatial Convolution Performance Metrics

This package owns the Python-side benchmark orchestration for the
`conv2d` method.

The layout is now intentionally split the same way as
`gemv`:

- `compute_node/compute_methods/conv2d/`
  - native runner source files, checked-in runner binaries, and build scripts
- `compute_node/input_matrix/conv2d/`
  - deterministic dataset generation for benchmark and runtime inputs
- `compute_node/performance_metrics/conv2d/`
  - backend probing, autotuning, measurement, ranking, and result serialization

## What This Benchmark Produces

`benchmark.py` measures the best available backend on this machine for the
fixed conv2d workloads, then writes a compact summary into
`result.json`.

That summary is later consumed by compute-node registration so the main node
can rank workers by measured backend performance.

## Main Files

- `benchmark.py`
  - entrypoint for the conv2d benchmark
- `dataset.py`
  - dataset layout helpers and validation
- `models.py`
  - benchmark result and workload model definitions
- `workloads.py`
  - default benchmark shapes for autotune and runtime measurement
- `scoring.py`
  - converts measured latency into normalized benchmark scores
- `backends/`
  - hardware-specific benchmark adapters

The native runner sources used by those backends live in the method directory:

- `../../compute_methods/conv2d/cpu/`
- `../../compute_methods/conv2d/cuda/`
- `../../compute_methods/conv2d/dx12/`
- `../../compute_methods/conv2d/metal/`

## Control Flow

`benchmark.py` does four things:

1. Resolve the requested benchmark shape.
2. Ensure the required datasets exist under `compute_node/input_matrix/conv2d/generated/`.
3. Probe and benchmark every selected backend.
4. Rank successful backends and write `result.json`.

When the default workload is used, the benchmark runs the mid dataset only. The
large dataset is now opt-in because it is much heavier on storage and time.
Pass `--workload-mode large` to benchmark only the large dataset or
`--workload-mode full` to force the explicit small-autotune-plus-large-measurement
flow. The CLI also supports `small`, `mid`, `large`, and `full` workload modes.

## Usage

Run the default benchmark:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py"
```

Benchmark only one backend:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py" --backend cpu
```

Run only the small workload:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py" --workload-mode small
```

Run the full default flow explicitly:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py" --workload-mode full
```

Run a custom convolution shape:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py" --h 512 --w 512 --cin 64 --cout 128 --k 3 --pad 1 --stride 2
```

Generate the datasets without benchmarking:

```bash
python "compute_node/input_matrix/conv2d/generate.py"
```

Force selected backends to rebuild their executables:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py" --rebuild
```

Override the CUDA batch/cooldown experiment knobs:

```bash
python "compute_node/performance_metrics/conv2d/benchmark.py" --backend cuda --output-channel-batch 16 --cooldown-ms 2.5
```

## Notes

- Generated datasets and benchmark reports are local-only and git-ignored.
- The benchmark reuses the checked-in runner that matches the current platform
  when possible, and rebuilds only when the source is newer or `--rebuild` is
  requested.
- Windows CPU/CUDA runners are intended to stay directly runnable on another
  Windows machine without rebuilding first.
- The conv2d runtime executor shares the same native runners, so
  method and benchmark paths stay aligned.
- The checked-in Metal runner embeds its compiled `metallib`, so once built it
  can run on another compatible macOS machine without Xcode or the Metal
  toolchain installed.
- Cross-implementation validation uses FP32 tolerances rather than exact
  checksum equality.
- This workspace finished its Sprint 1 role: benchmark, rank, and summarize
  local compute hardware for runtime registration. Sprint 2 will consume that
  summary for actual scheduling decisions.
