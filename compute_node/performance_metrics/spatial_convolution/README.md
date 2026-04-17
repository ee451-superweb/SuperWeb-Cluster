# Spatial Convolution Performance Metrics

This package owns the Python-side benchmark orchestration for the
`spatial_convolution` method.

The layout is now intentionally split the same way as
`fixed_matrix_vector_multiplication`:

- `compute_node/compute_methods/spatial_convolution/`
  - native runner source files, checked-in runner binaries, and build scripts
- `compute_node/input_matrix/spatial_convolution/`
  - deterministic dataset generation for benchmark and runtime inputs
- `compute_node/performance_metrics/spatial_convolution/`
  - backend probing, autotuning, measurement, ranking, and result serialization

## What This Benchmark Produces

`benchmark.py` measures the best available backend on this machine for the
fixed spatial-convolution workloads, then writes a compact summary into
`result.json`.

That summary is later consumed by compute-node registration so the main node
can rank workers by measured backend performance.

## Main Files

- `benchmark.py`
  - entrypoint for the spatial-convolution benchmark
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

- `../../compute_methods/spatial_convolution/cpu/`
- `../../compute_methods/spatial_convolution/cuda/`
- `../../compute_methods/spatial_convolution/dx12/`
- `../../compute_methods/spatial_convolution/metal/`

## Control Flow

`benchmark.py` does four things:

1. Resolve the requested benchmark shape.
2. Ensure the required datasets exist under `compute_node/input_matrix/spatial_convolution/generated/`.
3. Probe and benchmark every selected backend.
4. Rank successful backends and write `result.json`.

When the default workload is used, the benchmark autotunes on the small dataset
and then measures the winning configuration on the large dataset. The CLI also
supports `small`, `medium`, `large`, and `full` workload modes when you want to
benchmark only one size or force the full two-stage flow explicitly.

## Usage

Run the default benchmark:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py"
```

Benchmark only one backend:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py" --backend cpu
```

Run only the small workload:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py" --workload-mode small
```

Run the full default flow explicitly:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py" --workload-mode full
```

Run a custom convolution shape:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py" --h 512 --w 512 --cin 64 --cout 128 --k 3 --pad 1 --stride 2
```

Generate the datasets without benchmarking:

```bash
python "compute_node/input_matrix/spatial_convolution/generate.py"
```

Force selected backends to rebuild their executables:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py" --rebuild
```

Override the CUDA batch/cooldown experiment knobs:

```bash
python "compute_node/performance_metrics/spatial_convolution/benchmark.py" --backend cuda --output-channel-batch 16 --cooldown-ms 2.5
```

## Notes

- Generated datasets and benchmark reports are local-only and git-ignored.
- The benchmark reuses the checked-in runner that matches the current platform
  when possible, and rebuilds only when the source is newer or `--rebuild` is
  requested.
- Windows CPU/CUDA runners are intended to stay directly runnable on another
  Windows machine without rebuilding first.
- The spatial-convolution runtime executor shares the same native runners, so
  method and benchmark paths stay aligned.
- The checked-in Metal runner embeds its compiled `metallib`, so once built it
  can run on another compatible macOS machine without Xcode or the Metal
  toolchain installed.
- Cross-implementation validation uses FP32 tolerances rather than exact
  checksum equality.
- This workspace finished its Sprint 1 role: benchmark, rank, and summarize
  local compute hardware for runtime registration. Sprint 2 will consume that
  summary for actual scheduling decisions.
