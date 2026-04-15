# Performance Metrics

This workspace is the local compute-characterization stage for
`superweb-cluster` Sprint 3.

Its job is to answer:

- what hardware backends are available on this machine
- what their best measured GFLOPS are for each supported compute method
- how those backends should be ranked for later scheduling decisions

This workspace now uses a method-scoped layout:

- `performance_metrics/fixed_matrix_vector_multiplication/`
  - FMVM-specific benchmark result/config workspace
- `performance_metrics/spatial_convolution/`
  - Conv2D-specific benchmark result/config workspace
- top-level `performance_metrics/benchmark.py`
  - multi-method orchestrator that can run one method or both

The benchmark consumes method-scoped datasets from `../input_matrix/README.md`.
It also does not own the low-level method implementations themselves. Shared
CPU/CUDA/DX12/Metal runners live under `../compute_methods/`.

`bootstrap.py` is still the top-level project entrypoint. In normal use, this
benchmark workspace is invoked automatically when
`compute_node/performance_metrics/result.json` is missing.

## How It Fits Into The Project

The current flow is:

1. `bootstrap.py` prepares `.venv` and dependencies.
2. `bootstrap.py` ensures the shared input dataset and this benchmark result exist.
3. A compute node later loads that result summary during runtime registration.
4. It sends only the compact summary upstream:
   - hardware backend count
   - backend ranking
   - best effective GFLOPS per backend
5. The main node stores those reported backends as worker-hardware capability objects.
6. Sprint 2 scheduling will use that capability inventory to distribute load.

## Supported Methods

The benchmark currently measures two methods:

- `fixed_matrix_vector_multiplication`
- `spatial_convolution`

The default FMVM problem is:

- `A`: `16384 x 32768` float32, exactly `2 GiB`
- `x`: `32768` float32
- operation: `y = A x`

The default Conv2D problem uses:

- autotune workload: `256 x 256`, `32 -> 64`, `k=3`, `pad=1`, `stride=1`
- runtime workload: `2048 x 2048`, `128 -> 256`, `k=3`, `pad=1`, `stride=1`

The production benchmark keeps these shapes fixed so different machines compare
the same work.

The fixed FMVM workload now runs in two stages:

- autotune: every candidate config is timed with `3` repeats and ranked by
  average latency
- measurement: the winning config is rerun with `20` repeats and that average
  latency becomes the reported benchmark result

The numeric contract is explicit:

- input dtype: `fp32`
- output dtype: `fp32`
- default accumulation precision: `fp32`
- optional stricter mode: `fp64_accumulate`

Even when two backends use the same `fp32` accumulation mode, tiny result
differences can still happen because CPU and GPU do not add partial sums in the
same order.

## Control Flow

Top-level `benchmark.py` acts as the orchestrator:

1. select one method or all methods
2. resolve each method's dataset workspace
3. call the matching method-local generator only if the dataset is missing or mismatched
4. run all enabled backend adapters for that method
5. autotune each backend config
6. rerun the winning config with the final measurement policy
7. normalize the output into one shared `result.json` schema
8. write both the aggregate report and the per-method `result.json` files

On Windows, the default GPU routing is intentionally simple:

- if Device Manager reports an NVIDIA display adapter, the automatic benchmark includes `cuda`
- if Device Manager reports a non-NVIDIA display adapter such as AMD Radeon 780M or Intel graphics, the automatic benchmark includes `dx12`

## Main Files

- `benchmark.py`
  - top-level benchmark runner
  - handles dataset checks, backend execution, ranking, and JSON output
- `dataset_runner.py`
  - resolves dataset directories and shells out to `../input_matrix/generate.py` when needed
- `reporting.py`
  - assembles the final `result.json` structure from backend outcomes
- `../input_matrix/README.md`
  - documents the shared dataset format and generator CLI
- `../compute_methods/README.md`
  - documents the shared method-implementation layer used by runtime and benchmark code
- `../input_matrix/__init__.py`
  - package entrypoint for the shared dataset API
- `../input_matrix/spec.py`
  - dataset shape, layout dataclasses, and fixed constants
- `../input_matrix/storage.py`
  - dataset validation plus inspection helpers used by tests and benchmark flow
- `../input_matrix/generator.py`
  - deterministic streamed generation of `A.bin`, `x.bin`, and generic metadata
- `../input_matrix/generate.py`
  - CLI entrypoint that creates or refreshes the shared dataset on disk
- `backends/`
  - one adapter per hardware family
  - keeps benchmark orchestration hardware-agnostic
- `backends/cpu_backend.py`
  - Windows/macOS CPU backend adapter
- `backends/cuda_backend.py`
  - CUDA backend adapter
- `backends/dx12_backend.py`
  - Windows DX12 backend adapter for non-CUDA GPU paths
- `backends/metal_backend.py`
  - Metal backend adapter
- `../compute_methods/fixed_matrix_vector_multiplication/cpu/windows/fmvm_cpu_windows.cpp`
  - Windows CPU compute runner
- `../compute_methods/fixed_matrix_vector_multiplication/cpu/macos/fmvm_cpu_macos.cpp`
  - macOS CPU compute runner
- `../compute_methods/fixed_matrix_vector_multiplication/cuda/fmvm_cuda_runner.cu`
  - CUDA compute runner
- `../compute_methods/fixed_matrix_vector_multiplication/dx12/fmvm_dx12_runner.cpp`
  - Windows DX12 compute runner
  - benchmark mode uploads the fixed matrix once per benchmark process
  - compute-node runtime can also keep it resident across many assigned tasks
- `../compute_methods/fixed_matrix_vector_multiplication/metal/fmvm_metal_runner.mm`
  - Metal host runner
- `../compute_methods/fixed_matrix_vector_multiplication/metal/fmvm_metal_kernels.metal`
  - Metal compute kernel

## Backends

- `cpu`
  - sweeps worker counts in binary-tree order such as `16 -> 8 -> 32 -> 4 -> 64`
  - sweeps multiple tile sizes
  - reads the dataset once and benchmarks candidate configs in memory
  - on Windows, the checked-in runner is built with the static MSVC runtime (`/MT`)
  - the checked-in Windows CPU runner can run on a clean machine without Visual Studio
  - uses the current OS CPU binary when present, otherwise compiles the current OS source
- `cuda`
  - on Windows, prefers the checked-in self-contained runner before trying to rebuild
  - sweeps `transpose`, `block_size`, and `tile_size`
  - the checked-in Windows CUDA runner statically links `cudart` and the MSVC runtime
  - the checked-in Windows CUDA runner is built as a fat binary for `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, and `sm_120`
  - runtime needs only a compatible NVIDIA driver; `nvcc` and Visual Studio are only needed for rebuilding
  - keeps the matrix-vector arithmetic in FP32
- `dx12`
  - Windows-only backend for DirectX 12 compute
  - intended as the first native GPU path for integrated GPUs such as Radeon 780M
  - reserved for non-NVIDIA adapters so it complements CUDA instead of duplicating it on GeForce hardware
  - prefers the minimum-power DXGI adapter before falling back to general hardware enumeration
  - sweeps `thread_group_size` and `rows_per_thread`
  - uses a slightly longer autotune phase than the other backends so iGPU burst noise is less likely to pick the wrong config
  - uses a thread-group reduction kernel so adjacent GPU lanes walk adjacent matrix columns instead of striding across rows
  - links only against Windows Direct3D 12 system libraries and keeps the arithmetic in FP32
  - now participates in the default automatic benchmark order
- `metal`
  - enabled on macOS when a prebuilt self-contained runner is present, or when
    `xcrun` can resolve `metal`, `metallib`, and `clang++`
  - sweeps `block_size` and `tile_size`
  - compiles a `.metal` kernel plus an Objective-C++ host runner
  - embeds the compiled `metallib` into the runner executable, so runtime does
    not need a sidecar kernel file or a local Metal toolchain
  - keeps the matrix-vector arithmetic in FP32

## Output Schema

`result.json` now uses one shared schema for every method:

- top level
  - `device_overview`
    - host CPU name, GPU list, and memory configuration
  - `methods`
    - one entry per compute method

Each method entry stores:

- `dataset`
- `workload.autotune_plan`
- `workload.measurement_plan`
- `ranking`
- `best_backend`
- `backends`

Each backend entry stores:

- `device_name`
- `rank`
- `autotune_plan`
- `autotune_result`
- `best_config`
- `best_result`
- concise `notes`

That aggregate file is also the source for the compact runtime registration
summary sent by compute nodes to the main node.

## Usage

Run the benchmark with the default fixed dataset:

```bash
python "compute_node/performance_metrics/benchmark.py"
```

Run only the CPU backend:

```bash
python "compute_node/performance_metrics/benchmark.py" --backend cpu
```

Run only the CUDA backend:

```bash
python "compute_node/performance_metrics/benchmark.py" --backend cuda
```

Run only the Metal backend:

```bash
python "compute_node/performance_metrics/benchmark.py" --backend metal
```

Run only the DX12 backend:

```bash
python "compute_node/performance_metrics/benchmark.py" --backend dx12
```

Run the DX12 benchmark and force a local rebuild of the native runner:

```bash
python "compute_node/performance_metrics/benchmark.py" --backend dx12 --rebuild
```

Run the raw DX12 executable directly for one fixed config:

```bash
compute_node\compute_methods\fixed_matrix_vector_multiplication\dx12\build\fmvm_dx12_runner.exe ^
  --matrix ..\input_matrix\generated\A.bin ^
  --vector ..\input_matrix\generated\x.bin ^
  --output %TEMP%\fmvm_dx12_output.bin ^
  --rows 16384 ^
  --cols 32768 ^
  --row-start 0 ^
  --row-end 16384 ^
  --fixed-thread-group-size 512 ^
  --fixed-rows-per-thread 1 ^
  --iteration-count 1
```

Generate the FMVM dataset without running the benchmark:

```bash
python "compute_node/input_matrix/fixed_matrix_vector_multiplication/generate.py"
```

Manually tune the generator for a faster one-time dataset build:

```bash
python "compute_node/input_matrix/generate.py" --workers 4 --chunk-mib 32
```

Write the report to a different path:

```bash
python "compute_node/performance_metrics/benchmark.py" --output my_result.json
```

Force all selected backends to rebuild their executables instead of reusing
checked-in or cached binaries:

```bash
python "compute_node/performance_metrics/benchmark.py" --rebuild
```

Request stricter accumulation during benchmarking:

```bash
python "compute_node/performance_metrics/benchmark.py" --accumulation-precision fp64_accumulate
```

`--rows` and `--cols` remain available for tiny tests. When you use those
overrides without changing `--dataset-dir`, the generated files are written to
`compute_node/input_matrix/generated/overrides/<rows>x<cols>/` so the main
production dataset does not get overwritten.

## Notes

- Generated datasets in `compute_node/input_matrix/generated/` and benchmark
  reports such as `result.json` are local-only and git-ignored.
- `generate.py` uses 32 MiB streaming chunks by default and, when NumPy is
  installed, generates each chunk with vectorized integer math instead of a
  Python per-value loop.
- The matrix generator can also fan out chunk writes across multiple workers
  for a faster one-time dataset build on larger machines.
- The benchmark entrypoint itself is now split into smaller helper modules so
  dataset preparation and report assembly do not live in one giant file.
- The FMVM CPU/CUDA/DX12/Metal sources now live under `compute_node/compute_methods/`
  so `performance_metrics/` stays focused on benchmarking rather than owning
  method source trees.
- The shared dataset metadata is intentionally generic. It records dataset
  shape, dtype, seeds, hashes, and file layout without embedding
  `performance_metrics/`-specific tuning state.
- The benchmark picks the binary that matches the current OS. If that binary
  is missing or older than the current OS source, it falls back to compiling
  the current OS source.
- `result.json` now records the numeric contract explicitly, including
  `input_dtype`, `output_dtype`, and `accumulation_precision`.
- `--rebuild` disables that reuse path for the selected backends and requires
  a usable local toolchain.
- The checked-in Windows CPU/CUDA executables are intended to be directly runnable
  on another Windows machine without a local compiler or CUDA toolkit install.
- The Windows CUDA runner still requires the NVIDIA driver at runtime because
  the driver provides `nvcuda.dll`.
- The Windows DX12 runner depends on the system Direct3D 12 runtime and graphics
  driver, and it currently rebuilds locally instead of shipping a checked-in exe.
- DX12 JSON output now includes `setup_wall_clock_latency_seconds`,
  `static_upload_wall_clock_latency_seconds`,
  `vector_upload_wall_clock_latency_seconds`, and `dispatches_per_repeat` so it
  is easier to separate one-time setup cost, per-request vector refresh cost,
  and steady-state compute throughput.
- The self-contained Metal runner embeds its compiled `metallib`. Once built,
  it can run on another compatible macOS machine without Xcode or the Metal
  toolchain installed.
- Cross-implementation validation uses FP32 tolerances rather than exact
  checksum equality.
- This workspace finished its Sprint 1 role: benchmark, rank, and summarize
  local compute hardware for runtime registration. Sprint 2 will consume that
  summary for actual scheduling decisions.
