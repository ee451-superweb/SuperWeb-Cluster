# Performance Metrics

This workspace is the local compute-characterization stage for
`superweb-cluster` Sprint 1.

Its job is to answer:

- what hardware backends are available on this machine
- what their best measured GFLOPS are for the fixed workload
- how those backends should be ranked for later scheduling decisions

`bootstrap.py` is still the top-level project entrypoint. In normal use, this
benchmark workspace is invoked automatically when
`compute_node/performance_metrics/result.json` is missing.

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

- `Conv2D`

The default fixed problem is:

- `A`: `16384 x 32768` float32, exactly `2 GiB`
- `x`: `32768` float32
- operation: `y = A x`

The production benchmark keeps this shape fixed so different machines compare
the same work.

The fixed FMVM workload now runs in two stages:

- autotune: every candidate config is timed with `3` repeats and ranked by
  average latency
- measurement: the winning config is rerun with `20` repeats and that average
  latency becomes the reported benchmark result

## Pad And Stride

For the Conv2D benchmark:

- `pad`
  - how many zero-value pixels are added around the input border
- `stride`
  - how far the filter moves each time along height and width

The output shape is:

- `out_h = floor((h + 2 * pad - k) / stride) + 1`
- `out_w = floor((w + 2 * pad - k) / stride) + 1`

The current default workloads use:

- `pad = 1`
- `stride = 1`

So a `3x3` kernel keeps the spatial size unchanged. If you pass `--stride 2`,
the benchmark becomes a downsampling convolution and the output height/width
shrink accordingly.

## Control Flow

`benchmark.py` acts as the orchestrator:

1. resolve the benchmark shape
2. check whether `A.bin` and `x.bin` already exist
3. call `generate.py` only if the dataset is missing or mismatched
4. run all enabled backend adapters
5. autotune each backend config with the short repeat count
6. rerun the winning config with the long repeat count
7. rank backends and write `result.json`

## Main Files

- `benchmark.py`
  - top-level benchmark runner
  - handles dataset checks, backend execution, ranking, and JSON output
- `dataset.py`
  - dataset layout and validation
  - deterministic streamed generation of `A.bin`, `x.bin`, and metadata
- `backends/`
  - one adapter per hardware family
  - keeps benchmark orchestration hardware-agnostic
- `backends/cpu_backend.py`
  - Windows/macOS CPU backend adapter
- `backends/cuda_backend.py`
  - CUDA backend adapter
- `backends/metal_backend.py`
  - Metal backend adapter
- `conv2d_runners/cpu/windows/fmvm_cpu_windows.cpp`
  - Windows CPU compute runner
- `conv2d_runners/cpu/macos/fmvm_cpu_macos.cpp`
  - macOS CPU compute runner
- `conv2d_runners/cuda/fmvm_cuda_runner.cu`
  - CUDA compute runner
- `conv2d_runners/metal/fmvm_metal_runner.mm`
  - Metal host runner
- `conv2d_runners/metal/fmvm_metal_kernels.metal`
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
  - the checked-in Windows CUDA runner is built as a fat binary for `sm_75`, `sm_80`, `sm_86`, `sm_89`, and `sm_90`
  - runtime needs only a compatible NVIDIA driver; `nvcc` and Visual Studio are only needed for rebuilding
  - keeps the matrix-vector arithmetic in FP32
- `metal`
  - enabled on macOS when a prebuilt self-contained runner is present, or when
    `xcrun` can resolve `metal`, `metallib`, and `clang++`
  - sweeps `block_size` and `tile_size`
  - compiles a `.metal` kernel plus an Objective-C++ host runner
  - embeds the compiled `metallib` into the runner executable, so runtime does
    not need a sidecar kernel file or a local Metal toolchain
  - keeps the matrix-vector arithmetic in FP32

## Output Schema

`result.json` stores:

- `schema_version`
  - current benchmark result schema version
- `host`
  - local platform metadata such as `system`, `release`, and `machine`
- `workload`
  - the fixed repeat policy used by every backend
- `hardware_inventory`
  - one probe result per backend
  - records whether each backend looked usable before running workloads
- `detected_backends`
  - backends whose probe step succeeded on this machine
- `usable_backends`
  - backends that completed the benchmark successfully
- `backend_results`
  - one best entry per backend such as `cpu` or `cuda`
- `ranking`
  - fastest to slowest among successful backends
- `best_backend`
  - the first item in `ranking`

Each `hardware_inventory` entry stores:

- `probe_available`
  - whether that backend looked runnable during hardware detection
- `probe_message`
  - short human-readable probe detail

Each `backend_results` entry stores:

- `available`
  - whether that backend finished the benchmark successfully
- `rank`
  - placement among successful backends
- `best_config`
  - the autotuned config that won for that backend
- `autotune_result`
  - average latency and GFLOPS from the short autotune phase used for selection
- `best_result`
  - average latency, GFLOPS, checksum, and score from the `20`-repeat
    measurement phase
- `notes`
  - backend-level notes such as compile or probe details
- `trial_notes`
  - device-specific notes for the best trial

That file is also the source for the compact runtime registration summary sent
by compute nodes to the main node.

## Usage

Run the benchmark with the default fixed dataset:

```bash
python "compute_node/performance_metrics/benchmark.py"
```

Run a custom downsampling convolution:

```bash
python "compute_node/performance_metrics/benchmark.py" --h 512 --w 512 --cin 64 --cout 128 --k 3 --pad 1 --stride 2
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

Generate the dataset without running the benchmark:

```bash
python "compute_node/dataset/generate.py"
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

`--rows` and `--cols` remain available for tiny tests. When you use those
overrides without changing `--dataset-dir`, the generated files are written to
`compute_node/dataset/generated/overrides/<rows>x<cols>/` so the main
production dataset does not get overwritten.

## Notes

- Generated datasets and benchmark reports are local-only and git-ignored.
- The benchmark picks the binary that matches the current OS. If that binary
  is missing or older than the current OS source, it falls back to compiling
  the current OS source.
- `--rebuild` disables that reuse path for the selected backends and requires
  a usable local toolchain.
- The checked-in Windows CPU/CUDA executables are intended to be directly runnable
  on another Windows machine without a local compiler or CUDA toolkit install.
- The Windows CUDA runner still requires the NVIDIA driver at runtime because
  the driver provides `nvcuda.dll`.
- The self-contained Metal runner embeds its compiled `metallib`. Once built,
  it can run on another compatible macOS machine without Xcode or the Metal
  toolchain installed.
- Cross-implementation validation uses FP32 tolerances rather than exact
  checksum equality.
- This workspace finished its Sprint 1 role: benchmark, rank, and summarize
  local compute hardware for runtime registration. Sprint 2 will consume that
  summary for actual scheduling decisions.
