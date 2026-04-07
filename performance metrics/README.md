# Performance Matrix

This folder benchmarks one method for now:

- `fixed_matrix_vector_multiplication`

The workspace directory is `performance metrics`.

## Goals

- deterministic benchmark input generation
- linear scoring where shorter runtime means higher score
- default runtime under 5 minutes on mainstream machines
- automatic search for a better configuration instead of one hard-coded setup
- CPU first, CUDA optional when `nvcc` is available

## Scoring

For each workload:

```text
score = clamp(
    max_score * (zero_score_seconds - elapsed_seconds)
    / (zero_score_seconds - ideal_seconds),
    0,
    max_score
)
```

- `elapsed_seconds` is the measured time for one matrix-vector multiply
- `ideal_seconds` gives the full score
- `zero_score_seconds` gives score 0
- the score changes linearly between those two points

Default `max_score` is `1000`.

## Presets

- `smoke`: tiny dataset for tests and quick validation
- `quick`: faster local sanity benchmark
- `standard`: default benchmark preset
- `extended`: larger workload when you want more separation

`standard` is tuned to stay comfortably below 5 minutes even when the CPU
backend explores multiple configurations.

## Backends

- `cpu`: implemented with a Windows C++ runner
  - benchmark orchestration starts from the detected hardware worker count
  - worker search follows a binary-tree style order such as `16 -> 8/32 -> 4/64`
  - every worker candidate gets a full tile-size sweep
- `cuda`: optional
  - compiles `fixed_matrix_vector_multiplication/cuda/fmvm_cuda_runner.cu`
    with `nvcc` when CUDA is available
  - autotunes `transpose`, `block_size`, and `tile_size`
  - skips cleanly if `nvcc` is not present

## Usage

List presets:

```bash
python "performance metrics/benchmark.py" --list-presets
```

Run the default CPU + CUDA benchmark:

```bash
python "performance metrics/benchmark.py"
```

Run only CPU with a smaller smoke preset:

```bash
python "performance metrics/benchmark.py" --backend cpu --preset smoke
```

Generate only the input dataset:

```bash
python "performance metrics/fixed_matrix_vector_multiplication/input matrix/generate.py" --preset standard
```

Benchmark results are written to:

- `performance metrics/result.json`

## Notes

- The CPU backend now compiles and runs `fixed_matrix_vector_multiplication/cpu/windows/fmvm_cpu_windows.cpp`.
- The CUDA backend is designed so the current environment can skip it cleanly
  while CUDA-equipped machines can compile and run it later.
- `result.json` is intentionally summary-only and stores just the global best
  configuration and its measured result.
