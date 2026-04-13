# Input Matrix

This directory owns the shared matrix/vector dataset used by local compute
benchmarks.

It is intentionally separate from `compute_node/performance_metrics/`:

- `input_matrix/` defines and generates the dataset files
- `performance_metrics/` consumes those files to benchmark CPU/CUDA/Metal backends

That separation keeps the input-file format reusable for future workloads and
avoids coupling the dataset generator to one benchmark implementation.

## Files

- `__init__.py`
  - package entrypoint that re-exports the public dataset API
- `spec.py`
  - shared dataset shape, layout dataclasses, and fixed constants
- `storage.py`
  - dataset validation plus binary inspection helpers
- `splitmix.py`
  - deterministic counter-based float32 chunk generation helpers
- `generator.py`
  - streaming file writer and metadata emitter
- `generate.py`
  - CLI entrypoint for creating or refreshing the dataset on disk
- `generated/`
  - local dataset cache
  - contains `A.bin`, `x.bin`, `dataset_meta.json`, and tiny override datasets

## Default Dataset

The current production-sized dataset is:

- `A`: `16384 x 32768` float32, exactly `2 GiB`
- `x`: `32768` float32

The files themselves are generic binary assets:

- `A.bin`
  - dense row-major float32 matrix
- `x.bin`
  - dense float32 vector
- `dataset_meta.json`
  - dataset shape, dtype, layout, seed, algorithm, and SHA-256 metadata

## Usage

Generate the default dataset:

```bash
python "compute_node/input_matrix/generate.py"
```

Force a rebuild of the default dataset:

```bash
python "compute_node/input_matrix/generate.py" --force
```

Tune generation speed manually:

```bash
python "compute_node/input_matrix/generate.py" --workers 4 --chunk-mib 32
```

Generate a tiny override dataset for testing:

```bash
python "compute_node/input_matrix/generate.py" --rows 8 --cols 16 --output-dir tmp_dataset
```

## Notes

- The generator uses fixed seeds and a fixed algorithm so output is deterministic.
- When NumPy is installed, generation uses a vectorized fast path.
- The implementation is intentionally split by responsibility so the dataset
  package does not collapse back into one giant file.
- Generated files under `generated/` are local-only and git-ignored.
- `compute_node/performance_metrics/benchmark.py` will call this generator
  automatically when the required dataset is missing or mismatched.
