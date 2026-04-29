# Compression Evaluation Report

## 1. Goal

The goal of this round of experiments is to find the best compression option, under realistic deployment constraints, for the largest matrix file in `Client/download`.

The constraints are:

- The data is a raw `FP32` numeric file used to store a matrix.
- Compression must finish within `30` seconds, otherwise it is judged unusable.
- The sender side may have a `GPU`.
- The receiver side is only guaranteed to have a `CPU`.
- Two distinct requirements must be considered separately:
  - `Strict lossless`: the post-decompression `checksum` must match exactly.
  - `Very low error`: very small numerical error is acceptable in exchange for higher compression ratio.

## 2. Data and Environment

- Working directory: `C:\Users\dongm\ee451\Client\download`
- Test file: `conv-2-large.bin`
- File size: `4294967296` bytes, roughly `4.00 GiB`
- Inferred shape: `32768 x 32768` `float32` matrix
- Source file SHA256: `d29877d9d75a4a811d3916d3326ba0773ae8ada279b1d52c552369e522ce0213`
- Compression-side GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Driver version: `591.86`
- CUDA: `13.2`

### 2.1 Software and Dependency Versions

All Python dependencies for this experiment are installed under `Client/download/.vendor` to avoid polluting other environments. Below is the complete list of packages and versions that were either directly used by the experiment scripts / compression flows, or installed solely so the experiment could run on this machine.

The core software directly involved in the experiment scripts or compression pipelines:

| Category | Name | Version | Notes |
| --- | --- | --- | --- |
| Python runtime | `python` | `3.14.3` | Execution environment for all scripts in this round |
| Array / numerics | `numpy` | `2.4.4` | Matrix loading, error computation, streaming chunking |
| Lossless matrix compression | `blosc2` | `4.1.2` | `zstd/lz4 + shuffle/bitshuffle` testing |
| Scientific float compression | `zfpy` | `1.0.1` | `ZFP` Python binding |
| Scientific float compression | `fpzip` | `1.2.5` | `fpzip` Python package |
| Scientific float compression | `pysz` | `1.0.3` | `SZ3` Python binding, locally built from source |
| General compression backend | `zstandard` | `0.25.0` | `zstd` backend library |
| General compression backend | `lz4` | `4.4.5` | `lz4` backend library |
| Codec collection | `numcodecs` | `0.16.5` | `blosc2`-related dependency |

To successfully build or run these experiments under the current Windows + Python `3.14` environment, the following auxiliary packages were also installed:

| Category | Name | Version | Notes |
| --- | --- | --- | --- |
| Local build | `cmake` | `4.3.1` | Used when building `pysz` |
| Local build | `Cython` | `3.2.4` | Used when building `pysz` |
| Local build | `ninja` | `1.13.0` | Build dependency |
| Numerics dependency | `numexpr` | `2.14.1` | `blosc2` dependency |
| Encoding dependency | `msgpack` | `1.1.2` | `blosc2` dependency |
| Indexing dependency | `ndindex` | `1.10.1` | `blosc2` dependency |
| Network dependency | `requests` | `2.33.1` | Pulled in by install / dependency resolution |
| Network dependency | `urllib3` | `2.6.3` | `requests` dependency |
| Network dependency | `certifi` | `2026.2.25` | `requests` dependency |
| Network dependency | `charset_normalizer` | `3.4.7` | `requests` dependency |
| Network dependency | `idna` | `3.11` | `requests` dependency |
| Type compatibility | `typing_extensions` | `4.15.0` | Compatibility dependency |

Build and source notes:

- `zfpy`: installed directly from PyPI.
- `fpzip`: compiled and installed locally via `pip`.
- `pysz`: no prebuilt wheel available; built locally from the source archive `Client/download/_pkgsrc/pysz-1.0.3.tar.gz` using `cmake` and `Cython`, then placed into `.vendor`.
- The GPU is used only to confirm environment availability. The deployability conclusion of the final recommended option in this round assumes "the receiver side has only a CPU".

## 3. Scope of Experiments

This conclusion is not based on a single round of testing; it consolidates the results of several earlier rounds of experiments in this directory.

The categories that have been tested include:

- General lossless compression:
  - `gzip`
  - `bz2`
  - `lzma`
- Lossless compression aimed at binary numeric arrays:
  - `blosc2 + zstd/lz4 + shuffle/bitshuffle`
- Scientific float compression:
  - `zfp / zfpy`
  - `fpzip`
  - `SZ3`
- Structural approximation compression:
  - `SVD`
  - `H-matrix / HODLR`
- Final "deployability benchmark" round:
  - Only candidates whose `decompression on the receiver-side CPU` is feasible are kept.
  - `Compression time`, `CPU decompression time`, and `exact equality` or `error` are measured simultaneously.

### 3.1 Test Methodology and Decision Rules

To make the results reproducible, the methodology actually used in each round is written out clearly below.

Common rules:

- The experiment working directory is fixed at `Client/download`.
- Within a given round of candidate files, the "largest `.bin` file" is selected; if multiple files are tied for largest, the first one in lexicographic order by filename is chosen.
- For an `FP32` file, if `file byte count / 4` is a perfect square, it is inferred to be a 2D square matrix; in this round the primary sample is identified as `32768 x 32768`.
- The user-defined usability rule is: `if compression cannot finish within 30 seconds, it is unusable`.
- For lossless options, full-file `SHA256` is the preferred check for "compress-then-decompress yields a bit-identical result".
- For lossy options, full-matrix `max absolute error` and `RMSE` are used as error metrics.

Methodology of each round:

1. General compression baseline
- Algorithms: `gzip`, `bz2`, `lzma`
- Parameters: `gzip compresslevel=6`, `bz2 compresslevel=6`, `lzma preset=6`
- Implementation: streaming read/write, block size `8 MiB`
- Purpose: establish a baseline for "traditional general-purpose compressors" on this kind of `FP32` matrix file.

2. Lossless compression for matrix / numeric arrays
- Algorithms: `blosc2 + lz4/zstd + shuffle/bitshuffle`
- Implementation: `64 MiB` blocks, `typesize=4`, up to `16` threads
- Tested combinations:
  - `blosc2_lz4_shuffle_c5`
  - `blosc2_lz4_bitshuffle_c5`
  - `blosc2_zstd_shuffle_c1`
  - `blosc2_zstd_shuffle_c3`
  - `blosc2_zstd_shuffle_c5`
  - `blosc2_zstd_bitshuffle_c3`
- Purpose: verify whether applying `shuffle/bitshuffle` to an `FP32` matrix substantially improves lossless compression performance.

3. Scientific float compression
- Algorithms: `ZFP(zfpy)`, `fpzip`, `SZ3(pysz)`
- Implementation: chunk-wise compression in groups of `1024` rows to avoid copying the entire matrix at once.
- Tested parameters:
  - `zfpy_lossless`
  - `zfpy_tolerance_1e-3`
  - `zfpy_rate_16`
  - `fpzip_lossless`
  - `fpzip_precision_24`
  - `sz3_abs_1e-3_interp_lorenzo`
  - `sz3_abs_1e-3_nopred`
- Timeout policy: if compression does not finish within `30s`, the compression ratio over the data already processed is retained and a full-file projection is reported.
- Purpose: compare specialized scientific-float-array compressors on speed, ratio, and error.

4. Structural approximation compression
- Methods: `randomized SVD` and `HODLR / H-matrix-style` approximation.
- `SVD` ranks tested: `8`, `32`, `128`, `256`.
- `HODLR` configurations tested: `rank=32`, `leaf_size=16384/8192/4096`.
- Evaluation metrics: `relative Frobenius error`, captured energy fraction.
- Purpose: judge whether the matrix has sufficiently strong low-rank or hierarchical low-rank structure.

5. Final deployability benchmark
- Assumption: the sender side is flexible; the receiver side is only guaranteed to have a `CPU`.
- Key metrics: `compression time`, `CPU decompression time`, `compression ratio`, `whether SHA256 matches`, `error`.
- Final candidates:
  - Lossless: `blosc2_zstd_shuffle_c1`, `blosc2_lz4_bitshuffle_c5`, `zfpy_lossless`, `fpzip_lossless`
  - Low error: `zfpy_tolerance_1e-4`, `zfpy_tolerance_5e-4`, `zfpy_tolerance_1e-3`, `zfpy_precision_23`, `zfpy_rate_16`, `fpzip_precision_28`
- Purpose: produce candidates that are actually fit for deployment, rather than algorithms that merely win on a single metric.

6. Network-transfer speedup evaluation
- End-to-end model used:
  - `total time = compress + send-side checksum over the transmitted byte stream + network transfer + receive-side checksum over the transmitted byte stream + CPU decompress`
- `SHA256` throughput is measured on this machine by sequentially reading a `1 GiB` binary file of the same kind, and comes out at roughly `1354 MiB/s`.
- The analysis focuses on these links:
  - `300 Mbps WiFi`
  - `1 Gbps Ethernet`
- Purpose: judge whether compression actually helps "transfer faster end-to-end", rather than only counting how many bytes are saved.

## 4. Key Findings

### 4.1 Best Lossless Option

If the requirement is "post-decompression checksum must match", the best option currently is:

`blosc2_zstd_shuffle_c1`

Measured results:

- Compression time: `6.64s`
- CPU decompression time: `4.84s`
- Space savings: `15.52%`
- Post-decompression SHA256: bit-identical to the original file

It is the most practical lossless option available right now, and the reasons are clear:

- It comfortably satisfies the `30s` constraint.
- Its compression ratio is better than that of general-purpose lossless compressors.
- Its decompression speed is also fast.
- A full-file `SHA256` check has been performed, not just spot-sampling.

Comparison of lossless options:

| Method | Compress | CPU decompress | Space savings | Bit-identical? |
| --- | ---: | ---: | ---: | --- |
| `blosc2_zstd_shuffle_c1` | `6.64s` | `4.84s` | `15.52%` | yes |
| `blosc2_lz4_bitshuffle_c5` | `6.57s` | `4.70s` | `9.37%` | yes |
| `zfpy_lossless` | `28.02s` | `29.96s` | `0.61%` | yes |
| `fpzip_lossless` | `30.13s` | n/a | `~5.42%` | timeout |
| `gzip` | `135.50s` | not included in the final CPU-decompress round | `7.35%` | yes |
| `bz2` | `376.71s` | not included in the final CPU-decompress round | `5.01%` | yes |

### 4.2 Best Very-Low-Error Option

If "very low error" is acceptable, the safest current option is:

`zfpy_tolerance_1e-4`

Measured results:

- Compression time: `23.33s`
- CPU decompression time: `24.38s`
- Space savings: `25.42%`
- Max absolute error: `2.861e-05`
- `RMSE`: `5.268e-06`

What characterizes this configuration is that the error is very small, and compression still finishes within `30s`.

Comparison of very-low-error candidates:

| Method | Compress | CPU decompress | Space savings | Max abs error | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zfpy_tolerance_1e-4` | `23.33s` | `24.38s` | `25.42%` | `2.861e-05` | `5.268e-06` |
| `zfpy_tolerance_5e-4` | `21.72s` | `22.21s` | `34.79%` | `2.356e-04` | `4.198e-05` |
| `zfpy_tolerance_1e-3` | `21.18s` | `21.46s` | `37.92%` | `4.616e-04` | `8.395e-05` |
| `zfpy_precision_23` | `22.01s` | `22.54s` | `34.33%` | `4.482e-04` | `4.056e-05` |
| `zfpy_rate_16` | `19.22s` | `18.88s` | `50.00%` | `2.037e-02` | `1.401e-03` |

The reading here can be split into three tiers:

- If "smallest possible error" matters most, pick `zfpy_tolerance_1e-4`.
- If you want a compromise between "still very small error" and "saving more space", pick `zfpy_tolerance_5e-4`.
- `zfpy_rate_16` compresses harder, but its error is no longer in the "very low error" tier.

### 4.3 Methods Not Suitable For The Current Scenario

`fpzip`

- `fpzip_lossless` does not finish within `30s`.
- `fpzip_precision_28` actually has reasonable error, but still times out.

`SZ3`

- In earlier experiments, `SZ3` was competitive on compression ratio.
- But on this machine and through the current entry point, its compression time cannot meet the `30s` threshold.

`SVD` and `H-matrix`

- Both finish quickly.
- But this matrix is not low-rank, or rather it is not well suited to this kind of structural approximation.
- The compression ratio is bought by introducing very large reconstruction error, which does not fit the current requirements.

`gzip / bz2 / lzma`

- They are lossless.
- But they are too slow to meet the time budget.

### 4.4 Whether Compression Is Worthwhile From A Network-Transfer Perspective

If the goal is not "save space" but "shorten end-to-end transfer time", the right metric is `speedup`, not raw compression ratio.

The end-to-end model used in this report is:

`total time = compress + send-side checksum over the transmitted byte stream + network transfer + receive-side checksum over the transmitted byte stream + CPU decompress`

`checksum` is intentionally included, because once the transmitted bytes shrink, the time to checksum the transmitted byte stream also drops.

`SHA256` throughput on this machine is measured at roughly `1354 MiB/s`, so:

- One `SHA256` over the original `4 GiB` file takes about `3.03s`.
- One `SHA256` over the `blosc2_zstd_shuffle_c1` compressed payload takes about `2.56s`.
- One `SHA256` over the `zfpy_tolerance_5e-4` compressed payload takes about `1.97s`.

In other words, compression does shave off some checksum time, but the magnitude is small, typically only a few hundred milliseconds to about two seconds, not enough to flip the overall trend.

Under this model, the end-to-end `speedup` on typical links is:

| Method | 300 Mbps WiFi | 1 Gbps Ethernet |
| --- | ---: | ---: |
| `blosc2_zstd_shuffle_c1` | `1.064x` | `0.886x` |
| `zfpy_tolerance_1e-4` | `0.876x` | `0.519x` |
| `zfpy_tolerance_5e-4` | `0.984x` | `0.575x` |
| `zfpy_tolerance_1e-3` | `1.026x` | `0.597x` |

The interpretation here is critical:

- At `300 Mbps WiFi`, only `blosc2_zstd_shuffle_c1` shows a clear positive payoff, and even there it is only about `6.4%` faster.
- At the same `300 Mbps`, lossy `zfp` options, despite their higher compression ratio, still produce only marginal gains once compression and decompression are accounted for, and may even produce negative speedup.
- At `1 Gbps Ethernet`, every option shows negative speedup.

In terms of break-even points:

- `blosc2_zstd_shuffle_c1` breaks even at roughly `506 Mbps`.
- `zfpy_tolerance_5e-4` breaks even at roughly `286 Mbps`.
- `zfpy_tolerance_1e-3` breaks even at roughly `323 Mbps`.

This means:

- If the actual effective bandwidth is below a few hundred `Mbps`, compression may help end-to-end time.
- If it is already close to `1 Gbps`, compression is essentially meaningless as a transfer accelerator.

A stricter caveat:

- The model above assumes `checksum` is taken over the "transmitted byte stream".
- If the system actually requires a "checksum over the final decompressed original matrix",
- then that portion of time barely shrinks with compression, because in the end the same volume of original data still has to be checksummed.

So from a network-transfer standpoint, the conclusion on this dataset is:

- `WiFi 300 Mbps`: compression has only marginal value, and only the `blosc2` lossless option is even arguably worthwhile.
- `Ethernet 1 Gbps`: compressing solely for transfer speedup is essentially not worth it.
- If the real goal is to go faster, switching the data representation is usually a higher-leverage option to consider first, e.g. `FP16/BF16`, quantization, block-wise transfer, delta-only transfer, and similar.

## 5. Why GPU-Only Solutions Were Not Picked As The Default

This machine does have an NVIDIA GPU, but what actually decides the choice is not "can we compress on the GPU", but "can the receiver side reliably decompress with only a CPU".

That distinction directly changes the optimum:

- A GPU-specific format is not necessarily deployment-friendly.
- An option that the `CPU` side can stably decompress, where compression already takes only `6` to `23` seconds, is generally more practical than a faster but less compatible GPU-bound path.

I also cross-checked the official project descriptions:

- `cuSZp` is explicitly described as a GPU compression / decompression framework, and its environment requirements directly state `Linux OS with NVIDIA GPUs`.
- `nvCOMP` is also positioned as a GPU compression / decompression library in its official repo, although it does provide some CPU-interop examples.
- `MGARD` has both CPU and GPU branches, but it is primarily a scientific-data lossy-compression framework, not the most natural first pick for the current "default to bit-exact" requirement.

So in this project, the GPU is not without value, but rather:

- It can serve as a future direction for further optimizing sender-side throughput.
- It is not the deciding factor for the current "best deployable option".

## 6. Final Recommendation

### 6.1 Default Production Recommendation

If you need a single default option right now, the recommendation is:

`blosc2 + zstd + shuffle`

The corresponding best configuration in this round:

- `blosc2_zstd_shuffle_c1`

Reasons:

- Lossless.
- Fast.
- Highest compression ratio among currently usable lossless options.
- Fast CPU decompression.
- Already validated by full-file equality check.

### 6.2 If Very Low Error Is Acceptable

If you are willing to accept very small numerical error in exchange for a higher compression ratio, the priorities are:

- Most conservative option: `zfpy_tolerance_1e-4`
- More balanced option: `zfpy_tolerance_5e-4`

### 6.3 A Simple Decision Rule

- If the downstream side requires `checksum` equality: pick `blosc2_zstd_shuffle_c1`.
- If the downstream side allows tiny numerical error and you want to save more space: pick `zfpy_tolerance_1e-4` or `zfpy_tolerance_5e-4`.
- Do not treat `zfpy_rate_16` as a "low error" option.
- If the goal is "transfer over the network faster": only consider enabling compression seriously when the effective bandwidth is below roughly `500 Mbps`.

## 7. Related Output Files

After this report is archived under `SuperWeb-Cluster/docs`, the experiment scripts, raw results, and intermediate notes remain under `Client/download`. Viewed from the `docs` directory, the relative paths are:

- `../../Client/download/code/practical_cpu_decode_benchmark.py`
- `../../Client/download/artifacts/practical_cpu_decode_results.json`
- `../../Client/download/artifacts/practical_cpu_decode_summary.md`
- `../../Client/download/code/compression_benchmark.py`
- `../../Client/download/artifacts/compression_benchmark_30s_summary.md`
- `../../Client/download/code/matrix_compression_benchmark.py`
- `../../Client/download/artifacts/matrix_compression_30s_results.json`
- `../../Client/download/artifacts/matrix_compression_30s_summary.md`
- `../../Client/download/code/scientific_float_compression_benchmark.py`
- `../../Client/download/artifacts/scientific_float_compression_30s_results.json`
- `../../Client/download/artifacts/scientific_float_compression_30s_summary.md`
- `../../Client/download/artifacts/scientific_float_compression_notes.md`
- `../../Client/download/code/structural_matrix_compression_benchmark.py`
- `../../Client/download/artifacts/structural_matrix_compression_30s_results.json`
- `../../Client/download/artifacts/structural_matrix_compression_30s_summary.md`

## 8. References

For algorithmic capabilities and deployability judgments cited in this round, the following official project pages were consulted:

- ZFP: https://github.com/LLNL/zfp
- fpzip: https://github.com/LLNL/fpzip
- Python-Blosc2: https://github.com/Blosc/python-blosc2
- SZ3: https://github.com/szcompressor/SZ3
- cuSZp: https://github.com/szcompressor/cuSZp
- nvCOMP: https://github.com/NVIDIA/nvcomp
- MGARD: https://github.com/CODARcode/MGARD
- Turbo-Transpose: https://github.com/powturbo/Turbo-Transpose
