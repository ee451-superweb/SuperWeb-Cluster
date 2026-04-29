# Conv2d Runtime Debugging Notes — 2026-04-21

This round of debugging on `compute_node/compute_methods/conv2d` concentrated on fixing four issues. Each is recorded under "Symptom / Root Cause / Fix / Verification" so it can be revisited later.

## 1. Native runner stderr lost in TASK_FAIL logs

### Symptom
When the CUDA runner crashed, the main process log only showed `subprocess.CalledProcessError: Command '[...]' returned non-zero exit status 1.`, with no underlying CUDA error information visible. The previous debugging session even ended with the cluster shutting down entirely, so the original stderr text never even survived on screen.

### Root Cause
Two layers compounded the problem:

1. The conv2d task runs inside a ProcessPoolExecutor child process. The child has no logging handlers configured by default, so `_LOGGER.error(...)` there is a **dead path** — anything written goes nowhere and never appears in the main process log.
2. `subprocess.CalledProcessError.__str__()` only includes the command and exit code, not `stderr`. Even if you re-raise it as-is and let pickle carry it back to the main process, the stderr field does survive the trip but is invisible in the formatted TASK_FAIL message.

### Fix
`executor.py` introduces a new `RunnerProcessError(RuntimeError)` exception type whose constructor embeds the tail of stderr directly into the message string. After catching `CalledProcessError` / `TimeoutExpired`, `_run_native_runner` re-raises as `RunnerProcessError`, with a message that includes:

- command
- exit code
- stderr tail (`_tail_stream`, capped at 2KB)
- stdout tail

### Verification
A new regression test `test_runner_process_error_carries_stderr_through_pickle` confirms that after a pickle round-trip, stderr is still present in `str(exc)`. When CUDA 716 was reproduced afterward, the main log's TASK_FAIL line directly showed `stderr_tail='CUDA error at ... code=716 "misaligned address"'`.

### Lesson
**Pack diagnostic information into the exception's str; do not rely on the child process's logger.** This rule has been written into auto-memory `project_conv2d_runs_in_subprocess.md`.

---

## 2. CUDA `cudaErrorMisalignedAddress` (716)

### Symptom
Task slices where `c_out % 4 != 0` reliably triggered CUDA error 716. For example, `c_out=256` split into 3 parts gave worker-1 `oc=0..121` and worker-2 `oc=121..242`, where worker-2's starting offset of `121` is not a multiple of 4, and the runner crashed inside the kernel.

### Root Cause
`accumulate_output_lanes` / `store_output_lanes` use `float4` to load/store 4 lanes at a time. `float4` requires the pointer to be 16B aligned. The weight tensor is laid out as `[kh][kw][ic][oc]`, with the innermost stride being `c_out`. Only when both `c_out` and the task's starting `oc` are divisible by 4 is the float4 starting address guaranteed to be aligned; otherwise the runner trips 716.

### Fix
The kernel adds a runtime alignment check before entering the `float4` branch:

```cpp
if (valid_outputs == kOutputsPerThread &&
    (reinterpret_cast<uintptr_t>(weight_ptr) & 0xF) == 0) {
    // float4 fast path
} else {
    // scalar fallback
}
```

The same treatment is applied to `output_ptr`. This way the upper-layer partitioning strategy is not constrained, and the runner itself absorbs strides that are not divisible by 4.

### Verification
Re-running the `c_out=256 / 3 worker` partition no longer produced 716; the runner exited cleanly with 0.

---

## 3. CUDA worker 3.6–7.6× slower than CPU — autotune ran on every task

### Symptom
The first run of a 3-worker mixed cluster (2 CUDA + 1 CPU) had the CUDA nodes coming in **3–7× slower per channel than the CPU**. Intuition says it should be the other way around.

### Root Cause
Comparing the command-line construction for the Metal and CUDA backends in `executor.py`: the Metal branch passed `block_size` / `tile_size` from `best_config` to the runner, while the CUDA branch did not.

When the CUDA runner has no explicit `--block-sizes` / `--tile-sizes`, it falls into its built-in autotune — 14 tile candidates, each running 1 warmup + 1 measurement, totaling **28 full forward passes per task**. The benchmark sweep had already run once (results stored in `result.json`), but the runtime was not consuming that cache and instead re-swept from scratch for every dispatched task.

### Fix
The CUDA branch now also reads `block_size` / `tile_size` from `best_config`, and when present pins them to the runner via `--block-sizes` / `--tile-sizes`:

```python
elif backend_name == "cuda":
    ...
    cuda_block_size = int(best_config.get("block_size") or 0)
    cuda_tile_size = int(best_config.get("tile_size") or 0)
    if cuda_block_size > 0 and cuda_tile_size > 0:
        cmd.extend([
            "--block-sizes", str(cuda_block_size),
            "--tile-sizes", str(cuda_tile_size),
        ])
```

`test_task_executor.py` got new positive / negative assertion tests.

### Verification
Per-channel compute time on the large task: **2918ms → 506ms** (≈5.8× speedup). CUDA per-channel became 1.6–1.9× faster than CPU, consistent with the relative performance the benchmark sweep had reported.

### Lesson
**The runtime should consume the benchmark's results, not re-sweep every time.** The two backend branches need their command-line construction conventions aligned, otherwise either backend silently falling into autotune is very hard to spot.

---

## 4. Anomalous peripheral time on large tasks — 49 seconds of pure Python iteration

### Symptom
Even after the autotune fix, `logs/main-20260421-231411.txt` showed strange overhead on the `conv2d-3` task:

```
worker-3 (oc=242:256, 14 ch, CPU)  wall:20780ms  compute:13521ms  peripheral:7032ms
worker-1 (oc=0:121,  121 ch, CUDA) wall:110821ms compute:61180ms  peripheral:48800ms
worker-2 (oc=121:242,121 ch, CUDA) wall:111833ms compute:61665ms  peripheral:49121ms
```

CUDA nodes had **48–49 seconds of peripheral**, while the CPU node only had 7. Compute had been sped up; now peripheral was the dominant contributor to wall time.

### Root Cause
In `stats_only` mode, after each worker finishes the runner, it calls `_summarize_conv2d_slice_file` to scan the output file and compute `sum` / `sum_sq` / samples. The original implementation was:

```python
for x in values:           # array.array('f'), pure Python iterator
    xf = float(x)
    sum_v += xf
    sum_sq += xf * xf
    ...
```

The CPython interpreter loop processes about 2–3M float elements per second:
- CPU worker output is 14×H×W ≈ 140M float / 14 = **14× smaller** → 7 seconds;
- CUDA worker output is 121×H×W ≈ 1.2 billion floats → 49 seconds.

The ratio matches exactly.

### Fix
Switch to `numpy` vectorization, while still keeping 1MB chunked streaming reads to bound memory:

```python
with path.open("rb") as handle:
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        values = np.frombuffer(chunk, dtype=np.float32)
        values64 = values.astype(np.float64)
        sum_v += float(values64.sum())
        sum_sq += float(np.dot(values64, values64))
        if remaining > 0:
            take = min(remaining, values.size)
            samples.extend(float(x) for x in values[:take])
            remaining -= take
```

The `float64` accumulator preserves the same precision as the original Python `float` accumulator.

### Verification
- Parameter consistency test (100k floats): `sum` is bit-exact, `sum_sq` has a relative error of `8e-14`, samples are bit-exact.
- 120M float micro-benchmark: **49s → 435ms** (about 113× speedup).
- All 18 tests in `tests/test_task_executor.py` + `tests/test_runner_failure_log.py` pass.

### Lesson
**Do not iterate numerical data with a Python loop.** array / list / interpreter object boxing on every float operation amplifies microsecond differences into tens of seconds at the 1e8 element scale. The next time peripheral is more than a tenth of compute, suspect a Python loop first.

---

## 5. Capacity alignment — dispatch's default entry point flipped from benchmark to single pass

### Symptom
After section 3 pinned `block_size` / `tile_size`, the CUDA per-channel compute was already consistent with the relative performance from the benchmark sweep. But the upper-layer capacity bookkeeping was still distorted: the benchmark's reported per-channel time is the **cudaEvent time of a single kernel** (millisecond range), while the runtime-recorded `computation_ms_total` is the **subprocess wall-clock**, which includes process spawn, CUDA context init, D2H, checksum, file writes, and other resident overhead. The anomaly where CUDA was assigned 9× the work yet finished after CPU was the subprocess wall folding this resident overhead into the per-channel time.

### Root Cause
The runner's **default entry point is benchmark**: `main()` jumps straight into autotune over 14 candidates + multi-pass measurement + a separate output/checksum pass. Even when the executor has pinned a single candidate via `--block-sizes`/`--tile-sizes`, the runner still does multiple passes plus the separate output pass. The subprocess wall therefore naturally equals "benchmark structure + process resident overhead", and the number reaching the capacity algorithm is systematically inflated, with the inflation more pronounced for CUDA than for CPU (context + D2H).

### Fix
Flip the runner's default entry point: **dispatch by default, `--mode benchmark` triggers autotune + measurement**.

- `conv2d_cuda_runner.cu`: introduces `enum class RunnerMode { Dispatch, Benchmark }`, adds `--mode` / `--shared-input` CLI; the dispatch branch wraps kernel launch + sync + D2H + host copy in a single `cudaEventRecord(start)..(stop)` bracket, and the JSON gains `mode` and `compute_event_ms`. Only `--mode benchmark` runs the full 14-candidate autotune. (Cost: dispatch's cudaEvent includes ~3ms of D2H, slightly inflated relative to pure kernel, but much closer to the benchmark than subprocess wall.)
- `conv2d_cpu_windows.cpp` / `conv2d_cpu_macos.cpp`: same `--mode` branching. Dispatch uses `workers[0]` (passed in from the executor with the value selected by benchmark) to run `run_multithreaded` once, with `compute_event_ms = chrono single compute` converted from seconds to milliseconds.
- `conv2d_metal_runner.mm`: `--mode dispatch` defaults to a single MPSGraph submission; `--mode benchmark` keeps the autotune + measurement two-pass.
- `_run_runner` in `performance_metrics/conv2d/backends/{cuda,metal}_backend.py` explicitly injects `--mode benchmark`, ensuring the benchmark sweep path is unchanged from before.
- `executor.py`: the dispatch path pins `--shared-input` (otherwise `build_candidate_tiles` returns both shared=0/1 variants, and the first one dispatch picks may not match what benchmark selected); a new `_parse_compute_event_ms` parses `compute_event_ms` from the runner stdout JSON, and `computation_ms_total += min(subprocess_wall_ms, compute_event_ms)`, with subprocess wall retained only on the fallback path.

### Verification
- Unit: 306 passed / 8 skipped. New assertions: `cuda_backend._run_runner` command contains `--mode benchmark`; the executor dispatch command contains `--shared-input` and **does not contain** `--mode` (so it takes the default dispatch).
- Runner smoke test (`h=w=64, c_in=32, c_out=64, k=3`):
  - CPU dispatch `compute_event_ms=27.01ms` (worker=1) vs. benchmark sweep over 1/2/4 selecting `compute_event_ms=10.47ms` (autotune 9.95ms, 3 candidates).
  - CUDA dispatch `compute_event_ms=1.22ms` (default tile) vs. benchmark sweep over 14 candidates `compute_event_ms=0.316ms` (autotune 0.302ms).
  - In both dispatch and benchmark modes, both backends correctly emit `mode`/`compute_event_ms`/`trials_run` fields in JSON.
- Stage 6c's 2-worker/3-worker real-cluster comparison (whether CUDA vs CPU per-channel time matches the benchmark ratio) is operator-in-loop and is left for the next cluster run to verify.

### Lesson
**The capacity signal must be sampled in the same time domain as the benchmark signal.** If benchmark reports cudaEvent time, the runtime cannot use subprocess wall; otherwise the two differ by a systematic offset of "process resident overhead + autotune sweep + separate output pass", and multi-machine scheduling will inevitably be distorted. The default entry point should also be the one closer to the real dispatch path — autotune is safer as an explicit opt-in than as the default.

---

## Retrospective

These five issues share one pattern: **a performance or error signal got silently swallowed at some layer**.

- stderr swallowed by the child process logger → wrap into the exception str
- CUDA alignment error swallowed by `CalledProcessError` → same as above
- Autotune implicitly enabled, benchmark results ignored → align command-line construction across both backends
- Interpreter loop swallowed the chance for numpy vectorization → switch to numpy
- Dispatch treated autotune + resident overhead as compute → flip the runner's default entry point, consume `compute_event_ms`

The next time a number is off by an order of magnitude from expectations, check first which link in the chain is silently doing more work than expected.
