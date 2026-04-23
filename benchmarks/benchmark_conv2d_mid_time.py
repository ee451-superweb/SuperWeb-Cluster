#!/usr/bin/env python3
"""Run the canonical **mid** conv2d benchmark with full autotune, then print timings.

This uses the same pipeline as ``compute_node/performance_metrics/conv2d/benchmark.py`` with
``--workload-mode mid``. The native CUDA runner sweeps candidates, records the **minimum**
per-configuration time as ``autotune_wall_clock_latency_seconds`` (fastest mid run found during
autotune). That value is what this script highlights as the primary result; the follow-up
``measurement_*`` phase is a separate timed pass using the winning config.

Usage (from repo root)::

    python benchmarks/benchmark_conv2d_mid_time.py
    python benchmarks/benchmark_conv2d_mid_time.py --backend cuda
    python benchmarks/benchmark_conv2d_mid_time.py --cooldown-ms 0 --output-channel-batch 32

**GPU utilization:** the CUDA runner inserts an optional sleep between output-channel
batches (``cooldown_ms``, cluster default 2.5 ms). That adds huge idle time on large
``c_out`` and keeps utilization well below 100%. This script defaults
``--cooldown-ms`` to **0** so autotune finishes sooner and the GPU stays busier; use
``--cooldown-ms 2.5`` only to mimic production throttling.

**Still slow:** mid autotune sweeps many (tile, block, output-channel-batch) combos;
pass ``--output-channel-batch N`` to skip the batch sweep and only tune tile/block.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_node.performance_metrics.conv2d.benchmark import build_parser, run_benchmark
from compute_node.performance_metrics.conv2d.workloads import get_mid_spec


def _trial_seconds(trial: dict | None, key: str = "wall_clock_latency_seconds") -> float | None:
    if not isinstance(trial, dict):
        return None
    value = trial.get(key)
    return float(value) if value is not None else None


def _parse_native_subprocess_wall_notes(notes: list[str]) -> dict[str, float] | None:
    """Parse ``native_subprocess_wall_s:`` lines added by the CUDA backend."""
    for line in notes:
        if not line.startswith("native_subprocess_wall_s:"):
            continue
        out: dict[str, float] = {}
        for name in ("autotune_sweep", "measurement_pass", "sum"):
            match = re.search(rf"{name}=([0-9]+(?:\.[0-9]+)?)", line)
            if match:
                out[name] = float(match.group(1))
        return out if out else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark mid-sized conv2d with autotune (best block/tile/output-channel batch) "
            "and print wall-clock times from the native runner."
        ),
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        help="One of: cuda, cpu, metal, all (every default backend), auto (default probe order). "
        "Default: cuda.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override dataset directory (default: compute_node/input_matrix/conv2d/generated).",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Write full benchmark JSON to this path.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of native runners.",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=float,
        default=0.0,
        help="Sleep between output-channel batches in the CUDA runner (ms). Default: 0 for faster "
        "runs and higher GPU duty cycle; use 2.5 to match app.constants cluster default.",
    )
    parser.add_argument(
        "--output-channel-batch",
        type=int,
        default=None,
        metavar="N",
        help="If set, autotune only this output-channel batch size (skips sweeping 1,2,4,...,c_out).",
    )
    cli = parser.parse_args()

    bench_argv: list[str] = [
        "--workload-mode",
        "mid",
        "--role",
        "compute",
        "--cooldown-ms",
        str(cli.cooldown_ms),
    ]
    backend = cli.backend.strip().lower()
    if backend != "auto":
        bench_argv.extend(["--backend", backend])
    if cli.dataset_dir is not None:
        bench_argv.extend(["--dataset-dir", str(cli.dataset_dir.resolve())])
    if cli.rebuild:
        bench_argv.append("--rebuild")
    if cli.output_channel_batch is not None:
        bench_argv.extend(["--output-channel-batch", str(cli.output_channel_batch)])

    bench_args = build_parser().parse_args(bench_argv)

    spec = get_mid_spec()
    print("Mid conv2d (canonical workload)", flush=True)
    print(
        f"  HxW={spec.h}x{spec.w}, Cin={spec.c_in}, Cout={spec.c_out}, "
        f"k={spec.k}, pad={spec.pad}, stride={spec.stride}",
        flush=True,
    )
    print(f"  CUDA cooldown-ms={cli.cooldown_ms} (0 recommended for timing; >0 lowers GPU utilization)", flush=True)
    if cli.output_channel_batch is not None:
        print(f"  output-channel-batch fixed to {cli.output_channel_batch} (narrower autotune)", flush=True)
    print("", flush=True)

    t0 = time.perf_counter()
    report = run_benchmark(bench_args)
    wall_total = time.perf_counter() - t0

    if cli.write_report is not None:
        cli.write_report.parent.mkdir(parents=True, exist_ok=True)
        cli.write_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full JSON report: {cli.write_report}\n", flush=True)

    print(f"End-to-end wall time (Python, incl. dataset generation if any): {wall_total:.3f} s\n", flush=True)

    backend_results = report.get("backend_results") or {}
    for name in sorted(backend_results.keys()):
        res = backend_results[name]
        if not res.get("available"):
            notes = res.get("notes") or []
            detail_parts = notes[1:] if len(notes) > 1 else notes
            detail = "\n  ".join(detail_parts) if detail_parts else "unknown"
            print(f"[{name}] unavailable:\n  {detail}", flush=True)
            continue

        autotune = res.get("autotune_result") or res.get("autotune_trial")
        best = res.get("best_result") or res.get("best_trial")
        cfg = res.get("best_config") or res.get("selected_config")

        print(f"[{name}] OK", flush=True)
        if cfg:
            print(f"  Winning config (used for measurement + checksum): {cfg}", flush=True)

        aut_s = _trial_seconds(autotune)
        aut_gflops = (
            float(autotune["effective_gflops"])
            if isinstance(autotune, dict) and autotune.get("effective_gflops") is not None
            else None
        )
        meas_s = _trial_seconds(best)
        meas_gflops = float(best["effective_gflops"]) if isinstance(best, dict) and best.get("effective_gflops") is not None else None

        # Native runner: autotune_wall_clock_latency_seconds == fastest candidate time (mid workload).
        print("  --- Primary (fastest mid time after autotune sweep) ---", flush=True)
        if aut_s is not None:
            print(f"  Time: {aut_s:.9f} s  (autotune_wall_clock_latency_seconds)", flush=True)
        if aut_gflops is not None:
            print(f"  Effective GFLOPS: {aut_gflops:.2f}  (from autotune phase)", flush=True)

        print("  --- Secondary (timed pass with winning config, measurement_repeats) ---", flush=True)
        if meas_s is not None:
            print(f"  Time: {meas_s:.9f} s  (measurement_wall_clock_latency_seconds)", flush=True)
        if meas_gflops is not None:
            print(f"  Effective GFLOPS: {meas_gflops:.2f}", flush=True)
        print("", flush=True)

        wall_native = _parse_native_subprocess_wall_notes(res.get("notes") or [])
        if wall_native and "autotune_sweep" in wall_native and "measurement_pass" in wall_native:
            at = wall_native["autotune_sweep"]
            mp = wall_native["measurement_pass"]
            sm = wall_native.get("sum", at + mp)
            rest = max(0.0, wall_total - sm)
            print("  --- Wall-clock breakdown (CUDA native .exe subprocesses) ---", flush=True)
            print(
                f"  1st process (full autotune sweep inside runner): {at:.3f} s",
                flush=True,
            )
            print(
                f"  2nd process (measurement only, fixed best config): {mp:.3f} s",
                flush=True,
            )
            print(f"  Native runner sum: {sm:.3f} s", flush=True)
            print(
                f"  Rest of end-to-end (~dataset, probe, Python): ~{rest:.3f} s",
                flush=True,
            )
            print("", flush=True)
            print(
                "  (JSON autotune_wall_clock_latency_seconds is the *fastest candidate’s* one-forward time,",
                flush=True,
            )
            print(
                "   not the length of the autotune sweep; use 1st process row above for sweep wall time.)",
                flush=True,
            )
        print("", flush=True)

    best_backend = report.get("best_backend")
    if best_backend:
        br = backend_results.get(best_backend) or {}
        autotune = br.get("autotune_result") or br.get("autotune_trial")
        fastest = _trial_seconds(autotune)
        print(
            f"Summary (ranked best backend={best_backend}): "
            f"fastest mid time after autotune = {fastest:.9f} s"
            if fastest is not None
            else f"Summary: ranked best backend={best_backend}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
