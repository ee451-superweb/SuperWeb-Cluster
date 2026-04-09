"""Tests for the simplified performance-metrics workspace."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PERF_DIR = Path(__file__).resolve().parents[1] / "compute_node" / "performance metrics"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

import benchmark
from backends.cpu_backend import (
    CpuBackend,
    _binary_tree_worker_candidates,
    _candidate_tile_sizes as cpu_candidate_tile_sizes,
    _default_repeats as cpu_default_repeats,
)
from backends.cuda_backend import (
    CudaBackend,
    _candidate_block_sizes as cuda_candidate_block_sizes,
    _candidate_tile_sizes as cuda_candidate_tile_sizes,
    _candidate_transpose_modes as cuda_candidate_transpose_modes,
    _default_repeats as cuda_default_repeats,
)
from fmvm_dataset import (
    build_dataset_layout,
    compare_float32_vectors,
    dataset_is_generated,
    generate_dataset,
    load_float32_file,
)
from path_utils import to_relative_cli_path
from scoring import linear_time_score
from workloads import build_benchmark_spec
import subprocess


class PerformanceMatricsTests(unittest.TestCase):
    def test_linear_score_midpoint_is_half(self) -> None:
        score = linear_time_score(0.6, ideal_seconds=0.2, zero_score_seconds=1.0, max_score=100.0)
        self.assertAlmostEqual(score, 50.0)

    def test_worker_binary_tree_sequence(self) -> None:
        self.assertEqual(_binary_tree_worker_candidates(16), [16, 8, 32, 4, 64])

    def test_default_spec_matches_requested_2gib_shape(self) -> None:
        spec = build_benchmark_spec()
        self.assertEqual(spec.rows, 16_384)
        self.assertEqual(spec.cols, 32_768)
        self.assertEqual(spec.matrix_bytes, 2 * 1024**3)
        self.assertEqual(spec.vector_bytes, 32_768 * 4)

    def test_generate_dataset_with_small_override(self) -> None:
        spec = build_benchmark_spec(rows=8, cols=16)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            self.assertTrue(dataset_is_generated(layout, spec))
            self.assertEqual(layout.matrix_path.name, "A.bin")
            self.assertEqual(layout.vector_path.name, "x.bin")
            self.assertEqual(layout.matrix_path.stat().st_size, spec.matrix_bytes)
            self.assertEqual(layout.vector_path.stat().st_size, spec.vector_bytes)

    def test_dataset_is_generated_rejects_mismatched_metadata(self) -> None:
        spec = build_benchmark_spec(rows=8, cols=16)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
            metadata["benchmark"]["rows"] = 9
            layout.meta_path.write_text(json.dumps(metadata), encoding="utf-8")

            self.assertFalse(dataset_is_generated(layout, spec))

    def test_small_override_uses_override_dataset_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            default_dataset_dir = temp_root / "generated"
            args = argparse.Namespace(
                backend=["cpu"],
                dataset_dir=default_dataset_dir,
                output=temp_root / "result.json",
                time_budget=30.0,
                rows=8,
                cols=16,
            )
            with mock.patch.object(benchmark, "DEFAULT_DATASET_DIR", default_dataset_dir):
                report = benchmark.run_benchmark(args)

        dataset_root = Path(report["dataset"]["root_dir"])
        self.assertEqual(dataset_root.parts[-3:], ("generated", "overrides", "8x16"))

    def test_benchmark_auto_generates_and_runs_cpu(self) -> None:
        backend = CpuBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("Windows C++ CPU backend is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch("backends.cpu_backend.os.cpu_count", return_value=1):
            args = argparse.Namespace(
                backend=["cpu"],
                dataset_dir=Path(temp_dir),
                output=Path(temp_dir) / "result.json",
                time_budget=30.0,
                rows=8,
                cols=16,
            )
            report = benchmark.run_benchmark(args)

        self.assertEqual(report["method"], "fixed_matrix_vector_multiplication")
        self.assertEqual(report["best_backend"], "cpu")
        self.assertEqual(report["ranking"], ["cpu"])
        self.assertIn("backend_results", report)
        assert isinstance(report["backend_results"], dict)
        self.assertFalse(Path(report["dataset"]["root_dir"]).is_absolute())
        self.assertFalse(Path(report["dataset"]["matrix_path"]).is_absolute())
        self.assertFalse(Path(report["dataset"]["vector_path"]).is_absolute())
        self.assertIn("cpu", report["backend_results"])
        cpu_result = report["backend_results"]["cpu"]
        self.assertTrue(cpu_result["available"])
        self.assertEqual(cpu_result["rank"], 1)
        self.assertIsNotNone(cpu_result["best_config"])
        self.assertIsNotNone(cpu_result["best_result"])
        assert cpu_result["best_result"] is not None
        drive_pattern = re.compile(r"[A-Za-z]:[\\/]")
        self.assertFalse(any(drive_pattern.search(note or "") for note in cpu_result["notes"]))
        self.assertIn("checksum", cpu_result["best_result"])
        self.assertIn("effective_gflops", cpu_result["best_result"])

    def test_cuda_backend_probe_returns_status(self) -> None:
        backend = CudaBackend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)

    def test_cuda_backend_smoke_with_small_override(self) -> None:
        backend = CudaBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("CUDA backend is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir:
            spec = build_benchmark_spec(rows=8, cols=16)
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            result = backend.run(spec, layout, time_budget_seconds=60.0)

        self.assertTrue(result.available)
        self.assertIsNotNone(result.best_trial)
        assert result.best_trial is not None
        self.assertGreater(result.best_trial.effective_gflops, 0.0)
        self.assertTrue(result.best_trial.checksum.startswith("fnv1a64:"))

    def test_cpu_and_cuda_match_within_fp32_tolerance_on_small_override(self) -> None:
        cpu_backend = CpuBackend()
        cuda_backend = CudaBackend()
        cpu_available, _ = cpu_backend.probe()
        cuda_available, _ = cuda_backend.probe()
        if not cpu_available or not cuda_available:
            self.skipTest("CPU/CUDA cross-implementation comparison is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch("backends.cpu_backend.os.cpu_count", return_value=1):
            spec = build_benchmark_spec(rows=8, cols=16)
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)

            cpu_executable = cpu_backend._compile_if_needed()
            cuda_executable = cuda_backend._compile_if_needed()
            cpu_output = Path(temp_dir) / "cpu_output.bin"
            cuda_output = Path(temp_dir) / "cuda_output.bin"

            cpu_command = [
                str(cpu_executable),
                "--matrix",
                to_relative_cli_path(layout.matrix_path, start=PERF_DIR),
                "--vector",
                to_relative_cli_path(layout.vector_path, start=PERF_DIR),
                "--output",
                to_relative_cli_path(cpu_output, start=PERF_DIR),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--workers",
                ",".join(str(value) for value in _binary_tree_worker_candidates(1)),
                "--tile-sizes",
                ",".join(str(value) for value in cpu_candidate_tile_sizes(spec.cols)),
                "--repeats",
                str(cpu_default_repeats(spec)),
            ]
            cuda_command = [
                str(cuda_executable),
                "--matrix",
                to_relative_cli_path(layout.matrix_path, start=PERF_DIR),
                "--vector",
                to_relative_cli_path(layout.vector_path, start=PERF_DIR),
                "--output",
                to_relative_cli_path(cuda_output, start=PERF_DIR),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--transpose-modes",
                ",".join(str(value) for value in cuda_candidate_transpose_modes()),
                "--block-sizes",
                ",".join(str(value) for value in cuda_candidate_block_sizes()),
                "--tile-sizes",
                ",".join(str(value) for value in cuda_candidate_tile_sizes()),
                "--repeats",
                str(cuda_default_repeats(spec)),
            ]
            subprocess.run(cpu_command, check=True, capture_output=True, text=True, cwd=PERF_DIR)
            subprocess.run(cuda_command, check=True, capture_output=True, text=True, cwd=PERF_DIR)

            cpu_values = load_float32_file(cpu_output)
            cuda_values = load_float32_file(cuda_output)
            max_abs_error, max_rel_error, _abs_index, _rel_index = compare_float32_vectors(cpu_values, cuda_values)

        self.assertLessEqual(max_abs_error, 1e-3)
        self.assertLessEqual(max_rel_error, 1e-2)


if __name__ == "__main__":
    unittest.main()
