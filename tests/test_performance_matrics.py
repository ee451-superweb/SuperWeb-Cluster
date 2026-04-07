"""Smoke tests for the performance-matrix workspace."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PERF_DIR = Path(__file__).resolve().parents[1] / "performance metrics"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

import benchmark
from backends.cpu_backend import CpuBackend, _binary_tree_worker_candidates
from backends.cuda_backend import CudaBackend
from fmvm_dataset import ensure_dataset
from scoring import linear_time_score
from workloads import resolve_workload


class PerformanceMatricsTests(unittest.TestCase):
    def test_linear_score_midpoint_is_half(self) -> None:
        score = linear_time_score(0.6, ideal_seconds=0.2, zero_score_seconds=1.0, max_score=100.0)
        self.assertAlmostEqual(score, 50.0)

    def test_worker_binary_tree_sequence(self) -> None:
        self.assertEqual(_binary_tree_worker_candidates(16), [16, 8, 32, 4, 64])

    def test_generate_dataset_smoke(self) -> None:
        workload = resolve_workload("smoke")
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = ensure_dataset(Path(temp_dir), workload)
            self.assertTrue(dataset.matrix_path.exists())
            self.assertTrue(dataset.vector_path.exists())
            self.assertTrue(dataset.meta_path.exists())
            self.assertEqual(dataset.matrix_path.stat().st_size, workload.rows * workload.cols * 4)
            self.assertEqual(dataset.vector_path.stat().st_size, workload.cols * 4)

    def test_benchmark_smoke_cpu_only(self) -> None:
        backend = CpuBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("Windows C++ CPU backend is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch("backends.cpu_backend.os.cpu_count", return_value=1):
            args = argparse.Namespace(
                backend=["cpu"],
                preset="smoke",
                rows=None,
                cols=None,
                dataset_dir=Path(temp_dir),
                output=Path(temp_dir) / "result.json",
                time_budget=10.0,
                list_presets=False,
            )
            report = benchmark.run_benchmark(args)

        self.assertEqual(report["method"], "fixed_matrix_vector_multiplication")
        self.assertEqual(report["best_backend"], "cpu")
        self.assertIsNotNone(report["best_config"])
        self.assertIsNotNone(report["best_result"])
        assert report["best_result"] is not None
        self.assertTrue(report["best_result"]["verified"])

    def test_cuda_backend_gracefully_skips_without_nvcc(self) -> None:
        backend = CudaBackend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)


if __name__ == "__main__":
    unittest.main()
