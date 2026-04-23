"""Tests for compute-node runtime performance summary filtering."""

import json
import tempfile
import unittest
from pathlib import Path

from compute_node.compute_methods.conv2d import executor as spatial_executor
from compute_node.performance_metrics.performance_summary import load_compute_performance_summary, load_runtime_processor_inventory


class ComputeNodePerformanceSummaryTests(unittest.TestCase):
    """Validate weak-processor filtering before worker registration."""

    def test_load_runtime_processor_inventory_filters_weak_backend(self) -> None:
        payload = {
            "backend_results": {
                "cuda": {
                    "available": True,
                    "rank": 1,
                    "best_config": {"block_size": 256, "tile_size": 1, "transpose": False},
                    "best_result": {"effective_gflops": 100.0},
                },
                "cpu": {
                    "available": True,
                    "rank": 2,
                    "best_config": {"workers": 8, "tile_size": 512},
                    "best_result": {"effective_gflops": 10.0},
                },
            },
            "ranking": ["cuda", "cpu"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            inventory = load_runtime_processor_inventory(result_path)
            summary = load_compute_performance_summary(result_path)

        self.assertEqual(len(inventory.processors), 1)
        self.assertEqual(inventory.processors[0].hardware_type, "cuda")
        self.assertEqual(summary.hardware_count, 1)
        self.assertEqual(summary.ranked_hardware[0].hardware_type, "cuda")

    def test_load_runtime_processor_inventory_filters_disabled_dx12_backend(self) -> None:
        payload = {
            "backend_results": {
                "dx12": {
                    "available": True,
                    "rank": 1,
                    "best_config": {"thread_group_size": 128, "rows_per_thread": 1},
                    "best_result": {"effective_gflops": 500.0},
                },
                "cpu": {
                    "available": True,
                    "rank": 2,
                    "best_config": {"workers": 8, "tile_size": 512},
                    "best_result": {"effective_gflops": 50.0},
                },
            },
            "ranking": ["dx12", "cpu"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            inventory = load_runtime_processor_inventory(result_path)
            summary = load_compute_performance_summary(result_path)

        self.assertEqual(len(inventory.processors), 1)
        self.assertEqual(inventory.processors[0].hardware_type, "cpu")
        self.assertEqual(summary.hardware_count, 1)
        self.assertEqual(summary.ranked_hardware[0].hardware_type, "cpu")

    def test_pinned_backend_keeps_only_named_backend(self) -> None:
        payload = {
            "methods": {
                "gemv": {
                    "backend_results": {
                        "cuda": {
                            "available": True,
                            "rank": 1,
                            "best_config": {"block_size": 256, "tile_size": 1, "transpose": False},
                            "best_result": {"effective_gflops": 100.0},
                        },
                        "cpu": {
                            "available": True,
                            "rank": 2,
                            "best_config": {"workers": 8, "tile_size": 512},
                            "best_result": {"effective_gflops": 60.0},
                        },
                    },
                    "ranking": ["cuda", "cpu"],
                },
                "conv2d": {
                    "backend_results": {
                        "cuda": {
                            "available": True,
                            "rank": 1,
                            "best_config": {"tile": 16},
                            "best_result": {"effective_gflops": 200.0},
                        },
                        "cpu": {
                            "available": True,
                            "rank": 2,
                            "best_config": {"workers": 8},
                            "best_result": {"effective_gflops": 120.0},
                        },
                    },
                    "ranking": ["cuda", "cpu"],
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            inventory = load_runtime_processor_inventory(result_path, pinned_backend="cpu")
            summary = load_compute_performance_summary(result_path, pinned_backend="cpu")

        self.assertEqual(len(inventory.processors), 1)
        self.assertEqual(inventory.processors[0].hardware_type, "cpu")
        self.assertEqual(
            {method.method for method in summary.method_summaries},
            {"gemv", "conv2d"},
        )
        for method_summary in summary.method_summaries:
            self.assertEqual(
                [ranked.hardware_type for ranked in method_summary.ranked_hardware],
                ["cpu"],
                msg=f"method={method_summary.method} should only report cpu",
            )

    def test_pinned_backend_returns_empty_when_backend_absent(self) -> None:
        payload = {
            "backend_results": {
                "cuda": {
                    "available": True,
                    "rank": 1,
                    "best_config": {"block_size": 256},
                    "best_result": {"effective_gflops": 100.0},
                },
            },
            "ranking": ["cuda"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            inventory = load_runtime_processor_inventory(result_path, pinned_backend="metal")

        self.assertEqual(inventory.processors, ())


class Conv2dPinnedBackendSelectionTests(unittest.TestCase):
    """Ensure conv2d backend selection respects pinned_backend at dispatch time."""

    def _write_payload(self, directory: Path) -> Path:
        payload = {
            "methods": {
                "conv2d": {
                    "backend_results": {
                        "cuda": {
                            "available": True,
                            "rank": 1,
                            "best_config": {"tile": 16},
                            "best_result": {"effective_gflops": 200.0},
                        },
                        "cpu": {
                            "available": True,
                            "rank": 2,
                            "best_config": {"workers": 8},
                            "best_result": {"effective_gflops": 120.0},
                        },
                    },
                    "ranking": ["cuda", "cpu"],
                },
            },
        }
        result_path = directory / "result.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return result_path

    def test_best_backend_profile_returns_pinned_backend_even_when_not_top_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = self._write_payload(Path(temp_dir))
            profile = spatial_executor._best_backend_profile(result_path, pinned_backend="cpu")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.hardware_type, "cpu")

    def test_best_backend_name_returns_pinned_backend_even_without_result_file(self) -> None:
        name = spatial_executor._best_backend_name(Path("/does/not/exist.json"), pinned_backend="cpu")
        self.assertEqual(name, "cpu")


if __name__ == "__main__":
    unittest.main()
