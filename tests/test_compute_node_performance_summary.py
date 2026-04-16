"""Tests for compute-node runtime performance summary filtering."""

import json
import tempfile
import unittest
from pathlib import Path

from compute_node.performance_summary import load_compute_performance_summary, load_runtime_processor_inventory


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


if __name__ == "__main__":
    unittest.main()
