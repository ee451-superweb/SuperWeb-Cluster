"""Task-executor command wiring tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from compute_node.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile
from compute_node.task_executor import FixedMatrixVectorTaskExecutor, ProcessorTaskSlice


class TaskExecutorCommandTests(unittest.TestCase):
    """Keep task-runtime iteration semantics distinct from benchmark repeats."""

    def test_cpu_task_command_uses_iteration_count_flag(self) -> None:
        inventory = RuntimeProcessorInventory(
            processors=(
                RuntimeProcessorProfile(
                    hardware_type="cpu",
                    effective_gflops=24.0,
                    rank=1,
                    best_config={"workers": 8, "tile_size": 512},
                ),
            )
        )
        executor = FixedMatrixVectorTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        command = executor._build_runtime_command(
            ProcessorTaskSlice(processor=inventory.processors[0], row_start=0, row_end=4),
            7,
            Path("C:/tmp/x.bin"),
            Path("C:/tmp/y.bin"),
        )

        self.assertIn("--iteration-count", command)
        self.assertNotIn("--measurement-repeats", command)
        self.assertEqual(command[command.index("--iteration-count") + 1], "7")

    def test_cuda_task_command_uses_iteration_count_flag(self) -> None:
        inventory = RuntimeProcessorInventory(
            processors=(
                RuntimeProcessorProfile(
                    hardware_type="cuda",
                    effective_gflops=125.0,
                    rank=1,
                    best_config={"block_size": 256, "tile_size": 1, "transpose": False},
                ),
            )
        )
        executor = FixedMatrixVectorTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        command = executor._build_runtime_command(
            ProcessorTaskSlice(processor=inventory.processors[0], row_start=4, row_end=8),
            9,
            Path("C:/tmp/x.bin"),
            Path("C:/tmp/y.bin"),
        )

        self.assertIn("--iteration-count", command)
        self.assertNotIn("--measurement-repeats", command)
        self.assertEqual(command[command.index("--iteration-count") + 1], "9")


if __name__ == "__main__":
    unittest.main()
