"""Task-executor command wiring tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.constants import DX12_BACKEND_DISABLED_REASON
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
        self.assertEqual(command[command.index("--accumulation-precision") + 1], "fp32")

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
        self.assertEqual(command[command.index("--accumulation-precision") + 1], "fp32")

    def test_dx12_task_command_is_disabled(self) -> None:
        inventory = RuntimeProcessorInventory(
            processors=(
                RuntimeProcessorProfile(
                    hardware_type="dx12",
                    effective_gflops=42.0,
                    rank=1,
                    best_config={"thread_group_size": 128, "rows_per_thread": 4},
                ),
            )
        )
        executor = FixedMatrixVectorTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        with self.assertRaisesRegex(ValueError, DX12_BACKEND_DISABLED_REASON):
            executor._build_runtime_command(
                ProcessorTaskSlice(processor=inventory.processors[0], row_start=8, row_end=12),
                11,
                Path("C:/tmp/x.bin"),
                Path("C:/tmp/y.bin"),
            )


if __name__ == "__main__":
    unittest.main()
