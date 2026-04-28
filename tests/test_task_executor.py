"""Task-executor command wiring tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import subprocess

from core.constants import DX12_BACKEND_DISABLED_REASON, METHOD_CONV2D
from compute_node.compute_methods.conv2d import executor as spatial_executor
from compute_node.performance_metrics.conv2d.models import BenchmarkSpec as SpatialBenchmarkSpec
from compute_node.performance_metrics.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile
from compute_node.task_executor import GemvTaskExecutor, ProcessorTaskSlice
from wire.internal_protocol.transport import TaskAssign, TransferMode


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
        executor = GemvTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        with mock.patch.object(executor, "_resolve_runtime_executable_path", return_value=Path("C:/tmp/gemv_cpu_runner")):
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

    def test_cpu_task_command_uses_benchmark_selected_workers(self) -> None:
        inventory = RuntimeProcessorInventory(
            processors=(
                RuntimeProcessorProfile(
                    hardware_type="cpu",
                    effective_gflops=24.0,
                    rank=1,
                    best_config={"workers": 16, "tile_size": 512},
                ),
            )
        )
        executor = GemvTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        with mock.patch.object(executor, "_resolve_runtime_executable_path", return_value=Path("C:/tmp/gemv_cpu_runner")):
            command = executor._build_runtime_command(
                ProcessorTaskSlice(processor=inventory.processors[0], row_start=0, row_end=4),
                7,
                Path("C:/tmp/x.bin"),
                Path("C:/tmp/y.bin"),
            )

        self.assertIn("--fixed-workers", command)
        self.assertEqual(command[command.index("--fixed-workers") + 1], "16")

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
        executor = GemvTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
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

    def test_metal_task_command_uses_slice_flags_and_iteration_count(self) -> None:
        inventory = RuntimeProcessorInventory(
            processors=(
                RuntimeProcessorProfile(
                    hardware_type="metal",
                    effective_gflops=180.0,
                    rank=1,
                    best_config={"block_size": 256, "tile_size": 8},
                ),
            )
        )
        executor = GemvTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        with mock.patch.object(executor, "_resolve_runtime_executable_path", return_value=Path("C:/tmp/gemv_metal_runner")):
            command = executor._build_runtime_command(
                ProcessorTaskSlice(processor=inventory.processors[0], row_start=12, row_end=20),
                5,
                Path("C:/tmp/x.bin"),
                Path("C:/tmp/y.bin"),
            )

        self.assertIn("--row-start", command)
        self.assertIn("--row-end", command)
        self.assertEqual(command[command.index("--row-start") + 1], "12")
        self.assertEqual(command[command.index("--row-end") + 1], "20")
        self.assertIn("--block-sizes", command)
        self.assertEqual(command[command.index("--block-sizes") + 1], "256")
        self.assertIn("--tile-sizes", command)
        self.assertEqual(command[command.index("--tile-sizes") + 1], "8")
        self.assertNotIn("--headroom-fraction", command)
        self.assertIn("--row-chunk-size", command)
        self.assertEqual(command[command.index("--row-chunk-size") + 1], "8")
        self.assertIn("--iteration-count", command)
        self.assertEqual(command[command.index("--iteration-count") + 1], "5")

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
        executor = GemvTaskExecutor(inventory, dataset_root=Path("C:/tmp/generated"))
        with self.assertRaisesRegex(ValueError, DX12_BACKEND_DISABLED_REASON):
            executor._build_runtime_command(
                ProcessorTaskSlice(processor=inventory.processors[0], row_start=8, row_end=12),
                11,
                Path("C:/tmp/x.bin"),
                Path("C:/tmp/y.bin"),
            )

    def test_spatial_benchmark_spec_supports_dataclasses(self) -> None:
        spec = SpatialBenchmarkSpec(
            name="unit-test",
            h=8,
            w=8,
            c_in=4,
            c_out=8,
            k=3,
            pad=1,
            ideal_seconds=1.0,
            zero_score_seconds=10.0,
            stride=1,
        )

        self.assertEqual(spec.output_h, 8)
        self.assertEqual(spec.output_w, 8)
        self.assertEqual(spec.output_bytes, 8 * 8 * 8 * 4)

    def test_spatial_cuda_task_command_uses_batch_and_cooldown_flags(self) -> None:
        spec = spatial_executor.get_test_spec()
        expected_channels = 4
        expected_output_bytes = spec.output_h * spec.output_w * expected_channels * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "small_input.bin").write_bytes(b"\0")
            (temp_root / "large_input.bin").write_bytes(b"\0")
            fake_runner = temp_root / "spatial_cuda_runner.exe"
            fake_runner.write_text("runner placeholder\n", encoding="ascii")

            task = TaskAssign(
                request_id="req-1",
                node_id="node-1",
                task_id="task-1",
                method=METHOD_CONV2D,
                size="small",
                object_id="conv2d/small",
                stream_id="stream-1",
                timestamp_ms=0,
                iteration_count=3,
                start_oc=0,
                end_oc=expected_channels,
                tensor_h=spec.h,
                tensor_w=spec.w,
                channels_in=spec.c_in,
                channels_out=spec.c_out,
                kernel_size=spec.k,
                padding=spec.pad,
                stride=spec.stride,
                weight_data=b"\0" * (spec.k * spec.k * spec.c_in * expected_channels * 4),
            )

            captured: dict[str, list[str]] = {}

            def fake_run(command, **kwargs):
                captured["command"] = list(command)
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(b"\0" * expected_output_bytes)
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            executor = spatial_executor.Conv2dTaskExecutor(
                result_path=temp_root / "result.json",
                dataset_root=temp_root,
            )

            with (
                mock.patch.object(
                    spatial_executor,
                    "_best_backend_profile",
                    return_value=RuntimeProcessorProfile(
                        hardware_type="cuda",
                        effective_gflops=100.0,
                        rank=1,
                        best_config={"output_channel_batch": 8},
                    ),
                ),
                mock.patch.object(spatial_executor, "CONV2D_CUDA_COOLDOWN_MS", 1.5),
                mock.patch.object(spatial_executor, "_prepare_runner_path", return_value=fake_runner),
                mock.patch("compute_node.compute_methods.conv2d.executor.subprocess.run", side_effect=fake_run),
            ):
                executor.execute_task(task)

        command = captured["command"]
        self.assertIn("--output-channel-batch", command)
        self.assertEqual(command[command.index("--output-channel-batch") + 1], "4")
        self.assertIn("--cooldown-ms", command)
        self.assertEqual(command[command.index("--cooldown-ms") + 1], "1.5")
        # When the benchmarked best_config has no block/tile sizes, we must
        # NOT pin them — passing 0 would deactivate the autotune loop with a
        # bogus value. The runner falls back to its defaults.
        self.assertNotIn("--block-sizes", command)
        self.assertNotIn("--tile-sizes", command)
        self.assertNotIn("--shared-input", command)
        # Dispatch is the runner default — executor must not pass --mode.
        self.assertNotIn("--mode", command)

    def test_spatial_cuda_task_command_pins_block_and_tile_when_benchmarked(self) -> None:
        # Regression for the 2026-04-21 large-spec slowness: without --block-sizes
        # / --tile-sizes the runner sweeps 14 default tile candidates per task,
        # making CUDA ~30x slower than the standalone-measured throughput. When
        # the benchmark recorded a winning (block, tile) pair, pin it.
        spec = spatial_executor.get_test_spec()
        expected_channels = 4
        expected_output_bytes = spec.output_h * spec.output_w * expected_channels * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "small_input.bin").write_bytes(b"\0")
            (temp_root / "large_input.bin").write_bytes(b"\0")
            fake_runner = temp_root / "spatial_cuda_runner.exe"
            fake_runner.write_text("runner placeholder\n", encoding="ascii")

            task = TaskAssign(
                request_id="req-1",
                node_id="node-1",
                task_id="task-1",
                method=METHOD_CONV2D,
                size="small",
                object_id="conv2d/small",
                stream_id="stream-1",
                timestamp_ms=0,
                iteration_count=3,
                start_oc=0,
                end_oc=expected_channels,
                tensor_h=spec.h,
                tensor_w=spec.w,
                channels_in=spec.c_in,
                channels_out=spec.c_out,
                kernel_size=spec.k,
                padding=spec.pad,
                stride=spec.stride,
                weight_data=b"\0" * (spec.k * spec.k * spec.c_in * expected_channels * 4),
            )

            captured: dict[str, list[str]] = {}

            def fake_run(command, **kwargs):
                captured["command"] = list(command)
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(b"\0" * expected_output_bytes)
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            executor = spatial_executor.Conv2dTaskExecutor(
                result_path=temp_root / "result.json",
                dataset_root=temp_root,
            )

            with (
                mock.patch.object(
                    spatial_executor,
                    "_best_backend_profile",
                    return_value=RuntimeProcessorProfile(
                        hardware_type="cuda",
                        effective_gflops=200.0,
                        rank=1,
                        best_config={
                            "output_channel_batch": 8,
                            "block_size": 128,
                            "tile_size": 16,
                            "shared_input": 1,
                        },
                    ),
                ),
                mock.patch.object(spatial_executor, "_prepare_runner_path", return_value=fake_runner),
                mock.patch("compute_node.compute_methods.conv2d.executor.subprocess.run", side_effect=fake_run),
            ):
                executor.execute_task(task)

        command = captured["command"]
        self.assertIn("--block-sizes", command)
        self.assertEqual(command[command.index("--block-sizes") + 1], "128")
        self.assertIn("--tile-sizes", command)
        self.assertEqual(command[command.index("--tile-sizes") + 1], "16")
        self.assertIn("--shared-input", command)
        self.assertEqual(command[command.index("--shared-input") + 1], "1")

    def test_spatial_cpu_task_command_uses_benchmark_selected_workers(self) -> None:
        spec = spatial_executor.get_test_spec()
        expected_channels = 4
        expected_output_bytes = spec.output_h * spec.output_w * expected_channels * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "small_input.bin").write_bytes(b"\0")
            (temp_root / "large_input.bin").write_bytes(b"\0")
            fake_runner = temp_root / "spatial_cpu_runner.exe"
            fake_runner.write_text("runner placeholder\n", encoding="ascii")

            task = TaskAssign(
                request_id="req-1",
                node_id="node-1",
                task_id="task-1",
                method=METHOD_CONV2D,
                size="small",
                object_id="conv2d/small",
                stream_id="stream-1",
                timestamp_ms=0,
                iteration_count=3,
                start_oc=0,
                end_oc=expected_channels,
                tensor_h=spec.h,
                tensor_w=spec.w,
                channels_in=spec.c_in,
                channels_out=spec.c_out,
                kernel_size=spec.k,
                padding=spec.pad,
                stride=spec.stride,
                weight_data=b"\0" * (spec.k * spec.k * spec.c_in * expected_channels * 4),
            )

            captured: dict[str, list[str]] = {}

            def fake_run(command, **kwargs):
                captured["command"] = list(command)
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(b"\0" * expected_output_bytes)
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            executor = spatial_executor.Conv2dTaskExecutor(
                result_path=temp_root / "result.json",
                dataset_root=temp_root,
            )

            with (
                mock.patch.object(
                    spatial_executor,
                    "_best_backend_profile",
                    return_value=RuntimeProcessorProfile(
                        hardware_type="cpu",
                        effective_gflops=24.0,
                        rank=1,
                        best_config={"workers": 16, "requested_workers": 16},
                    ),
                ),
                mock.patch.object(spatial_executor, "_prepare_runner_path", return_value=fake_runner),
                mock.patch("compute_node.compute_methods.conv2d.executor.subprocess.run", side_effect=fake_run),
            ):
                executor.execute_task(task)

        command = captured["command"]
        self.assertIn("--workers", command)
        self.assertEqual(command[command.index("--workers") + 1], "16")

    def test_spatial_metal_task_command_passes_benchmark_config(self) -> None:
        spec = spatial_executor.get_test_spec()
        expected_channels = 5
        expected_output_bytes = spec.output_h * spec.output_w * expected_channels * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "small_input.bin").write_bytes(b"\0")
            (temp_root / "large_input.bin").write_bytes(b"\0")
            fake_runner = temp_root / "spatial_metal_runner"
            fake_runner.write_text("runner placeholder\n", encoding="ascii")

            task = TaskAssign(
                request_id="req-1",
                node_id="node-1",
                task_id="task-1",
                method=METHOD_CONV2D,
                size="small",
                object_id="conv2d/small",
                stream_id="stream-1",
                timestamp_ms=0,
                iteration_count=3,
                start_oc=0,
                end_oc=expected_channels,
                tensor_h=spec.h,
                tensor_w=spec.w,
                channels_in=spec.c_in,
                channels_out=spec.c_out,
                kernel_size=spec.k,
                padding=spec.pad,
                stride=spec.stride,
                weight_data=b"\0" * (spec.k * spec.k * spec.c_in * expected_channels * 4),
            )

            captured: dict[str, list[str]] = {}

            def fake_run(command, **kwargs):
                captured["command"] = list(command)
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(b"\0" * expected_output_bytes)
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            executor = spatial_executor.Conv2dTaskExecutor(
                result_path=temp_root / "result.json",
                dataset_root=temp_root,
            )

            with (
                mock.patch.object(
                    spatial_executor,
                    "_best_backend_profile",
                    return_value=RuntimeProcessorProfile(
                        hardware_type="metal",
                        effective_gflops=120.0,
                        rank=1,
                        best_config={
                            "block_size": 256,
                            "tile_size": 16,
                            "output_channel_batch": 3,
                        },
                    ),
                ),
                mock.patch.object(spatial_executor, "_prepare_runner_path", return_value=fake_runner),
                mock.patch("compute_node.compute_methods.conv2d.executor.subprocess.run", side_effect=fake_run),
            ):
                executor.execute_task(task)

        command = captured["command"]
        self.assertNotIn("--headroom-fraction", command)
        self.assertIn("--block-sizes", command)
        self.assertEqual(command[command.index("--block-sizes") + 1], "256")
        self.assertIn("--tile-sizes", command)
        self.assertEqual(command[command.index("--tile-sizes") + 1], "16")
        self.assertIn("--output-channel-batch", command)
        self.assertEqual(command[command.index("--output-channel-batch") + 1], "3")

    def test_spatial_prepare_runner_path_builds_metal_when_binary_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            missing_runner = temp_root / "missing_metal_runner"
            built_runner = temp_root / "built_metal_runner"

            with (
                mock.patch.object(spatial_executor, "_runner_path", return_value=missing_runner),
                mock.patch(
                    "compute_node.performance_metrics.conv2d.backends.metal_backend.MetalBackend._compile_if_needed",
                    return_value=(built_runner, "compiled"),
                ) as compile_mock,
            ):
                resolved = spatial_executor._prepare_runner_path("metal")

        self.assertEqual(resolved, built_runner)
        compile_mock.assert_called_once_with(force_rebuild=False)

    def test_spatial_artifact_transfer_returns_file_backed_result(self) -> None:
        spec = spatial_executor.get_test_spec()
        expected_channels = 4
        expected_output_bytes = spec.output_h * spec.output_w * expected_channels * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            (temp_root / "small_input.bin").write_bytes(b"\0")
            (temp_root / "large_input.bin").write_bytes(b"\0")
            fake_runner = temp_root / "spatial_cuda_runner.exe"
            fake_runner.write_text("runner placeholder\n", encoding="ascii")

            task = TaskAssign(
                request_id="req-1",
                node_id="node-1",
                task_id="task-1",
                method=METHOD_CONV2D,
                size="small",
                object_id="conv2d/small",
                stream_id="stream-1",
                timestamp_ms=0,
                iteration_count=3,
                transfer_mode=TransferMode.ARTIFACT_REQUIRED,
                start_oc=0,
                end_oc=expected_channels,
                tensor_h=spec.h,
                tensor_w=spec.w,
                channels_in=spec.c_in,
                channels_out=spec.c_out,
                kernel_size=spec.k,
                padding=spec.pad,
                stride=spec.stride,
                weight_data=b"\0" * (spec.k * spec.k * spec.c_in * expected_channels * 4),
            )

            def fake_run(command, **kwargs):
                del kwargs
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(b"\0" * expected_output_bytes)
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

            executor = spatial_executor.Conv2dTaskExecutor(
                result_path=temp_root / "result.json",
                dataset_root=temp_root,
            )

            with (
                mock.patch.object(
                    spatial_executor,
                    "_best_backend_profile",
                    return_value=RuntimeProcessorProfile(
                        hardware_type="cuda",
                        effective_gflops=100.0,
                        rank=1,
                        best_config={"output_channel_batch": 8},
                    ),
                ),
                mock.patch.object(spatial_executor, "_prepare_runner_path", return_value=fake_runner),
                mock.patch("compute_node.compute_methods.conv2d.executor.subprocess.run", side_effect=fake_run),
            ):
                result = executor.execute_task(task)

        self.assertEqual(result.output_vector, b"")
        self.assertTrue(result.local_result_path)
        result_path = Path(result.local_result_path)
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.stat().st_size, expected_output_bytes)
        result_path.unlink(missing_ok=True)


class GemmExecutorCommandTests(unittest.TestCase):
    """Verify the cuBLAS GEMM runner invocation matches its CLI contract."""

    def _build_task(self, *, size: str = "small", m: int = 1024, m_start: int = 0, m_end: int = 512, iteration_count: int = 3) -> TaskAssign:
        from core.constants import METHOD_GEMM
        from wire.internal_protocol.transport import GemmTaskPayload

        return TaskAssign(
            request_id="req-gemm",
            node_id="worker-1",
            task_id="req-gemm:worker-1",
            method=METHOD_GEMM,
            size=size,
            object_id=f"gemm/{size}",
            stream_id="",
            timestamp_ms=0,
            iteration_count=iteration_count,
            task_payload=GemmTaskPayload(
                m_start=m_start, m_end=m_end, m=m, n=m, k=m,
            ),
        )

    def test_gemm_command_uses_dispatch_mode_and_iteration_flag(self) -> None:
        from compute_node.compute_methods.gemm.executor import GemmTaskExecutor
        from compute_node.compute_methods.gemm.paths import CUDA_EXECUTABLE_PATH
        from compute_node.input_matrix.gemm import build_dataset_layout, build_spec, dataset_prefix_for_size

        dataset_root = Path("C:/tmp/gemm_generated")
        executor = GemmTaskExecutor(dataset_root=dataset_root)
        task = self._build_task(size="small")
        spec = build_spec(default_variant="small")
        layout = build_dataset_layout(dataset_root, prefix=dataset_prefix_for_size("small"))

        command = executor._build_runtime_command(
            task,
            spec=spec,
            dataset_layout=layout,
            output_path=dataset_root / ".task_C.bin",
        )

        self.assertEqual(command[0], str(CUDA_EXECUTABLE_PATH))
        self.assertIn("--input-a", command)
        self.assertIn("--input-b", command)
        self.assertIn("--m-start", command)
        self.assertEqual(command[command.index("--m-start") + 1], str(task.m_start))
        self.assertIn("--m-end", command)
        self.assertEqual(command[command.index("--m-end") + 1], str(task.m_end))
        self.assertEqual(command[command.index("--m") + 1], str(spec.m))
        self.assertEqual(command[command.index("--n") + 1], str(spec.n))
        self.assertEqual(command[command.index("--k") + 1], str(spec.k))
        self.assertEqual(
            command[command.index("--iteration-count") + 1],
            str(task.iteration_count),
        )
        self.assertEqual(command[command.index("--mode") + 1], "dispatch")


if __name__ == "__main__":
    unittest.main()
