"""Tests for the simplified performance-metrics workspace."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PERF_DIR = (
    Path(__file__).resolve().parents[1]
    / "compute_node"
    / "performance_metrics"
    / "gemv"
)

from compute_node.performance_metrics import benchmark
from compute_node.performance_metrics import result_format
from compute_node.performance_metrics.conv2d import benchmark as spatial_benchmark
from compute_node.performance_metrics.conv2d import backends as conv2d_backend_registry
from compute_node.performance_metrics.gemv import dataset_runner as gemv_dataset_runner
from compute_node.performance_metrics.gemv import backends as backend_registry
from compute_node.performance_metrics.gemv.backends.cpu_backend import (
    CpuArtifacts,
    CpuBackend,
    _binary_tree_worker_candidates,
    _candidate_tile_sizes as cpu_candidate_tile_sizes,
    _cpu_artifacts_for_platform,
)
from compute_node.performance_metrics.gemv.backends.cuda_backend import (
    CudaBackend,
    _candidate_block_sizes as cuda_candidate_block_sizes,
    _candidate_tile_sizes as cuda_candidate_tile_sizes,
    _candidate_transpose_modes as cuda_candidate_transpose_modes,
    _windows_gencode_args,
)
from compute_node.performance_metrics.conv2d.backends.cuda_backend import (
    CudaBackend as SpatialCudaBackend,
)
from compute_node.performance_metrics.conv2d.backends.cpu_backend import (
    CpuArtifacts as SpatialCpuArtifacts,
    CpuBackend as SpatialCpuBackend,
)
from compute_node.performance_metrics.conv2d.backends.metal_backend import (
    MetalBackend as SpatialMetalBackend,
)
from compute_node.performance_metrics.gemv.backends.metal_backend import (
    MetalBackend,
    _candidate_block_sizes as metal_candidate_block_sizes,
    _candidate_tile_sizes as metal_candidate_tile_sizes,
)
from compute_node.input_matrix import (
    build_dataset_layout,
    build_input_matrix_spec,
    compare_float32_vectors,
    dataset_is_generated,
    generate_dataset,
    load_float32_file,
)
from compute_node.input_matrix import generator as shared_input_generator
from compute_node.input_matrix import generate as input_matrix_generate_cli
from compute_node.input_matrix.gemv import generate as gemv_generate_cli
from compute_node.input_matrix.conv2d import (
    build_dataset_layout as build_conv_dataset_layout,
    build_input_matrix_spec as build_conv_input_matrix_spec,
    dataset_is_generated as conv_dataset_is_generated,
    generate_dataset as generate_conv_dataset,
    get_test_input_matrix_spec as get_conv_test_input_matrix_spec,
)
from compute_node.input_matrix.gemv import (
    get_test_input_matrix_spec as get_gemv_test_input_matrix_spec,
)
from compute_node.input_matrix.conv2d import generate as spatial_generate_cli
from compute_node.performance_metrics.gemv.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
)
from compute_node.performance_metrics.conv2d.models import (
    BenchmarkSpec as SpatialBenchmarkSpec,
)
from compute_node.performance_metrics.path_utils import to_relative_cli_path
from compute_node.performance_metrics.gemv.scoring import linear_time_score
from compute_node.performance_metrics.gemv.workloads import build_benchmark_spec
import subprocess

from core.constants import DX12_BACKEND_DISABLED_REASON, METHOD_GEMV, METHOD_CONV2D
from tests.support import require_integration


class _FakeNativeRunnerProcess:
    """Minimal duck-type of ``subprocess.Popen`` for backend unit tests.

    The conv2d Python backends spawn the native runner via ``subprocess.Popen``
    and drain ``.stdout`` / ``.stderr`` on dedicated pump threads before
    calling ``.wait(timeout=...)``. This stub yields a single empty-JSON line
    on stdout so ``json.loads`` succeeds, no stderr, and returns 0 from wait.
    """

    def __init__(self, stdout_text: str = "{}\n", stderr_text: str = "", return_code: int = 0) -> None:
        self.stdout = iter([stdout_text]) if stdout_text else iter([])
        self.stderr = iter([stderr_text]) if stderr_text else iter([])
        self._return_code = return_code

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return self._return_code

    def kill(self) -> None:
        pass


class PerformanceMatricsTests(unittest.TestCase):
    def test_linear_score_midpoint_is_half(self) -> None:
        score = linear_time_score(0.6, ideal_seconds=0.2, zero_score_seconds=1.0, max_score=100.0)
        self.assertAlmostEqual(score, 50.0)

    def test_worker_binary_tree_sequence(self) -> None:
        self.assertEqual(_binary_tree_worker_candidates(16), [16, 8, 32, 4, 64])

    def test_benchmark_parser_accepts_rebuild_flag(self) -> None:
        args = benchmark.build_parser().parse_args(["--rebuild"])
        self.assertTrue(args.rebuild)

    def test_benchmark_parser_accepts_accumulation_precision(self) -> None:
        args = benchmark.build_parser().parse_args(["--accumulation-precision", "fp64_accumulate"])
        self.assertEqual(args.accumulation_precision, "fp64_accumulate")

    def test_benchmark_parser_defaults_to_all_methods(self) -> None:
        args = benchmark.build_parser().parse_args([])
        self.assertEqual(args.method, "all")

    def test_benchmark_parser_accepts_workload_mode(self) -> None:
        args = benchmark.build_parser().parse_args(["--workload-mode", "large"])
        self.assertEqual(args.workload_mode, "large")

    def test_spatial_benchmark_parser_accepts_cuda_channel_batch_flags(self) -> None:
        args = spatial_benchmark.build_parser().parse_args(
            [
                "--workload-mode",
                "small",
                "--output-channel-batch",
                "8",
                "--cooldown-ms",
                "2.5",
            ]
        )
        self.assertEqual(args.workload_mode, "small")
        self.assertEqual(args.output_channel_batch, 8)
        self.assertEqual(args.cooldown_ms, 2.5)

    def test_spatial_benchmark_parser_defaults_to_full_workload(self) -> None:
        args = spatial_benchmark.build_parser().parse_args([])
        self.assertEqual(args.workload_mode, "full")

    def test_gemv_cpu_hardware_label_prefers_friendly_detected_cpu_name(self) -> None:
        with mock.patch.object(
            benchmark.gemv_runner,
            "detect_cpu_name",
            return_value="AMD Ryzen 9 8945HS w/ Radeon 780M Graphics",
        ):
            label = benchmark.gemv_runner._hardware_label_for_backend("cpu", "")

        self.assertEqual(label, "AMD Ryzen 9 8945HS w/ Radeon 780M Graphics")

    def test_conv2d_cpu_hardware_label_prefers_friendly_detected_cpu_name(self) -> None:
        with mock.patch.object(
            spatial_benchmark,
            "detect_cpu_name",
            return_value="AMD Ryzen 9 8945HS w/ Radeon 780M Graphics",
        ):
            label = spatial_benchmark._hardware_label_for_backend("cpu", "", None)

        self.assertEqual(label, "AMD Ryzen 9 8945HS w/ Radeon 780M Graphics")

    def test_top_level_benchmark_parser_leaves_workload_mode_unset_by_default(self) -> None:
        args = benchmark.build_parser().parse_args([])
        self.assertIsNone(args.workload_mode)

    def test_input_matrix_cli_defaults_to_all_methods(self) -> None:
        args = input_matrix_generate_cli.build_parser().parse_args([])
        self.assertEqual(args.method, "all")
        self.assertEqual(args.workers, max(1, os.cpu_count() or 1))
        self.assertEqual(args.chunk_mib, 8)

    def test_method_local_generate_parsers_use_cpu_count_and_8mib_chunks(self) -> None:
        gemv_args = gemv_generate_cli.build_parser().parse_args([])
        spatial_args = spatial_generate_cli.build_parser().parse_args([])

        expected_workers = max(1, os.cpu_count() or 1)
        self.assertEqual(gemv_args.workers, expected_workers)
        self.assertEqual(spatial_args.workers, expected_workers)
        self.assertEqual(gemv_args.chunk_mib, 8)
        self.assertEqual(spatial_args.chunk_mib, 8)

    def test_normalize_method_report_adds_structured_cuda_environment(self) -> None:
        raw_method = {
            "generated_at_unix": 123.0,
            "benchmark_elapsed_seconds": 1.5,
            "workload": {
                "autotune": {"name": "test-conv2d"},
                "measurement": {"name": "runtime-conv2d"},
                "autotune_repeats": 3,
                "measurement_repeats": 1,
                "full_runtime_measurement": True,
            },
            "hardware_inventory": {
                "cuda": {
                    "probe_message": "CUDA backend available for 'NVIDIA GeForce RTX 4060 Laptop GPU'. Existing binary is older than the source, so it will be rebuilt.",
                }
            },
            "backends": {
                "cuda": {
                    "available": True,
                    "rank": 1,
                    "best_config": {"block_size": 256, "tile_size": 16, "trials_run": 4},
                    "autotune_result": {
                        "wall_clock_latency_seconds": 0.1,
                        "effective_gflops": 100.0,
                        "checksum": "chk_a",
                        "score": 1000.0,
                    },
                    "best_result": {
                        "wall_clock_latency_seconds": 0.2,
                        "effective_gflops": 200.0,
                        "checksum": "chk_b",
                        "score": 1000.0,
                    },
                    "notes": [
                        "compiled CUDA runner from compute_node/compute_methods/conv2d/cuda/conv2d_cuda_runner.cu; fatbin SMs: sm_75, sm_89",
                        "Autotuned on test-conv2d and measured on runtime-conv2d.",
                    ],
                    "trial_notes": ["device=NVIDIA GeForce RTX 4060 Laptop GPU"],
                }
            },
            "backends_considered": ["cuda"],
            "detected_backends": ["cuda"],
            "usable_backends": ["cuda"],
            "ranking": ["cuda"],
            "best_backend": "cuda",
        }

        with (
            mock.patch.object(
                result_format,
                "_detect_cuda_gpu_inventory",
                return_value=[
                    {
                        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                        "sm_digits": "89",
                        "driver_version": "576.52",
                    }
                ],
            ),
            mock.patch.object(result_format, "_detect_nvcc_version", return_value="13.2"),
        ):
            normalized = result_format.normalize_method_report(
                method_name=METHOD_CONV2D,
                raw_method=raw_method,
                dataset_root="compute_node/input_matrix/conv2d/generated",
                device_overview={},
            )

        cuda_backend = normalized["backends"]["cuda"]
        self.assertEqual(
            cuda_backend["cuda_environment"],
            {
                "detected_architecture": "Ada",
                "sm_version": "sm_89",
                "driver_version": "576.52",
                "nvcc_version": "13.2",
                "binary_status": "recompiled",
            },
        )
        self.assertEqual(
            cuda_backend["notes"],
            ["binary recompiled", "small autotune, large measurement"],
        )

    def test_compact_note_tracks_small_to_mid_measurement_flow(self) -> None:
        compact = result_format._compact_note(
            "Autotuned on small-conv2d-256x256 and measured on mid-conv2d-512x512."
        )

        self.assertEqual(compact, "small autotune, mid measurement")

    def test_write_float32_file_parallelizes_even_when_one_chunk_would_cover_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            shared_input_generator,
            "_write_float32_file_parallel",
            return_value="parallel-sha256",
        ) as parallel_writer:
            output_path = Path(temp_dir) / "test.bin"
            sha256_hex = shared_input_generator.write_float32_file(
                output_path,
                total_values=32,
                seed=123,
                chunk_values=1024,
                label="test.bin",
                worker_count=2,
            )

        self.assertEqual(sha256_hex, "parallel-sha256")
        parallel_writer.assert_called_once()

    def test_windows_default_backend_order_pairs_cpu_with_detected_gpu(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.gemv.backends.os.name",
                "nt",
            ),
            mock.patch(
                "compute_node.performance_metrics.gemv.backends.detect_nvidia_windows_adapter",
                return_value=("NVIDIA GeForce RTX 4060 Laptop GPU", ""),
            ),
        ):
            names = [backend.name for backend in backend_registry.build_backends()]

        self.assertEqual(names, ["cpu", "cuda"])

    def test_spatial_windows_default_backend_order_pairs_cpu_with_detected_gpu(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.os.name",
                "nt",
            ),
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.detect_nvidia_windows_adapter",
                return_value=("NVIDIA GeForce RTX 4060 Laptop GPU", ""),
            ),
        ):
            names = [backend.name for backend in conv2d_backend_registry.build_backends()]

        self.assertEqual(names, ["cpu", "cuda"])

    def test_windows_default_backend_order_falls_back_to_cpu_without_gpu(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.gemv.backends.os.name",
                "nt",
            ),
            mock.patch(
                "compute_node.performance_metrics.gemv.backends.detect_nvidia_windows_adapter",
                return_value=(None, "no adapter"),
            ),
        ):
            names = [backend.name for backend in backend_registry.build_backends()]

        self.assertEqual(names, ["cpu"])

    def test_spatial_windows_default_backend_order_falls_back_to_cpu_without_gpu(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.os.name",
                "nt",
            ),
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.detect_nvidia_windows_adapter",
                return_value=(None, "no adapter"),
            ),
        ):
            names = [backend.name for backend in conv2d_backend_registry.build_backends()]

        self.assertEqual(names, ["cpu"])

    def test_macos_default_backend_order_keeps_cpu_with_metal(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.gemv.backends.os.name",
                "posix",
            ),
            mock.patch(
                "compute_node.performance_metrics.gemv.backends.sys.platform",
                "darwin",
            ),
        ):
            names = [backend.name for backend in backend_registry.build_backends()]

        self.assertEqual(names, ["cpu", "metal"])

    def test_spatial_macos_default_backend_order_keeps_cpu_with_metal(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.os.name",
                "posix",
            ),
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.sys.platform",
                "darwin",
            ),
        ):
            names = [backend.name for backend in conv2d_backend_registry.build_backends()]

        self.assertEqual(names, ["cpu", "metal"])

    def test_top_level_conv2d_dispatch_defaults_to_full_workload(self) -> None:
        args = argparse.Namespace(
            workload_mode=None,
            role="compute",
            backend=None,
            rebuild=False,
            h=None,
            w=None,
            cin=None,
            cout=None,
            k=None,
            pad=None,
            stride=None,
        )

        captured_command: list[str] = []

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = iter(())

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                return 0

        def fake_popen(command, **kwargs):
            del kwargs
            captured_command[:] = list(command)
            return _FakeProcess()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(benchmark, "active_python_path", return_value=Path(sys.executable)),
            mock.patch.object(benchmark, "emit_status"),
            mock.patch.object(benchmark.tempfile, "TemporaryDirectory", return_value=tempfile.TemporaryDirectory(dir=temp_dir)),
            mock.patch.object(benchmark.subprocess, "Popen", side_effect=fake_popen),
            mock.patch.object(benchmark, "METHOD_DATASET_DIRS", {**benchmark.METHOD_DATASET_DIRS, METHOD_CONV2D: Path(temp_dir)}),
            mock.patch.object(Path, "read_text", return_value=json.dumps({"usable_backends": [], "ranking": []})),
        ):
            benchmark._run_conv2d_benchmark(args)

        self.assertIn("--workload-mode", captured_command)
        self.assertEqual(captured_command[captured_command.index("--workload-mode") + 1], "full")

    def test_conv2d_small_only_timeout_floor_is_extended(self) -> None:
        timeout_seconds = spatial_benchmark._backend_timeout_seconds(
            "small",
            "small",
            spatial_benchmark.get_small_spec(),
        )

        self.assertEqual(timeout_seconds, 300.0)

    def test_conv2d_small_to_mid_timeout_floor_is_extended(self) -> None:
        timeout_seconds = spatial_benchmark._backend_timeout_seconds(
            "small",
            "mid",
            spatial_benchmark.get_mid_spec(),
        )

        self.assertEqual(timeout_seconds, 360.0)

    def test_gemv_measurement_phase_title_uses_full_run_for_large_measurement(self) -> None:
        title = benchmark.gemv_runner._measurement_phase_title("small", "large")

        self.assertEqual(title, "Full run stage")

    def test_conv2d_measurement_phase_title_uses_final_measurement_for_same_dataset(self) -> None:
        title = spatial_benchmark._measurement_phase_title("small", "small")

        self.assertEqual(title, "Final measurement stage")

    def test_conv2d_measurement_phase_title_uses_full_run_for_mid_measurement(self) -> None:
        title = spatial_benchmark._measurement_phase_title("small", "mid")

        self.assertEqual(title, "Full run stage")

    def test_conv2d_default_workload_plan_uses_small_autotune_and_mid_measurement(self) -> None:
        args = argparse.Namespace(
            workload_mode="full",
            h=None,
            w=None,
            cin=None,
            cout=None,
            k=None,
            pad=None,
            stride=None,
        )

        (
            workload_mode,
            autotune_variant,
            measurement_variant,
            autotune_spec,
            measurement_spec,
        ) = spatial_benchmark._resolve_workload_plan(args)

        self.assertEqual(workload_mode, "full")
        self.assertEqual(autotune_variant, "small")
        self.assertEqual(measurement_variant, "mid")
        self.assertEqual(autotune_spec.name, spatial_benchmark.get_small_spec().name)
        self.assertEqual(measurement_spec.name, spatial_benchmark.get_mid_spec().name)

    def test_conv2d_selected_config_format_is_compact(self) -> None:
        formatted = spatial_benchmark._format_selected_config(
            {
                "transpose": False,
                "block_size": 256,
                "tile_size": 16,
                "output_channel_batch": 128,
            }
        )

        self.assertEqual(
            formatted,
            "transpose=false block_size=256 tile_size=16 output_channel_batch=128",
        )

    def test_gemv_selected_config_format_is_compact(self) -> None:
        formatted = benchmark.gemv_runner._format_selected_config(
            {
                "transpose": False,
                "block_size": 128,
                "tile_size": 4,
            }
        )

        self.assertEqual(
            formatted,
            "transpose=false block_size=128 tile_size=4",
        )

    def test_conv2d_phase_callback_prints_final_measurement_separately(self) -> None:
        args = argparse.Namespace(
            backend=None,
            dataset_dir=Path("C:/tmp/generated"),
            output=Path("C:/tmp/result.json"),
            role="compute",
            h=None,
            w=None,
            cin=None,
            cout=None,
            k=None,
            pad=None,
            stride=None,
            workload_mode="small",
            output_channel_batch=None,
            cooldown_ms=None,
            rebuild=False,
        )

        class _FakeBackend:
            name = "cuda"

            def diagnostic_context(self, _spec=None):
                return {"device_name": "Fake GPU"}

            def probe(self):
                return True, "CUDA backend available for 'Fake GPU'."

            def run(
                self,
                spec,
                dataset,
                *,
                measurement_spec=None,
                measurement_dataset=None,
                time_budget_seconds,
                force_rebuild=False,
                phase_callback=None,
                verbose=False,
            ):
                del spec, dataset, measurement_spec, measurement_dataset, time_budget_seconds, force_rebuild, verbose
                if callable(phase_callback):
                    phase_callback("final_measurement", {"block_size": 256, "tile_size": 16})
                return {
                    "available": True,
                    "best_trial": {"effective_gflops": 1.0},
                    "notes": [],
                }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(spatial_benchmark, "build_backends", return_value=[_FakeBackend()]),
            mock.patch.object(spatial_benchmark, "_generate_if_needed"),
            mock.patch.object(spatial_benchmark, "METHOD_DATASET_DIR", Path(temp_dir)),
            mock.patch("builtins.print") as print_mock,
        ):
            args.dataset_dir = Path(temp_dir)
            spatial_benchmark.run_benchmark(args)

        printed_lines = [" ".join(str(part) for part in call.args) for call in print_mock.call_args_list]
        autotune_index = next(
            index for index, line in enumerate(printed_lines) if "Autotune stage on Fake GPU:" in line
        )
        measurement_index = next(
            index for index, line in enumerate(printed_lines) if "Final measurement stage on Fake GPU:" in line
        )
        self.assertLess(autotune_index, measurement_index)

    def test_cpu_artifacts_follow_platform(self) -> None:
        windows_artifacts = _cpu_artifacts_for_platform("win32")
        assert windows_artifacts is not None
        self.assertEqual(windows_artifacts.platform_key, "windows")
        self.assertEqual(windows_artifacts.executable_path.name, "gemv_cpu_windows.exe")

        macos_artifacts = _cpu_artifacts_for_platform("darwin")
        assert macos_artifacts is not None
        self.assertEqual(macos_artifacts.platform_key, "macos")
        self.assertEqual(macos_artifacts.executable_path.name, "gemv_cpu_macos")

        self.assertIsNone(_cpu_artifacts_for_platform("linux"))

    def test_cpu_backend_prefers_existing_binary_before_compiling(self) -> None:
        backend = CpuBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts = CpuArtifacts(
                platform_key="macos",
                platform_label="macOS",
                source_path=temp_root / "gemv_cpu_macos.cpp",
                build_dir=temp_root / "build",
                executable_path=temp_root / "build" / "gemv_cpu_macos",
            )
            artifacts.build_dir.mkdir(parents=True, exist_ok=True)
            artifacts.source_path.write_text("// source placeholder\n", encoding="utf-8")
            artifacts.executable_path.write_text("binary placeholder\n", encoding="utf-8")

            with mock.patch.object(backend, "_compile_macos_runner", side_effect=AssertionError("should not compile")):
                executable_path, note = backend._resolve_executable_path(artifacts)

        self.assertEqual(executable_path, artifacts.executable_path)
        self.assertIn("using prebuilt", note)

    def test_cpu_backend_force_rebuild_compiles_even_with_existing_binary(self) -> None:
        backend = CpuBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts = CpuArtifacts(
                platform_key="macos",
                platform_label="macOS",
                source_path=temp_root / "gemv_cpu_macos.cpp",
                build_dir=temp_root / "build",
                executable_path=temp_root / "build" / "gemv_cpu_macos",
            )
            artifacts.build_dir.mkdir(parents=True, exist_ok=True)
            artifacts.source_path.write_text("// source placeholder\n", encoding="utf-8")
            artifacts.executable_path.write_text("binary placeholder\n", encoding="utf-8")

            with (
                mock.patch.object(backend, "_can_build_for_artifacts", return_value=(True, "")),
                mock.patch.object(backend, "_compile_macos_runner") as compile_mock,
            ):
                executable_path, note = backend._resolve_executable_path(artifacts, force_rebuild=True)

        self.assertEqual(executable_path, artifacts.executable_path)
        compile_mock.assert_called_once_with(artifacts)
        self.assertIn("rebuild was explicitly requested", note)

    def test_spatial_cpu_backend_compiles_missing_macos_binary(self) -> None:
        backend = SpatialCpuBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts = SpatialCpuArtifacts(
                platform_key="macos",
                platform_label="macOS",
                source_path=temp_root / "conv2d_cpu_macos.cpp",
                build_dir=temp_root / "build",
                executable_path=temp_root / "build" / "conv2d_cpu_macos",
            )
            artifacts.build_dir.mkdir(parents=True, exist_ok=True)
            artifacts.source_path.write_text("// source placeholder\n", encoding="utf-8")

            def fake_compile(_artifacts) -> None:
                artifacts.executable_path.write_text("binary placeholder\n", encoding="utf-8")

            with mock.patch.object(backend, "_compile_macos_runner", side_effect=fake_compile) as compile_mock:
                executable_path, note = backend._resolve_executable_path(artifacts)

        self.assertEqual(executable_path, artifacts.executable_path)
        compile_mock.assert_called_once_with(artifacts)
        self.assertIn("binary is missing", note)

    def test_spatial_metal_backend_prefers_existing_binary_before_compiling(self) -> None:
        backend = SpatialMetalBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            binary_path = temp_root / "conv2d_metal_runner"
            host_source = temp_root / "conv2d_metal_runner.mm"
            kernel_source = temp_root / "conv2d_metal_kernels.metal"
            host_source.write_text("// host placeholder\n", encoding="utf-8")
            kernel_source.write_text("// kernel placeholder\n", encoding="utf-8")
            binary_path.write_text("binary placeholder\n", encoding="utf-8")
            os.utime(binary_path, None)

            with (
                mock.patch(
                    "compute_node.performance_metrics.conv2d.backends.metal_backend.METAL_EXECUTABLE_PATH",
                    binary_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.conv2d.backends.metal_backend.METAL_HOST_SOURCE_PATH",
                    host_source,
                ),
                mock.patch(
                    "compute_node.performance_metrics.conv2d.backends.metal_backend.METAL_KERNEL_SOURCE_PATH",
                    kernel_source,
                ),
                mock.patch(
                    "compute_node.performance_metrics.conv2d.backends.metal_backend._binary_is_stale",
                    return_value=False,
                ),
                mock.patch.object(
                    backend,
                    "_toolchain_status",
                    side_effect=AssertionError("should not probe toolchain"),
                ),
            ):
                executable_path, note = backend._compile_if_needed()

        self.assertEqual(executable_path, binary_path)
        self.assertIn("using prebuilt", note)

    def test_default_spec_matches_requested_2gib_shape(self) -> None:
        spec = build_benchmark_spec()
        self.assertEqual(spec.rows, 16_384)
        self.assertEqual(spec.cols, 32_768)
        self.assertEqual(spec.matrix_bytes, 2 * 1024**3)
        self.assertEqual(spec.vector_bytes, 32_768 * 4)

    def test_small_default_workloads_are_reduced(self) -> None:
        gemv_spec = get_gemv_test_input_matrix_spec()
        conv_spec = get_conv_test_input_matrix_spec()

        self.assertEqual((gemv_spec.rows, gemv_spec.cols), (2_048, 4_096))
        self.assertEqual((conv_spec.h, conv_spec.w), (256, 256))
        self.assertEqual((conv_spec.c_in, conv_spec.c_out), (32, 64))

    def test_generate_dataset_with_small_override(self) -> None:
        spec = build_input_matrix_spec(rows=8, cols=16)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            self.assertTrue(dataset_is_generated(layout, spec))
            self.assertEqual(layout.matrix_path.name, "A.bin")
            self.assertEqual(layout.vector_path.name, "x.bin")
            self.assertEqual(layout.matrix_path.stat().st_size, spec.matrix_bytes)
            self.assertEqual(layout.vector_path.stat().st_size, spec.vector_bytes)

    def test_parallel_generate_dataset_matches_single_process_bytes(self) -> None:
        spec = build_input_matrix_spec(rows=64, cols=64)
        with tempfile.TemporaryDirectory() as sequential_dir, tempfile.TemporaryDirectory() as parallel_dir:
            sequential_layout = build_dataset_layout(Path(sequential_dir))
            parallel_layout = build_dataset_layout(Path(parallel_dir))

            generate_dataset(sequential_layout, spec, generator_workers=1, chunk_values=16)
            generate_dataset(parallel_layout, spec, generator_workers=2, chunk_values=16)

            self.assertEqual(
                sequential_layout.matrix_path.read_bytes(),
                parallel_layout.matrix_path.read_bytes(),
            )
            self.assertEqual(
                sequential_layout.vector_path.read_bytes(),
                parallel_layout.vector_path.read_bytes(),
            )
            self.assertEqual(
                json.loads(sequential_layout.meta_path.read_text(encoding="utf-8"))["files"]["matrix"]["sha256"],
                json.loads(parallel_layout.meta_path.read_text(encoding="utf-8"))["files"]["matrix"]["sha256"],
            )

    def test_dataset_is_generated_rejects_mismatched_metadata(self) -> None:
        spec = build_input_matrix_spec(rows=8, cols=16)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
            metadata["dataset"]["rows"] = 9
            layout.meta_path.write_text(json.dumps(metadata), encoding="utf-8")

            self.assertFalse(dataset_is_generated(layout, spec))

    def test_conv2d_dataset_generation_supports_small_override(self) -> None:
        spec = build_conv_input_matrix_spec(h=16, w=16, c_in=4, c_out=8, k=3, pad=1, stride=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_conv_dataset_layout(Path(temp_dir), prefix="test_")
            generate_conv_dataset(layout, spec, generator_workers=2, chunk_values=16)
            self.assertTrue(conv_dataset_is_generated(layout, spec))
            self.assertEqual(layout.input_path.stat().st_size, spec.input_bytes)
            self.assertEqual(layout.weight_path.stat().st_size, spec.weight_bytes)

    def test_normalize_conv2d_dataset_uses_runtime_artifacts_for_large_only_mode(self) -> None:
        raw_method = {
            "workload": {
                "autotune_dataset_variant": "large",
                "measurement_dataset_variant": "large",
                "full_runtime_measurement": False,
            }
        }

        normalized = result_format.normalize_method_report(
            method_name=METHOD_CONV2D,
            raw_method=raw_method,
            dataset_root="compute_node/input_matrix/conv2d/generated",
            device_overview={},
        )

        dataset = normalized["dataset"]
        self.assertEqual(
            dataset["artifacts"]["autotune_input"],
            "compute_node/input_matrix/conv2d/generated/large_input.bin",
        )
        self.assertEqual(
            dataset["artifacts"]["autotune_weight"],
            "compute_node/input_matrix/conv2d/generated/large_weight.bin",
        )
        self.assertEqual(
            dataset["artifacts"]["measurement_input"],
            "compute_node/input_matrix/conv2d/generated/large_input.bin",
        )
        self.assertEqual(
            dataset["artifacts"]["measurement_weight"],
            "compute_node/input_matrix/conv2d/generated/large_weight.bin",
        )

    @require_integration("Dataset override flow is validated through a real benchmark run.")
    def test_small_override_uses_override_dataset_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            default_dataset_dir = temp_root / "generated"
            args = argparse.Namespace(
                method=METHOD_GEMV,
                backend=["cpu"],
                dataset_dir=default_dataset_dir,
                output=temp_root / "result.json",
                time_budget=30.0,
                rebuild=False,
                accumulation_precision="fp32",
                rows=8,
                cols=16,
                role="compute",
                h=None,
                w=None,
                cin=None,
                cout=None,
                k=None,
                pad=None,
                stride=None,
            )
            with mock.patch.object(benchmark, "DEFAULT_DATASET_DIR", default_dataset_dir):
                report = benchmark.run_benchmark(args)

        method_report = report["methods"][METHOD_GEMV]
        dataset_root = Path(method_report["dataset"]["root_dir"])
        self.assertEqual(dataset_root.parts[-3:], ("generated", "overrides", "8x16"))

    @require_integration("Real benchmark execution is reserved for integration runs.")
    def test_benchmark_auto_generates_and_runs_cpu(self) -> None:
        backend = CpuBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("CPU backend is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "compute_node.performance_metrics.gemv.backends.cpu_backend.os.cpu_count",
            return_value=1,
        ):
            args = argparse.Namespace(
                method=METHOD_GEMV,
                backend=["cpu"],
                dataset_dir=Path(temp_dir),
                output=Path(temp_dir) / "result.json",
                time_budget=30.0,
                rebuild=False,
                accumulation_precision="fp32",
                rows=8,
                cols=16,
                role="compute",
                h=None,
                w=None,
                cin=None,
                cout=None,
                k=None,
                pad=None,
                stride=None,
            )
            report = benchmark.run_benchmark(args)

        self.assertEqual(report["schema_version"], 6)
        self.assertIn("device_overview", report)
        self.assertIn("methods", report)
        method_report = report["methods"][METHOD_GEMV]
        self.assertEqual(method_report["method"], METHOD_GEMV)
        self.assertEqual(method_report["best_backend"], "cpu")
        self.assertEqual(method_report["ranking"], ["cpu"])
        self.assertIn("workload", method_report)
        self.assertIn("dataset", method_report)
        self.assertIn("detected_backends", method_report)
        self.assertIn("usable_backends", method_report)
        self.assertIn("backends", method_report)
        assert isinstance(method_report["backends"], dict)
        self.assertFalse(Path(method_report["dataset"]["root_dir"]).is_absolute())
        self.assertFalse(Path(method_report["dataset"]["artifacts"]["autotune_matrix"]).is_absolute())
        self.assertFalse(Path(method_report["dataset"]["artifacts"]["autotune_vector"]).is_absolute())
        self.assertFalse(Path(method_report["dataset"]["artifacts"]["measurement_matrix"]).is_absolute())
        self.assertFalse(Path(method_report["dataset"]["artifacts"]["measurement_vector"]).is_absolute())
        self.assertIn("cpu", method_report["backends"])
        self.assertIn("cpu", method_report["detected_backends"])
        self.assertIn("cpu", method_report["usable_backends"])
        self.assertEqual(
            method_report["workload"]["autotune_plan"]["autotune_repeats"],
            DEFAULT_AUTOTUNE_REPEATS,
        )
        self.assertEqual(
            method_report["workload"]["measurement_plan"]["measurement_repeats"],
            DEFAULT_MEASUREMENT_REPEATS,
        )
        self.assertEqual(method_report["workload"]["autotune_plan"]["accumulation_precision"], "fp32")
        self.assertEqual(method_report["workload"]["autotune_plan"]["input_dtype"], "fp32")
        self.assertEqual(method_report["workload"]["autotune_plan"]["output_dtype"], "fp32")
        cpu_result = method_report["backends"]["cpu"]
        self.assertTrue(cpu_result["available"])
        self.assertEqual(cpu_result["rank"], 1)
        self.assertIsNotNone(cpu_result["best_config"])
        self.assertIsNotNone(cpu_result["autotune_result"])
        self.assertIsNotNone(cpu_result["best_result"])
        assert cpu_result["autotune_result"] is not None
        assert cpu_result["best_result"] is not None
        assert cpu_result["best_config"] is not None
        drive_pattern = re.compile(r"[A-Za-z]:[\\/]")
        self.assertFalse(any(drive_pattern.search(note or "") for note in cpu_result["notes"]))
        self.assertGreater(int(cpu_result["best_config"]["workers"]), 0)
        self.assertGreater(int(cpu_result["best_config"]["tile_size"]), 0)
        self.assertEqual(str(cpu_result["best_config"]["accumulation_precision"]), "fp32")
        self.assertEqual(int(cpu_result["best_config"]["autotune_repeats"]), DEFAULT_AUTOTUNE_REPEATS)
        self.assertEqual(int(cpu_result["best_config"]["measurement_repeats"]), DEFAULT_MEASUREMENT_REPEATS)
        self.assertTrue(str(cpu_result["autotune_result"]["checksum"]).startswith("fnv1a64:"))
        self.assertTrue(str(cpu_result["best_result"]["checksum"]).startswith("fnv1a64:"))
        self.assertGreater(float(cpu_result["best_result"]["effective_gflops"]), 0.0)
        self.assertIn("checksum", cpu_result["best_result"])
        self.assertIn("effective_gflops", cpu_result["best_result"])
        self.assertTrue(str(cpu_result["device_name"]))

    def test_cuda_backend_probe_returns_status(self) -> None:
        backend = CudaBackend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)

    def test_windows_cuda_fat_binary_targets_common_sms(self) -> None:
        args = _windows_gencode_args("89")
        self.assertIn("-gencode=arch=compute_75,code=sm_75", args)
        self.assertIn("-gencode=arch=compute_80,code=sm_80", args)
        self.assertIn("-gencode=arch=compute_86,code=sm_86", args)
        self.assertIn("-gencode=arch=compute_89,code=sm_89", args)
        self.assertIn("-gencode=arch=compute_90,code=sm_90", args)
        self.assertIn("-gencode=arch=compute_120,code=sm_120", args)

    def test_gemv_dx12_backend_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, re.escape(DX12_BACKEND_DISABLED_REASON)):
            backend_registry.build_backends(["dx12"])

    def test_cuda_backend_probe_accepts_prebuilt_windows_runner_without_nvcc(self) -> None:
        backend = CudaBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            binary_path = temp_root / "gemv_cuda_runner.exe"
            source_path = temp_root / "gemv_cuda_runner.cu"
            binary_path.write_text("binary placeholder\n", encoding="utf-8")
            source_path.write_text("// source placeholder\n", encoding="utf-8")

            with (
                mock.patch(
                    "compute_node.performance_metrics.gemv.backends.cuda_backend.os.name",
                    "nt",
                ),
                mock.patch(
                    "compute_node.performance_metrics.gemv.backends.cuda_backend.CUDA_EXECUTABLE_PATH",
                    binary_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.gemv.backends.cuda_backend.CUDA_SOURCE_PATH",
                    source_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.gemv.backends.cuda_backend._detect_compute_capability",
                    return_value="89",
                ),
                mock.patch.object(backend, "_toolchain_status", return_value=(False, "nvcc missing")),
            ):
                available, message = backend.probe()

        self.assertTrue(available)
        self.assertIn("self-contained Windows runner", message)

    def test_cuda_backend_force_rebuild_requires_toolchain(self) -> None:
        backend = CudaBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            binary_path = temp_root / "gemv_cuda_runner.exe"
            source_path = temp_root / "gemv_cuda_runner.cu"
            binary_path.write_text("binary placeholder\n", encoding="utf-8")
            source_path.write_text("// source placeholder\n", encoding="utf-8")

            with (
                mock.patch(
                    "compute_node.performance_metrics.gemv.backends.cuda_backend.CUDA_EXECUTABLE_PATH",
                    binary_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.gemv.backends.cuda_backend.CUDA_SOURCE_PATH",
                    source_path,
                ),
                mock.patch.object(backend, "_toolchain_status", return_value=(False, "nvcc missing")),
            ):
                with self.assertRaises(FileNotFoundError):
                    backend._resolve_executable_path(force_rebuild=True)

    def test_spatial_dx12_backend_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, re.escape(DX12_BACKEND_DISABLED_REASON)):
            conv2d_backend_registry.build_backends(["dx12"])

    def test_spatial_cuda_runner_command_includes_batch_candidates_and_cooldown_flags(self) -> None:
        backend = SpatialCudaBackend()
        spec = SpatialBenchmarkSpec(
            name="unit-test",
            h=8,
            w=8,
            c_in=4,
            c_out=8,
            k=3,
            pad=1,
            ideal_seconds=1.0,
            zero_score_seconds=5.0,
            stride=1,
        )
        layout = build_conv_dataset_layout(Path("C:/tmp/generated"), prefix="test_")
        captured: dict[str, list[str]] = {}

        def fake_popen(command, **kwargs):
            captured["command"] = list(command)
            return _FakeNativeRunnerProcess()

        with (
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.cuda_backend.CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE",
                (8,),
            ),
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.cuda_backend.CONV2D_CUDA_COOLDOWN_MS",
                2.5,
            ),
            mock.patch(
                "compute_node.performance_metrics.conv2d.backends.cuda_backend.subprocess.Popen",
                side_effect=fake_popen,
            ),
        ):
            backend._run_runner(
                Path("C:/tmp/fake_runner.exe"),
                spec,
                layout,
                block_sizes=[64],
                tile_sizes=[8],
                transpose_modes=[0],
                output_channel_batches=[8],
                autotune_repeats=1,
                measurement_repeats=1,
                timeout_seconds=30.0,
            )

        command = captured["command"]
        self.assertIn("--output-channel-batches", command)
        self.assertEqual(command[command.index("--output-channel-batches") + 1], "8")
        self.assertIn("--cooldown-ms", command)
        self.assertEqual(command[command.index("--cooldown-ms") + 1], "2.5")
        self.assertIn("--mode", command)
        self.assertEqual(command[command.index("--mode") + 1], "benchmark")

    def test_spatial_metal_runner_command_includes_preparation_timing_flag(self) -> None:
        backend = SpatialMetalBackend()
        spec = SpatialBenchmarkSpec(
            name="unit-test",
            h=8,
            w=8,
            c_in=4,
            c_out=8,
            k=3,
            pad=1,
            ideal_seconds=1.0,
            zero_score_seconds=5.0,
            stride=1,
        )
        layout = build_conv_dataset_layout(Path("/tmp/generated"), prefix="test_")
        captured: dict[str, list[str]] = {}

        def fake_popen(command, **kwargs):
            captured["command"] = list(command)
            return _FakeNativeRunnerProcess()

        with mock.patch(
            "compute_node.performance_metrics.conv2d.backends.metal_backend.subprocess.Popen",
            side_effect=fake_popen,
        ):
            backend._run_runner(
                Path("/tmp/fake_runner"),
                spec,
                layout,
                block_sizes=[256],
                tile_sizes=[16],
                output_channel_batch=7,
                autotune_repeats=1,
                measurement_repeats=1,
                timeout_seconds=30.0,
            )

        command = captured["command"]
        self.assertIn("--include-preparation-in-metrics", command)
        self.assertEqual(command[command.index("--include-preparation-in-metrics") + 1], "1")
        self.assertIn("--mode", command)
        self.assertEqual(command[command.index("--mode") + 1], "benchmark")

    def test_metal_backend_probe_returns_status(self) -> None:
        backend = MetalBackend()
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

    def test_metal_backend_smoke_with_small_override(self) -> None:
        backend = MetalBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("Metal backend is unavailable in this environment.")

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

    @require_integration("Cross-backend runner comparisons are reserved for integration runs.")
    def test_cpu_and_metal_match_within_fp32_tolerance_on_small_override(self) -> None:
        cpu_backend = CpuBackend()
        metal_backend = MetalBackend()
        cpu_available, _ = cpu_backend.probe()
        metal_available, _ = metal_backend.probe()
        if not cpu_available or not metal_available:
            self.skipTest("CPU/Metal cross-implementation comparison is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "compute_node.performance_metrics.gemv.backends.cpu_backend.os.cpu_count",
            return_value=1,
        ):
            spec = build_benchmark_spec(rows=8, cols=16)
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)

            cpu_executable = cpu_backend._compile_if_needed()
            metal_executable, _note = metal_backend._compile_if_needed()
            cpu_output = Path(temp_dir) / "cpu_output.bin"
            metal_output = Path(temp_dir) / "metal_output.bin"

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
                "--autotune-repeats",
                str(DEFAULT_AUTOTUNE_REPEATS),
                "--measurement-repeats",
                str(DEFAULT_MEASUREMENT_REPEATS),
            ]
            metal_command = [
                str(metal_executable),
                "--matrix",
                to_relative_cli_path(layout.matrix_path, start=PERF_DIR),
                "--vector",
                to_relative_cli_path(layout.vector_path, start=PERF_DIR),
                "--output",
                to_relative_cli_path(metal_output, start=PERF_DIR),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--block-sizes",
                ",".join(str(value) for value in metal_candidate_block_sizes()),
                "--tile-sizes",
                ",".join(str(value) for value in metal_candidate_tile_sizes()),
                "--autotune-repeats",
                str(DEFAULT_AUTOTUNE_REPEATS),
                "--measurement-repeats",
                str(DEFAULT_MEASUREMENT_REPEATS),
            ]
            subprocess.run(cpu_command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=PERF_DIR)
            subprocess.run(metal_command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=PERF_DIR)

            cpu_values = load_float32_file(cpu_output)
            metal_values = load_float32_file(metal_output)
            max_abs_error, max_rel_error, _abs_index, _rel_index = compare_float32_vectors(cpu_values, metal_values)

        self.assertLessEqual(max_abs_error, 1e-3)
        self.assertLessEqual(max_rel_error, 1e-2)

    @require_integration("Cross-backend runner comparisons are reserved for integration runs.")
    def test_cpu_and_cuda_match_within_fp32_tolerance_on_small_override(self) -> None:
        cpu_backend = CpuBackend()
        cuda_backend = CudaBackend()
        cpu_available, _ = cpu_backend.probe()
        cuda_available, _ = cuda_backend.probe()
        if not cpu_available or not cuda_available:
            self.skipTest("CPU/CUDA cross-implementation comparison is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "compute_node.performance_metrics.gemv.backends.cpu_backend.os.cpu_count",
            return_value=1,
        ):
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
                "--autotune-repeats",
                str(DEFAULT_AUTOTUNE_REPEATS),
                "--measurement-repeats",
                str(DEFAULT_MEASUREMENT_REPEATS),
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
                "--autotune-repeats",
                str(DEFAULT_AUTOTUNE_REPEATS),
                "--measurement-repeats",
                str(DEFAULT_MEASUREMENT_REPEATS),
            ]
            subprocess.run(cpu_command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=PERF_DIR)
            subprocess.run(cuda_command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=PERF_DIR)

            cpu_values = load_float32_file(cpu_output)
            cuda_values = load_float32_file(cuda_output)
            max_abs_error, max_rel_error, _abs_index, _rel_index = compare_float32_vectors(cpu_values, cuda_values)

        self.assertLessEqual(max_abs_error, 1e-3)
        self.assertLessEqual(max_rel_error, 1e-2)


if __name__ == "__main__":
    unittest.main()
