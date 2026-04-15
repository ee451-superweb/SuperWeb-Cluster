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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PERF_DIR = (
    Path(__file__).resolve().parents[1]
    / "compute_node"
    / "performance_metrics"
    / "fixed_matrix_vector_multiplication"
)

from compute_node.performance_metrics import benchmark
from compute_node.performance_metrics.fixed_matrix_vector_multiplication import backends as backend_registry
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cpu_backend import (
    CpuArtifacts,
    CpuBackend,
    _binary_tree_worker_candidates,
    _candidate_tile_sizes as cpu_candidate_tile_sizes,
    _cpu_artifacts_for_platform,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend import (
    CudaBackend,
    _candidate_block_sizes as cuda_candidate_block_sizes,
    _candidate_tile_sizes as cuda_candidate_tile_sizes,
    _candidate_transpose_modes as cuda_candidate_transpose_modes,
    _windows_gencode_args,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.dx12_backend import (
    Dx12Backend,
    _detect_non_nvidia_windows_adapter,
    _candidate_rows_per_thread as dx12_candidate_rows_per_thread,
    _candidate_thread_group_sizes as dx12_candidate_thread_group_sizes,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.metal_backend import (
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
from compute_node.input_matrix import generate as input_matrix_generate_cli
from compute_node.input_matrix.spatial_convolution import (
    build_dataset_layout as build_conv_dataset_layout,
    build_input_matrix_spec as build_conv_input_matrix_spec,
    dataset_is_generated as conv_dataset_is_generated,
    generate_dataset as generate_conv_dataset,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
)
from compute_node.performance_metrics.path_utils import to_relative_cli_path
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.scoring import linear_time_score
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.workloads import build_benchmark_spec
import subprocess

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION


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

    def test_input_matrix_cli_defaults_to_all_methods(self) -> None:
        args = input_matrix_generate_cli.build_parser().parse_args([])
        self.assertEqual(args.method, "all")

    def test_windows_default_backend_order_routes_gpu_by_display_adapter(self) -> None:
        with (
            mock.patch(
                "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.os.name",
                "nt",
            ),
            mock.patch(
                "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.detect_nvidia_windows_adapter",
                return_value=("NVIDIA GeForce RTX 4060 Laptop GPU", ""),
            ),
            mock.patch(
                "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.detect_non_nvidia_windows_adapter",
                return_value=("AMD Radeon 780M Graphics", ""),
            ),
        ):
            names = [backend.name for backend in backend_registry.build_backends()]

        self.assertEqual(names, ["cpu", "cuda", "dx12"])

    def test_cpu_artifacts_follow_platform(self) -> None:
        windows_artifacts = _cpu_artifacts_for_platform("win32")
        assert windows_artifacts is not None
        self.assertEqual(windows_artifacts.platform_key, "windows")
        self.assertEqual(windows_artifacts.executable_path.name, "fmvm_cpu_windows.exe")

        macos_artifacts = _cpu_artifacts_for_platform("darwin")
        assert macos_artifacts is not None
        self.assertEqual(macos_artifacts.platform_key, "macos")
        self.assertEqual(macos_artifacts.executable_path.name, "fmvm_cpu_macos")

        self.assertIsNone(_cpu_artifacts_for_platform("linux"))

    def test_cpu_backend_prefers_existing_binary_before_compiling(self) -> None:
        backend = CpuBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifacts = CpuArtifacts(
                platform_key="macos",
                platform_label="macOS",
                source_path=temp_root / "fmvm_cpu_macos.cpp",
                build_dir=temp_root / "build",
                executable_path=temp_root / "build" / "fmvm_cpu_macos",
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
                source_path=temp_root / "fmvm_cpu_macos.cpp",
                build_dir=temp_root / "build",
                executable_path=temp_root / "build" / "fmvm_cpu_macos",
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

    def test_default_spec_matches_requested_2gib_shape(self) -> None:
        spec = build_benchmark_spec()
        self.assertEqual(spec.rows, 16_384)
        self.assertEqual(spec.cols, 32_768)
        self.assertEqual(spec.matrix_bytes, 2 * 1024**3)
        self.assertEqual(spec.vector_bytes, 32_768 * 4)

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

    def test_spatial_convolution_dataset_generation_supports_small_override(self) -> None:
        spec = build_conv_input_matrix_spec(h=16, w=16, c_in=4, c_out=8, k=3, pad=1, stride=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_conv_dataset_layout(Path(temp_dir), prefix="test_")
            generate_conv_dataset(layout, spec, generator_workers=2, chunk_values=16)
            self.assertTrue(conv_dataset_is_generated(layout, spec))
            self.assertEqual(layout.input_path.stat().st_size, spec.input_bytes)
            self.assertEqual(layout.weight_path.stat().st_size, spec.weight_bytes)

    def test_small_override_uses_override_dataset_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            default_dataset_dir = temp_root / "generated"
            args = argparse.Namespace(
                method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
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

        method_report = report["methods"][METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION]
        dataset_root = Path(method_report["dataset"]["root_dir"])
        self.assertEqual(dataset_root.parts[-3:], ("generated", "overrides", "8x16"))

    def test_benchmark_auto_generates_and_runs_cpu(self) -> None:
        backend = CpuBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("CPU backend is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cpu_backend.os.cpu_count",
            return_value=1,
        ):
            args = argparse.Namespace(
                method=METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
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

        self.assertEqual(report["schema_version"], 4)
        self.assertIn("device_overview", report)
        self.assertIn("methods", report)
        method_report = report["methods"][METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION]
        self.assertEqual(method_report["method"], METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION)
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

    def test_dx12_candidate_search_spaces_are_short(self) -> None:
        self.assertEqual(dx12_candidate_thread_group_sizes(), [256, 512])
        self.assertEqual(dx12_candidate_rows_per_thread(), [1, 2])

    def test_cuda_backend_probe_accepts_prebuilt_windows_runner_without_nvcc(self) -> None:
        backend = CudaBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            binary_path = temp_root / "fmvm_cuda_runner.exe"
            source_path = temp_root / "fmvm_cuda_runner.cu"
            binary_path.write_text("binary placeholder\n", encoding="utf-8")
            source_path.write_text("// source placeholder\n", encoding="utf-8")

            with (
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend.os.name",
                    "nt",
                ),
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend.CUDA_EXECUTABLE_PATH",
                    binary_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend.CUDA_SOURCE_PATH",
                    source_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend._detect_compute_capability",
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
            binary_path = temp_root / "fmvm_cuda_runner.exe"
            source_path = temp_root / "fmvm_cuda_runner.cu"
            binary_path.write_text("binary placeholder\n", encoding="utf-8")
            source_path.write_text("// source placeholder\n", encoding="utf-8")

            with (
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend.CUDA_EXECUTABLE_PATH",
                    binary_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cuda_backend.CUDA_SOURCE_PATH",
                    source_path,
                ),
                mock.patch.object(backend, "_toolchain_status", return_value=(False, "nvcc missing")),
            ):
                with self.assertRaises(FileNotFoundError):
                    backend._resolve_executable_path(force_rebuild=True)

    def test_dx12_backend_probe_returns_status(self) -> None:
        backend = Dx12Backend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)

    def test_detect_non_nvidia_windows_adapter_prefers_amd_or_intel(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["powershell"],
            returncode=0,
            stdout=json.dumps(
                [
                    {"Name": "NVIDIA GeForce RTX 4060 Laptop GPU", "AdapterCompatibility": "NVIDIA", "PNPDeviceID": "PCI\\VEN_10DE"},
                    {"Name": "AMD Radeon 780M Graphics", "AdapterCompatibility": "Advanced Micro Devices, Inc.", "PNPDeviceID": "PCI\\VEN_1002"},
                ]
            ),
            stderr="",
        )
        with mock.patch(
            "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.dx12_backend.subprocess.run",
            return_value=completed,
        ):
            adapter_name, message = _detect_non_nvidia_windows_adapter()

        self.assertEqual(adapter_name, "AMD Radeon 780M Graphics")
        self.assertEqual(message, "")

    def test_dx12_backend_probe_rejects_nvidia_only_hosts(self) -> None:
        backend = Dx12Backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "fmvm_dx12_runner.cpp"
            source_path.write_text("// dx12 placeholder\n", encoding="utf-8")
            with (
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.dx12_backend.os.name",
                    "nt",
                ),
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.dx12_backend.DX12_SOURCE_PATH",
                    source_path,
                ),
                mock.patch(
                    "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.dx12_backend._detect_non_nvidia_windows_adapter",
                    return_value=(None, "no AMD or Intel adapter"),
                ),
            ):
                available, message = backend.probe()

        self.assertFalse(available)
        self.assertIn("no AMD or Intel adapter", message)

    def test_dx12_backend_rejects_fp64_accumulate(self) -> None:
        backend = Dx12Backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            spec = build_benchmark_spec(rows=8, cols=16, accumulation_precision="fp64_accumulate")
            result = backend.run(spec, layout, time_budget_seconds=1.0)

        self.assertFalse(result.available)
        self.assertIn("only fp32 accumulation", result.notes[0])

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

    def test_cpu_and_metal_match_within_fp32_tolerance_on_small_override(self) -> None:
        cpu_backend = CpuBackend()
        metal_backend = MetalBackend()
        cpu_available, _ = cpu_backend.probe()
        metal_available, _ = metal_backend.probe()
        if not cpu_available or not metal_available:
            self.skipTest("CPU/Metal cross-implementation comparison is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cpu_backend.os.cpu_count",
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
            subprocess.run(cpu_command, check=True, capture_output=True, text=True, cwd=PERF_DIR)
            subprocess.run(metal_command, check=True, capture_output=True, text=True, cwd=PERF_DIR)

            cpu_values = load_float32_file(cpu_output)
            metal_values = load_float32_file(metal_output)
            max_abs_error, max_rel_error, _abs_index, _rel_index = compare_float32_vectors(cpu_values, metal_values)

        self.assertLessEqual(max_abs_error, 1e-3)
        self.assertLessEqual(max_rel_error, 1e-2)

    def test_cpu_and_cuda_match_within_fp32_tolerance_on_small_override(self) -> None:
        cpu_backend = CpuBackend()
        cuda_backend = CudaBackend()
        cpu_available, _ = cpu_backend.probe()
        cuda_available, _ = cuda_backend.probe()
        if not cpu_available or not cuda_available:
            self.skipTest("CPU/CUDA cross-implementation comparison is unavailable in this environment.")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.cpu_backend.os.cpu_count",
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
            subprocess.run(cpu_command, check=True, capture_output=True, text=True, cwd=PERF_DIR)
            subprocess.run(cuda_command, check=True, capture_output=True, text=True, cwd=PERF_DIR)

            cpu_values = load_float32_file(cpu_output)
            cuda_values = load_float32_file(cuda_output)
            max_abs_error, max_rel_error, _abs_index, _rel_index = compare_float32_vectors(cpu_values, cuda_values)

        self.assertLessEqual(max_abs_error, 1e-3)
        self.assertLessEqual(max_rel_error, 1e-2)


if __name__ == "__main__":
    unittest.main()
