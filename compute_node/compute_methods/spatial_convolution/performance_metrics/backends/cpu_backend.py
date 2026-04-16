"""CPU backend for the Convolution benchmark on Windows and macOS."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from compute_node.compute_methods.spatial_convolution.performance_metrics.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.path_utils import (
    sanitize_text,
    to_relative_cli_path,
    to_relative_string,
)
from compute_node.compute_methods.spatial_convolution.performance_metrics.scoring import (
    linear_time_score,
)
from compute_node.performance_metrics.benchmark_status import emit_status

ROOT_DIR = Path(__file__).resolve().parents[1]
CPU_ROOT_DIR = ROOT_DIR / "conv2d_runners" / "cpu"
WINDOWS_SELF_CONTAINED_NOTE = (
    "Windows CPU runner is built as a self-contained executable with the static MSVC runtime "
    "(`/MT`), so runtime does not require Visual Studio or the VC++ redistributable."
)


@dataclass(frozen=True, slots=True)
class CpuArtifacts:
    platform_key: str
    platform_label: str
    source_path: Path
    build_dir: Path
    executable_path: Path


def _cpu_artifacts_for_platform(platform: str) -> CpuArtifacts | None:
    if platform == "win32":
        platform_dir = CPU_ROOT_DIR / "windows"
        build_dir = platform_dir / "build"
        return CpuArtifacts(
            platform_key="windows", platform_label="Windows",
            source_path=platform_dir / "fmvm_cpu_windows.cpp",
            build_dir=build_dir, executable_path=build_dir / "fmvm_cpu_windows.exe",
        )
    if platform == "darwin":
        platform_dir = CPU_ROOT_DIR / "macos"
        build_dir = platform_dir / "build"
        return CpuArtifacts(
            platform_key="macos", platform_label="macOS",
            source_path=platform_dir / "fmvm_cpu_macos.cpp",
            build_dir=build_dir, executable_path=build_dir / "fmvm_cpu_macos",
        )
    return None


def _current_cpu_artifacts() -> CpuArtifacts | None:
    return _cpu_artifacts_for_platform(sys.platform)


def _relative_project_path(path: Path) -> str:
    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    return to_relative_cli_path(path, start=ROOT_DIR)


def _windows_vsdevcmd_setup_lines() -> list[str]:
    return [
        "setlocal",
        "set \"VSDEVCMD=\"",
        "if exist \"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" (",
        "  for /f \"usebackq delims=\" %%i in (`\"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" -latest -products * -find Common7\\Tools\\VsDevCmd.bat`) do set \"VSDEVCMD=%%i\"",
        ")",
        "if not defined VSDEVCMD set \"VSDEVCMD=%ProgramFiles(x86)%\\Microsoft Visual Studio\\18\\BuildTools\\Common7\\Tools\\VsDevCmd.bat\"",
        "if not exist \"%VSDEVCMD%\" exit /b 1",
        "call \"%VSDEVCMD%\" -arch=x64 -host_arch=x64 >nul",
    ]


def _binary_tree_worker_candidates(hardware_workers: int) -> list[int]:
    root = max(1, hardware_workers)
    candidates = [root, max(1, root // 2), root * 2, max(1, root // 4), root * 4]
    ordered: list[int] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _candidate_tile_sizes(cout: int) -> list[int]:
    """Return the tile sizes the CPU runner should sweep for Conv2D."""
    candidates: list[int] = []
    for value in (8, 16, 32, 64, 128):
        if value <= cout:
            candidates.append(value)
    if not candidates:
        candidates.append(cout)
    if cout not in candidates:
        candidates.append(cout)
    return candidates


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    if not binary_path.exists(): return True
    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    return DEFAULT_MEASUREMENT_REPEATS


class CpuBackend:
    name = "cpu"

    def diagnostic_context(self, spec: BenchmarkSpec | None = None) -> dict[str, object]:
        hardware_workers = os.cpu_count() or 1
        cout = 1 if spec is None else spec.c_out
        return {
            "hardware_workers": hardware_workers,
            "worker_search_order": _binary_tree_worker_candidates(hardware_workers),
            "tile_size_candidates": _candidate_tile_sizes(cout),
            "autotune_repeats": _autotune_repeats(spec),
            "measurement_repeats": _measurement_repeats(spec),
        }

    def probe(self) -> tuple[bool, str]:
        artifacts = _current_cpu_artifacts()
        if artifacts is None:
            return False, f"CPU backend is only available on Windows and macOS. Current platform: {sys.platform}."

        worker_candidates = _binary_tree_worker_candidates(os.cpu_count() or 1)
        source_is_newer_than_binary = (
                artifacts.executable_path.exists() and artifacts.source_path.exists() and
                artifacts.executable_path.stat().st_mtime < artifacts.source_path.stat().st_mtime
        )
        if artifacts.executable_path.exists():
            if not source_is_newer_than_binary:
                return (True, f"{artifacts.platform_label} CPU backend available. worker search order: {worker_candidates}")
            can_build, _ = self._can_build_for_artifacts(artifacts)
            return (True, f"{artifacts.platform_label} CPU backend available. worker search order: {worker_candidates}")

        if not artifacts.source_path.exists():
            return False, f"missing CPU runner source at {_relative_project_path(artifacts.source_path)}"

        can_build, build_message = self._can_build_for_artifacts(artifacts)
        if not can_build:
            return False, build_message
        return (True, f"{artifacts.platform_label} CPU backend available. worker search order: {worker_candidates}")

    def run(
        self,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        measurement_spec: BenchmarkSpec | None = None,
        measurement_dataset: DatasetLayout | None = None,
        time_budget_seconds: float,
        force_rebuild: bool = False,
    ) -> BackendResult:
        available, message = self.probe()
        notes = [message]
        if not available:
            return BackendResult(self.name, False, None, None, None, [], notes)

        try:
            executable_path, executable_note = self._resolve_executable_path(force_rebuild=force_rebuild)
            notes.append(executable_note)
            if sys.platform == "win32": notes.append(WINDOWS_SELF_CONTAINED_NOTE)
            if sys.platform == "darwin": notes.append(self._macos_linkage_note(executable_path))
        except Exception as exc:
            notes.append(_sanitize_note(f"failed to prepare CPU runner: {exc}"))
            return BackendResult(self.name, False, None, None, None, [], notes)

        measurement_spec = spec if measurement_spec is None else measurement_spec
        measurement_dataset = dataset if measurement_dataset is None else measurement_dataset
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else _measurement_repeats(measurement_spec)
        )
        worker_candidates = _binary_tree_worker_candidates(os.cpu_count() or 1)
        tile_candidates = _candidate_tile_sizes(spec.c_out)

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                workers=worker_candidates,
                tile_sizes=tile_candidates,
                autotune_repeats=_autotune_repeats(spec),
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except Exception as exc:
            notes.append(_sanitize_note(f"CPU autotune failed: {exc}"))
            return BackendResult(self.name, False, None, None, None, [], notes)

        selected_workers = int(autotune_metrics["actual_workers"])
        selected_tile_size = int(autotune_metrics["tile_size"])

        try:
            measurement_metrics = self._run_runner(
                executable_path,
                measurement_spec,
                measurement_dataset,
                workers=[selected_workers],
                tile_sizes=[selected_tile_size],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except Exception as exc:
            notes.append(_sanitize_note(f"CPU runtime measurement failed: {exc}"))
            return BackendResult(self.name, False, None, None, None, [], notes)

        if measurement_spec != spec or measurement_dataset != dataset:
            notes.append(f"Autotuned on {spec.name} and measured on {measurement_spec.name}.")

        autotune_score = linear_time_score(
            float(autotune_metrics["autotune_wall_clock_latency_seconds"]),
            ideal_seconds=spec.ideal_seconds,
            zero_score_seconds=spec.zero_score_seconds
        )
        measurement_score = linear_time_score(
            float(measurement_metrics["measurement_wall_clock_latency_seconds"]),
            ideal_seconds=measurement_spec.ideal_seconds,
            zero_score_seconds=measurement_spec.zero_score_seconds
        )

        autotune_config = {
            "workers": selected_workers, "requested_workers": int(autotune_metrics["requested_workers"]),
            "tile_size": selected_tile_size, "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]), "trials_run": int(autotune_metrics["trials_run"]),
        }
        measurement_config = {
            "workers": int(measurement_metrics["actual_workers"]), "requested_workers": int(measurement_metrics["requested_workers"]),
            "tile_size": int(measurement_metrics["tile_size"]), "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(measurement_metrics["measurement_repeats"]), "trials_run": int(autotune_metrics["trials_run"]),
        }
        autotune_trial = TrialRecord(self.name, autotune_config, float(autotune_metrics["autotune_wall_clock_latency_seconds"]), float(autotune_metrics["autotune_effective_gflops"]), str(autotune_metrics["autotune_checksum"]), autotune_score, [])
        trial = TrialRecord(self.name, measurement_config, float(measurement_metrics["measurement_wall_clock_latency_seconds"]), float(measurement_metrics["measurement_effective_gflops"]), str(measurement_metrics["measurement_checksum"]), measurement_score, [])
        return BackendResult(self.name, True, dict(measurement_config), autotune_trial, trial, [autotune_trial, trial], notes)

    def _run_runner(
        self,
        executable_path: Path,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        workers: list[int],
        tile_sizes: list[int],
        autotune_repeats: int,
        measurement_repeats: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        command = [
            str(executable_path),
            "--input", _relative_cli_path(dataset.input_path),
            "--weight", _relative_cli_path(dataset.weight_path),
            "--h", str(spec.h), "--w", str(spec.w),
            "--cin", str(spec.c_in), "--cout", str(spec.c_out),
            "--k", str(spec.k), "--pad", str(spec.pad),
            "--stride", str(spec.stride),
            "--workers", ",".join(str(v) for v in workers),
            "--tile-sizes", ",".join(str(v) for v in tile_sizes),
            "--autotune-repeats", str(autotune_repeats),
            "--measurement-repeats", str(measurement_repeats),
        ]
        emit_status(
            "method.spatial_convolution.backend.native_runner.start",
            status="running",
            method="spatial_convolution",
            backend=self.name,
            command=command,
            timeout_seconds=timeout_seconds,
            autotune_repeats=autotune_repeats,
            measurement_repeats=measurement_repeats,
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=ROOT_DIR,
        )
        emit_status(
            "method.spatial_convolution.backend.native_runner.complete",
            status="running",
            method="spatial_convolution",
            backend=self.name,
            command=command,
        )
        return json.loads(completed.stdout)

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> Path:
        executable_path, _ = self._resolve_executable_path(force_rebuild=force_rebuild)
        return executable_path

    def _resolve_executable_path(self, artifacts: CpuArtifacts | None = None, *, force_rebuild: bool = False) -> tuple[Path, str]:
        resolved_artifacts = _current_cpu_artifacts() if artifacts is None else artifacts
        if resolved_artifacts is None: raise FileNotFoundError("Unsupported platform")
        resolved_artifacts.build_dir.mkdir(parents=True, exist_ok=True)
        build_inputs = [resolved_artifacts.source_path, Path(__file__)]

        if force_rebuild or (resolved_artifacts.executable_path.exists() and _binary_is_stale(resolved_artifacts.executable_path, build_inputs)):
            if resolved_artifacts.platform_key == "windows": self._compile_windows_runner(resolved_artifacts)
            elif resolved_artifacts.platform_key == "macos": self._compile_macos_runner(resolved_artifacts)
        return resolved_artifacts.executable_path, "CPU runner resolved"

    def _compile_windows_runner(self, artifacts: CpuArtifacts) -> None:
        compile_script_path = artifacts.build_dir / "build_fmvm_cpu_windows.cmd"
        compile_script_path.write_text(
            "\n".join(["@echo off", *_windows_vsdevcmd_setup_lines(), "pushd \"%~dp0\"", f"cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc /Fe:{artifacts.executable_path.name} ..\\{artifacts.source_path.name}", "set \"BUILD_EXIT=%ERRORLEVEL%\"", "popd", "exit /b %BUILD_EXIT%"]) + "\n", encoding="ascii"
        )
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"& '{compile_script_path}'"], capture_output=True, text=True, check=True)

    def _compile_macos_runner(self, artifacts: CpuArtifacts) -> None:
        compiler = self._find_macos_cpp_compiler()
        subprocess.run([compiler, f"../{artifacts.source_path.name}", "-std=c++20", "-O3", "-ffast-math", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-pthread", "-Wl,-dead_strip", "-Wl,-dead_strip_dylibs", "-o", artifacts.executable_path.name], cwd=artifacts.build_dir, capture_output=True, text=True, check=True)

    def _can_build_for_artifacts(self, artifacts: CpuArtifacts) -> tuple[bool, str]:
        return True, ""

    def _macos_linkage_note(self, executable_path: Path) -> str:
        return "macOS linkage notes"

    @staticmethod
    def _find_macos_cpp_compiler() -> str | None:
        for candidate in ("clang++", "c++", "g++"):
            if shutil.which(candidate): return candidate
        return None

    @staticmethod
    def _find_vsdevcmd() -> Path | None:
        return None  # Simplified for brevity; relies on script
