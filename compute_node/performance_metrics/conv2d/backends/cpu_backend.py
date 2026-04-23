"""CPU backend for the conv2d benchmark on Windows and macOS.

Use this module to compile or reuse the native CPU runner, sweep worker and
tile-size settings, and convert the runner's JSON output into benchmark models.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from compute_node.compute_methods.conv2d import (
    CPU_MACOS_BUILD_DIR,
    CPU_MACOS_EXECUTABLE_PATH,
    CPU_MACOS_SOURCE_PATH,
    CPU_WINDOWS_BUILD_DIR,
    CPU_WINDOWS_EXECUTABLE_PATH,
    CPU_WINDOWS_SOURCE_PATH,
)
from compute_node.performance_metrics.path_utils import (
    sanitize_text,
    to_relative_cli_path,
    to_relative_string,
)
from compute_node.performance_metrics.conv2d.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.performance_metrics.conv2d.scoring import (
    linear_time_score,
)
from compute_node.performance_metrics.benchmark_status import emit_status

ROOT_DIR = Path(__file__).resolve().parents[1]
WINDOWS_SELF_CONTAINED_NOTE = (
    "Windows CPU runner is built as a self-contained executable with the static MSVC runtime "
    "(`/MT`), so runtime does not require Visual Studio or the VC++ redistributable."
)


@dataclass(frozen=True, slots=True)
class CpuArtifacts:
    """Describe the platform-specific files needed by the CPU backend."""

    platform_key: str
    platform_label: str
    source_path: Path
    build_dir: Path
    executable_path: Path


def _cpu_artifacts_for_platform(platform: str) -> CpuArtifacts | None:
    """Resolve CPU-runner paths for one supported Python platform tag.

    Args:
        platform: Python platform tag such as ``win32`` or ``darwin``.

    Returns:
        The matching CPU-artifact bundle, or ``None`` if unsupported.
    """
    if platform == "win32":
        return CpuArtifacts(
            platform_key="windows",
            platform_label="Windows",
            source_path=CPU_WINDOWS_SOURCE_PATH,
            build_dir=CPU_WINDOWS_BUILD_DIR,
            executable_path=CPU_WINDOWS_EXECUTABLE_PATH,
        )
    if platform == "darwin":
        return CpuArtifacts(
            platform_key="macos",
            platform_label="macOS",
            source_path=CPU_MACOS_SOURCE_PATH,
            build_dir=CPU_MACOS_BUILD_DIR,
            executable_path=CPU_MACOS_EXECUTABLE_PATH,
        )
    return None


def _current_cpu_artifacts() -> CpuArtifacts | None:
    """Resolve CPU-runner paths for the current host platform.

    Returns:
        The current platform's CPU-artifact bundle, or ``None``.
    """
    return _cpu_artifacts_for_platform(sys.platform)


def _relative_project_path(path: Path) -> str:
    """Render one project-local path without an absolute machine prefix."""
    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    """Strip absolute-path prefixes from backend note text."""
    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    """Render one path for use in subprocess command lines."""
    return to_relative_cli_path(path, start=ROOT_DIR)


def _windows_vsdevcmd_setup_lines() -> list[str]:
    """Return batch-file lines that initialize the MSVC build environment."""
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
    """Build the ordered worker-count sweep used during CPU autotune.

    Args:
        hardware_workers: Capped hardware worker count for this machine.

    Returns:
        Worker counts ordered around the hardware default.
    """
    root = max(1, hardware_workers)
    candidates = [root, max(1, root // 2), root * 2, max(1, root // 4), root * 4]
    ordered: list[int] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _candidate_tile_sizes(cout: int) -> list[int]:
    """Return the tile sizes the CPU runner should sweep for Conv2D.

    Args:
        cout: Output-channel count for the autotune workload.

    Returns:
        Tile sizes worth testing for this workload.
    """
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
    """Return whether a compiled binary predates any relevant input files.

    Args:
        binary_path: Runner binary that may need rebuilding.
        inputs: Source or recipe files that should be newer than the binary.

    Returns:
        ``True`` when the binary is missing or older than any input.
    """
    if not binary_path.exists(): return True
    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed autotune repeat count for CPU Conv2D benchmarks."""
    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed measurement repeat count for CPU Conv2D benchmarks."""
    return DEFAULT_MEASUREMENT_REPEATS


_RAW_REPORT_KEYS = (
    "flops_per_run",
    "bytes_input",
    "bytes_weight",
    "bytes_output",
    "bytes_kernel_compulsory_memory_traffic",
    "notes_schema",
    "trials",
)


def _extract_raw_report(metrics: dict[str, object]) -> dict[str, object]:
    """Copy analysis-only fields out of the runner JSON for the benchmark report."""
    return {key: metrics[key] for key in _RAW_REPORT_KEYS if key in metrics}


def _default_worker_candidates(logical_cpu_count: int | None = None) -> list[int]:
    """Return worker-count candidates rooted at this machine's logical CPU count.

    Args:
        logical_cpu_count: Optional explicit logical CPU count override.

    Returns:
        Worker counts derived from the binary-tree sweep around the logical CPU count.
    """
    resolved = logical_cpu_count if logical_cpu_count is not None else os.cpu_count()
    return _binary_tree_worker_candidates(max(1, int(resolved or 1)))


class CpuBackend:
    """Compile, probe, and run the native spatial CPU benchmark backend."""

    name = "cpu"

    def diagnostic_context(self, spec: BenchmarkSpec | None = None) -> dict[str, object]:
        """Return structured tuning context for status logging.

        Args:
            spec: Optional benchmark spec being prepared.

        Returns:
            A dictionary describing worker and tile search settings.
        """
        logical_cpu_count = os.cpu_count() or 1
        cout = 1 if spec is None else spec.c_out
        return {
            "logical_cpu_count": logical_cpu_count,
            "hardware_workers": logical_cpu_count,
            "worker_search_order": _binary_tree_worker_candidates(logical_cpu_count),
            "tile_size_candidates": _candidate_tile_sizes(cout),
            "autotune_repeats": _autotune_repeats(spec),
            "measurement_repeats": _measurement_repeats(spec),
        }

    def probe(self) -> tuple[bool, str]:
        """Check whether this machine can run the spatial CPU backend.

        Returns:
            A ``(available, message)`` tuple describing backend readiness.
        """
        artifacts = _current_cpu_artifacts()
        if artifacts is None:
            return False, f"CPU backend is only available on Windows and macOS. Current platform: {sys.platform}."

        worker_candidates = _default_worker_candidates()
        source_is_newer_than_binary = (
            artifacts.executable_path.exists()
            and artifacts.source_path.exists()
            and artifacts.executable_path.stat().st_mtime < artifacts.source_path.stat().st_mtime
        )
        if artifacts.executable_path.exists():
            if not source_is_newer_than_binary:
                return (
                    True,
                    f"{artifacts.platform_label} CPU backend available via self-contained binary at "
                    f"{_relative_project_path(artifacts.executable_path)}. worker search order: {worker_candidates}",
                )
            can_build, build_message = self._can_build_for_artifacts(artifacts)
            if not can_build:
                return (
                    True,
                    f"{artifacts.platform_label} CPU backend available via self-contained binary at "
                    f"{_relative_project_path(artifacts.executable_path)}. Source is newer, but the build toolchain "
                    f"is unavailable ({build_message}), so the existing binary will be used. "
                    f"worker search order: {worker_candidates}",
                )
            return (
                True,
                f"{artifacts.platform_label} CPU backend available. Existing binary at "
                f"{_relative_project_path(artifacts.executable_path)} is older than the source, so the runner can be "
                f"rebuilt locally. worker search order: {worker_candidates}",
            )

        if not artifacts.source_path.exists():
            return False, f"missing CPU runner source at {_relative_project_path(artifacts.source_path)}"

        can_build, build_message = self._can_build_for_artifacts(artifacts)
        if not can_build:
            return False, build_message
        return (
            True,
            f"{artifacts.platform_label} CPU backend available. Binary is missing, so it can be built locally "
            f"({build_message}). worker search order: {worker_candidates}",
        )

    def run(
        self,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        measurement_spec: BenchmarkSpec | None = None,
        measurement_dataset: DatasetLayout | None = None,
        time_budget_seconds: float,
        force_rebuild: bool = False,
        phase_callback=None,
        verbose: bool = False,
    ) -> BackendResult:
        """Run autotune plus final measurement for the CPU backend.

        Args:
            spec: Benchmark spec used for autotune.
            dataset: Dataset layout used for autotune.
            measurement_spec: Optional spec for the final measurement phase.
            measurement_dataset: Optional dataset for the final measurement phase.
            time_budget_seconds: Overall time budget for the backend run.
            force_rebuild: Whether to force recompiling the native runner.

        Returns:
            A ``BackendResult`` describing availability and measured performance.
        """
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
        worker_candidates = _default_worker_candidates()
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
                verbose=verbose,
            )
        except Exception as exc:
            notes.append(_sanitize_note(f"CPU autotune failed: {exc}"))
            return BackendResult(self.name, False, None, None, None, [], notes)

        selected_workers = int(autotune_metrics["actual_workers"])
        selected_tile_size = int(autotune_metrics["tile_size"])
        selected_config = {
            "workers": selected_workers,
            "requested_workers": selected_workers,
            "tile_size": selected_tile_size,
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": final_measurement_repeats,
        }
        if callable(phase_callback):
            phase_callback("final_measurement", selected_config)

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
                verbose=verbose,
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
        raw_report = {
            "autotune": _extract_raw_report(autotune_metrics),
            "measurement": _extract_raw_report(measurement_metrics),
        }
        return BackendResult(self.name, True, dict(measurement_config), autotune_trial, trial, [autotune_trial, trial], notes, raw_report)

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
        verbose: bool = False,
    ) -> dict[str, object]:
        """Invoke the native CPU runner and parse its JSON metrics.

        Args:
            executable_path: Runner binary to execute.
            spec: Benchmark spec being measured.
            dataset: Dataset layout read by the runner.
            workers: Worker-count candidates to sweep.
            tile_sizes: Tile sizes to sweep.
            autotune_repeats: Repeat count for autotune trials.
            measurement_repeats: Repeat count for the selected configuration.
            timeout_seconds: Subprocess timeout for the native runner.
            verbose: Request per-trial progress on the runner's stderr channel.

        Returns:
            The parsed JSON metrics emitted by the native runner.
        """
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
        if verbose:
            command.append("--verbose")
        emit_status(
            "method.conv2d.backend.native_runner.start",
            status="running",
            method="conv2d",
            backend=self.name,
            command=command,
            timeout_seconds=timeout_seconds,
            autotune_repeats=autotune_repeats,
            measurement_repeats=measurement_repeats,
        )
        # Stream stderr line-by-line so verbose progress reaches the parent
        # terminal (and any bootstrap log tee) as it happens, while stdout is
        # read in a separate thread and parsed as JSON after the process exits.
        # We cannot use ``process.communicate()`` here: it spawns its own
        # drain thread that would race our stderr pump for the same pipe and
        # silently swallow per-trial progress lines.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT_DIR,
        )
        stderr_chunks: list[str] = []
        stdout_chunks: list[str] = []

        def _pump_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                stderr_chunks.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()

        def _pump_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_chunks.append(line)

        stderr_pump = threading.Thread(target=_pump_stderr, daemon=True)
        stdout_pump = threading.Thread(target=_pump_stdout, daemon=True)
        stderr_pump.start()
        stdout_pump.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            stderr_pump.join(timeout=1.0)
            stdout_pump.join(timeout=1.0)
            raise
        stderr_pump.join(timeout=5.0)
        stdout_pump.join(timeout=5.0)
        stdout_data = "".join(stdout_chunks)
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output=stdout_data,
                stderr="".join(stderr_chunks),
            )
        emit_status(
            "method.conv2d.backend.native_runner.complete",
            status="running",
            method="conv2d",
            backend=self.name,
            command=command,
        )
        return json.loads(stdout_data)

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> Path:
        """Ensure the native CPU runner exists and return its path.

        Args:
            force_rebuild: Whether to force a rebuild of the runner binary.

        Returns:
            The resolved runner executable path.
        """
        executable_path, _ = self._resolve_executable_path(force_rebuild=force_rebuild)
        return executable_path

    def _resolve_executable_path(self, artifacts: CpuArtifacts | None = None, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Resolve or build the runner executable for the current platform.

        Args:
            artifacts: Optional explicit artifact bundle to use.
            force_rebuild: Whether to force recompiling the runner.

        Returns:
            A tuple of ``(executable_path, note)`` describing the resolution.
        """
        resolved_artifacts = _current_cpu_artifacts() if artifacts is None else artifacts
        if resolved_artifacts is None:
            raise FileNotFoundError(f"CPU backend is unsupported on platform {sys.platform!r}")
        resolved_artifacts.build_dir.mkdir(parents=True, exist_ok=True)
        compile_reason = "binary is missing"
        build_inputs = [resolved_artifacts.source_path, Path(__file__)]
        if force_rebuild:
            compile_reason = "rebuild was explicitly requested"
            can_build, build_message = self._can_build_for_artifacts(resolved_artifacts)
            if not can_build:
                raise FileNotFoundError(
                    f"force rebuild requested for the {resolved_artifacts.platform_label} CPU runner, but the build "
                    f"toolchain is unavailable ({build_message})"
                )
        elif resolved_artifacts.executable_path.exists():
            if _binary_is_stale(resolved_artifacts.executable_path, build_inputs):
                can_build, build_message = self._can_build_for_artifacts(resolved_artifacts)
                if can_build:
                    compile_reason = "source or build recipe is newer than binary"
                else:
                    return (
                        resolved_artifacts.executable_path,
                        f"using prebuilt {resolved_artifacts.platform_label} CPU binary at "
                        f"{_relative_project_path(resolved_artifacts.executable_path)} because the source or build "
                        f"recipe is newer but the build toolchain is unavailable ({build_message})",
                    )
            else:
                return (
                    resolved_artifacts.executable_path,
                    f"using prebuilt {resolved_artifacts.platform_label} CPU binary at "
                    f"{_relative_project_path(resolved_artifacts.executable_path)}",
                )

        if not resolved_artifacts.source_path.exists():
            raise FileNotFoundError(
                f"missing CPU runner source at {_relative_project_path(resolved_artifacts.source_path)}"
            )

        if resolved_artifacts.platform_key == "windows":
            self._compile_windows_runner(resolved_artifacts)
        elif resolved_artifacts.platform_key == "macos":
            self._compile_macos_runner(resolved_artifacts)
        else:
            raise FileNotFoundError(f"no CPU build flow is registered for platform {resolved_artifacts.platform_key!r}")

        if not resolved_artifacts.executable_path.exists():
            raise FileNotFoundError(
                f"CPU runner build completed without producing {_relative_project_path(resolved_artifacts.executable_path)}"
            )
        return (
            resolved_artifacts.executable_path,
            f"compiled {resolved_artifacts.platform_label} CPU runner from "
            f"{_relative_project_path(resolved_artifacts.source_path)} because {compile_reason}",
        )

    def _compile_windows_runner(self, artifacts: CpuArtifacts) -> None:
        """Build the Windows CPU runner through an on-disk batch script.

        Args:
            artifacts: Windows artifact bundle describing source and output paths.

        Returns:
            ``None`` after the compile step finishes successfully.
        """
        compile_script_path = artifacts.build_dir / "build_conv2d_cpu_windows.cmd"
        compile_script_path.write_text(
            "\n".join(["@echo off", *_windows_vsdevcmd_setup_lines(), "pushd \"%~dp0\"", f"cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc /Fe:{artifacts.executable_path.name} ..\\{artifacts.source_path.name}", "set \"BUILD_EXIT=%ERRORLEVEL%\"", "popd", "exit /b %BUILD_EXIT%"]) + "\n", encoding="ascii"
        )
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"& '{compile_script_path}'"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

    def _compile_macos_runner(self, artifacts: CpuArtifacts) -> None:
        """Build the macOS CPU runner with the detected C++ compiler.

        Args:
            artifacts: macOS artifact bundle describing source and output paths.

        Returns:
            ``None`` after the compile step finishes successfully.
        """
        compiler = self._find_macos_cpp_compiler()
        if compiler is None:
            raise FileNotFoundError("clang++, c++, or g++ was not found")
        subprocess.run(
            [
                compiler,
                f"../{artifacts.source_path.name}",
                "-std=c++20",
                "-O3",
                "-ffast-math",
                "-fvisibility=hidden",
                "-fvisibility-inlines-hidden",
                "-pthread",
                "-Wl,-dead_strip",
                "-Wl,-dead_strip_dylibs",
                "-o",
                artifacts.executable_path.name,
            ],
            cwd=artifacts.build_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

    def _can_build_for_artifacts(self, artifacts: CpuArtifacts) -> tuple[bool, str]:
        """Return whether the backend can build the given artifact bundle.

        Args:
            artifacts: Platform-specific artifact bundle.

        Returns:
            A ``(can_build, message)`` tuple for probe-time checks.
        """
        if artifacts.platform_key == "windows":
            if self._find_vsdevcmd() is None:
                return False, "VsDevCmd.bat was not found; MSVC build environment is unavailable."
            return True, "MSVC build environment is available"

        compiler = self._find_macos_cpp_compiler()
        if compiler is None:
            return False, "No C++ compiler (clang++, c++, or g++) was found on PATH."
        return True, f"compiler {Path(compiler).name} is available"

    def _macos_linkage_note(self, executable_path: Path) -> str:
        """Return a short linkage note for the macOS runner binary.

        Args:
            executable_path: Resolved macOS runner binary path.

        Returns:
            A short note describing macOS linkage expectations.
        """
        completed = subprocess.run(
            ["otool", "-L", str(executable_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            return "macOS CPU linkage inspection failed; full static linkage is not expected on Apple toolchains."

        dynamic_libraries: list[str] = []
        for line in completed.stdout.splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            library_path = stripped.split(" (compatibility version", 1)[0].strip()
            dynamic_libraries.append(Path(library_path).name or library_path)

        if not dynamic_libraries:
            return "macOS CPU linkage inspection found no explicit dylib records."

        return (
            "macOS CPU build uses best-effort self-contained flags, but Apple toolchains still require "
            f"dynamic system libraries: {', '.join(dynamic_libraries)}"
        )

    @staticmethod
    def _find_macos_cpp_compiler() -> str | None:
        """Find an available macOS C++ compiler on ``PATH``.

        Returns:
            The compiler executable name, or ``None`` when unavailable.
        """
        for candidate in ("clang++", "c++", "g++"):
            if shutil.which(candidate): return candidate
        return None

    @staticmethod
    def _find_vsdevcmd() -> Path | None:
        """Return the VS developer command path when available.

        Returns:
            The resolved ``VsDevCmd`` path, or ``None`` when unavailable.
        """
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        if not program_files_x86:
            return None

        vswhere_path = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere_path.exists():
            completed = subprocess.run(
                [
                    str(vswhere_path),
                    "-latest",
                    "-products",
                    "*",
                    "-find",
                    "Common7\\Tools\\VsDevCmd.bat",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode == 0:
                resolved = completed.stdout.strip().splitlines()
                if resolved:
                    return Path(resolved[0])

        fallback = Path(program_files_x86) / "Microsoft Visual Studio" / "18" / "BuildTools" / "Common7" / "Tools" / "VsDevCmd.bat"
        if fallback.exists():
            return fallback
        return None
