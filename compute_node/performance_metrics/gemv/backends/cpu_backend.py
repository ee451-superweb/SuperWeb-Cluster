"""CPU backend for the fixed matrix-vector benchmark on Windows and macOS.

This file exists to bridge two worlds:

- Python orchestration in `benchmark.py`
- the hardware-specific C++ runners in
  `compute_node/compute_methods/gemv/cpu/<platform>/`

The important design choice is that the C++ program reads `A.bin` and `x.bin`
only once, then searches multiple worker/tile configurations internally. That
keeps the benchmark focused on compute performance instead of repeatedly paying
the 2 GiB input-file I/O cost.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from compute_node.compute_methods.gemv import (
    CPU_MACOS_BUILD_DIR,
    CPU_MACOS_EXECUTABLE_PATH,
    CPU_MACOS_SOURCE_PATH,
    CPU_WINDOWS_BUILD_DIR,
    CPU_WINDOWS_EXECUTABLE_PATH,
    CPU_WINDOWS_SOURCE_PATH,
)
from compute_node.performance_metrics.gemv.backends._native_runner_launcher import (
    run_native_runner_with_streaming,
)
from compute_node.performance_metrics.gemv.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.performance_metrics.gemv.scoring import linear_time_score
from compute_node.performance_metrics.path_utils import sanitize_text, to_relative_cli_path, to_relative_string

ROOT_DIR = Path(__file__).resolve().parents[1]
WINDOWS_SELF_CONTAINED_NOTE = (
    "Windows CPU runner is built as a self-contained executable with the static MSVC runtime "
    "(`/MT`), so runtime does not require Visual Studio or the VC++ redistributable."
)


@dataclass(frozen=True, slots=True)
class CpuArtifacts:
    """All per-platform files needed to run the CPU backend."""

    platform_key: str
    platform_label: str
    source_path: Path
    build_dir: Path
    executable_path: Path


def _cpu_artifacts_for_platform(platform: str) -> CpuArtifacts | None:
    """Resolve the CPU runner paths for one supported Python platform tag."""

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
    """Resolve the CPU runner paths for the current platform."""

    return _cpu_artifacts_for_platform(sys.platform)


def _relative_project_path(path: Path) -> str:
    """Show a project-local path without an absolute prefix."""

    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    """Remove project/home absolute prefixes from backend notes."""

    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    """Render a relative path for subprocess calls."""

    return to_relative_cli_path(path, start=ROOT_DIR)


def _windows_vsdevcmd_setup_lines() -> list[str]:
    """Emit batch-script lines that locate VsDevCmd without project absolute paths."""

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
    """Return the worker counts requested by the user.

    For example, on a 16-core machine this returns:

    - 16
    - 8
    - 32
    - 4
    - 64
    """

    root = max(1, hardware_workers)
    candidates = [
        root,
        max(1, root // 2),
        root * 2,
        max(1, root // 4),
        root * 4,
    ]

    ordered: list[int] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _candidate_tile_sizes(cols: int) -> list[int]:
    """Return the tile sizes the CPU runner should sweep.

    We keep the list short because the default benchmark already uses a 2 GiB
    matrix, so each extra configuration has a noticeable runtime cost.
    """

    candidates: list[int] = []
    for value in (256, 512, 1024, 2048, 4096, 8192):
        if value <= cols:
            candidates.append(value)
    if not candidates:
        candidates.append(cols)
    if cols not in candidates:
        candidates.append(cols)
    return candidates


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a built artifact predates any source or build-recipe input."""

    if not binary_path.exists():
        return True

    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed autotune repeat count for every CPU benchmark."""

    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed sustained-measurement repeat count for every CPU benchmark."""

    return DEFAULT_MEASUREMENT_REPEATS


def _default_worker_candidates(logical_cpu_count: int | None = None) -> list[int]:
    """Return the CPU benchmark worker sweep rooted at this machine's logical CPU count."""

    resolved = logical_cpu_count if logical_cpu_count is not None else os.cpu_count()
    return _binary_tree_worker_candidates(max(1, int(resolved or 1)))


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


class CpuBackend:
    """Compile and invoke the current platform's C++ CPU runner."""

    name = "cpu"

    def probe(self) -> tuple[bool, str]:
        """Check whether this machine can run the current platform's CPU backend."""

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
                    f"{_relative_project_path(artifacts.executable_path)}. No local compiler toolchain is required at "
                    f"runtime. worker search order: {worker_candidates}",
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
                f"{_relative_project_path(artifacts.executable_path)} is older than source, so "
                f"{_relative_project_path(artifacts.source_path)} will be rebuilt. "
                f"worker search order: {worker_candidates}",
            )

        if not artifacts.source_path.exists():
            return False, f"missing CPU runner source at {_relative_project_path(artifacts.source_path)}"

        can_build, build_message = self._can_build_for_artifacts(artifacts)
        if not can_build:
            return False, build_message
        return (
            True,
            f"{artifacts.platform_label} CPU backend available. Binary missing, will compile "
            f"{_relative_project_path(artifacts.source_path)}{build_message}. worker search order: {worker_candidates}",
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
        """Run the C++ CPU executable once and return its best found configuration."""

        available, message = self.probe()
        notes = [message]
        if not available:
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        try:
            executable_path, executable_note = self._resolve_executable_path(force_rebuild=force_rebuild)
            notes.append(executable_note)
            if sys.platform == "win32":
                notes.append(WINDOWS_SELF_CONTAINED_NOTE)
            if sys.platform == "darwin":
                notes.append(self._macos_linkage_note(executable_path))
        except (OSError, subprocess.CalledProcessError) as exc:
            details = ""
            if isinstance(exc, subprocess.CalledProcessError):
                details = (exc.stderr or exc.stdout or "").strip()
            notes.append(_sanitize_note(f"failed to prepare CPU runner: {details or exc}"))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        measurement_spec = spec if measurement_spec is None else measurement_spec
        measurement_dataset = dataset if measurement_dataset is None else measurement_dataset
        worker_candidates = _default_worker_candidates()
        tile_candidates = _candidate_tile_sizes(spec.cols)
        autotune_repeats = _autotune_repeats(spec)
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else _measurement_repeats(spec)
        )

        notes.append(f"tile search order: {tile_candidates}")
        notes.append(f"autotune_repeats_per_config: {autotune_repeats}")
        notes.append(f"measurement_repeats_for_best_config: {final_measurement_repeats}")

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                workers=worker_candidates,
                tile_sizes=tile_candidates,
                autotune_repeats=autotune_repeats,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
            )
        except subprocess.TimeoutExpired:
            notes.append("CPU benchmark timed out")
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            notes.append(_sanitize_note(stderr if stderr else str(exc)))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        selected_workers = int(autotune_metrics["actual_workers"])
        selected_tile_size = int(autotune_metrics["tile_size"])
        selected_config = {
            "workers": selected_workers,
            "tile_size": selected_tile_size,
            "accumulation_precision": str(
                autotune_metrics.get("accumulation_precision") or spec.accumulation_precision
            ),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": final_measurement_repeats,
        }
        if callable(phase_callback):
            phase_callback("final_measurement", selected_config)

        try:
            metrics = self._run_runner(
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
        except subprocess.TimeoutExpired:
            notes.append("CPU benchmark timed out")
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            notes.append(_sanitize_note(stderr if stderr else str(exc)))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        if measurement_spec != spec or measurement_dataset != dataset:
            notes.append(f"Autotuned on {spec.name} and measured on {measurement_spec.name}.")
        autotune_score = linear_time_score(
            float(autotune_metrics["autotune_wall_clock_latency_seconds"]),
            ideal_seconds=spec.ideal_seconds,
            zero_score_seconds=spec.zero_score_seconds,
        )
        measurement_score = linear_time_score(
            float(metrics["measurement_wall_clock_latency_seconds"]),
            ideal_seconds=measurement_spec.ideal_seconds,
            zero_score_seconds=measurement_spec.zero_score_seconds,
        )

        autotune_config = {
            "workers": selected_workers,
            "requested_workers": int(autotune_metrics["requested_workers"]),
            "tile_size": selected_tile_size,
            "accumulation_precision": str(autotune_metrics.get("accumulation_precision") or spec.accumulation_precision),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        config = {
            "workers": int(metrics["actual_workers"]),
            "requested_workers": int(metrics["requested_workers"]),
            "tile_size": int(metrics["tile_size"]),
            "accumulation_precision": str(metrics.get("accumulation_precision") or measurement_spec.accumulation_precision),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        trial_notes = [f"accumulation_precision={config['accumulation_precision']}"]
        autotune_trial = TrialRecord(
            backend=self.name,
            config=autotune_config,
            wall_clock_latency_seconds=float(autotune_metrics["autotune_wall_clock_latency_seconds"]),
            effective_gflops=float(autotune_metrics["autotune_effective_gflops"]),
            checksum=str(autotune_metrics["autotune_checksum"]),
            score=autotune_score,
            notes=trial_notes,
        )
        trial = TrialRecord(
            backend=self.name,
            config=config,
            wall_clock_latency_seconds=float(metrics["measurement_wall_clock_latency_seconds"]),
            effective_gflops=float(metrics["measurement_effective_gflops"]),
            checksum=str(metrics["measurement_checksum"]),
            score=measurement_score,
            notes=trial_notes,
        )
        raw_report = {
            "autotune": _extract_raw_report(autotune_metrics),
            "measurement": _extract_raw_report(metrics),
        }
        return BackendResult(
            backend=self.name,
            available=True,
            selected_config=dict(config),
            autotune_trial=autotune_trial,
            best_trial=trial,
            trials=[autotune_trial, trial],
            notes=notes,
            raw_report=raw_report,
        )

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
            tile_sizes: Tile-size candidates to sweep.
            autotune_repeats: Repeat count for autotune trials.
            measurement_repeats: Repeat count for the selected configuration.
            timeout_seconds: Subprocess timeout for the native runner.

        Returns:
            The parsed JSON metrics emitted by the native runner.
        """
        command = [
            str(executable_path),
            "--matrix",
            _relative_cli_path(dataset.matrix_path),
            "--vector",
            _relative_cli_path(dataset.vector_path),
            "--rows",
            str(spec.rows),
            "--cols",
            str(spec.cols),
            "--accumulation-precision",
            spec.accumulation_precision,
            "--workers",
            ",".join(str(value) for value in workers),
            "--tile-sizes",
            ",".join(str(value) for value in tile_sizes),
            "--autotune-repeats",
            str(autotune_repeats),
            "--measurement-repeats",
            str(measurement_repeats),
        ]
        if verbose:
            command.append("--verbose")
        return run_native_runner_with_streaming(
            command,
            timeout_seconds=timeout_seconds,
            cwd=ROOT_DIR,
        )

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> Path:
        """Return the current platform CPU executable, compiling only when needed."""

        executable_path, _note = self._resolve_executable_path(force_rebuild=force_rebuild)
        return executable_path

    def _resolve_executable_path(
        self,
        artifacts: CpuArtifacts | None = None,
        *,
        force_rebuild: bool = False,
    ) -> tuple[Path, str]:
        """Prefer the current OS binary and fall back to compiling the current OS source."""

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
                        f"using prebuilt self-contained {resolved_artifacts.platform_label} CPU binary at "
                        f"{_relative_project_path(resolved_artifacts.executable_path)} because the source or build "
                        f"recipe is newer but "
                        f"the build toolchain is unavailable ({build_message})",
                    )
            else:
                return (
                    resolved_artifacts.executable_path,
                    f"using prebuilt self-contained {resolved_artifacts.platform_label} CPU binary at "
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
            f"compiled self-contained {resolved_artifacts.platform_label} CPU runner from "
            f"{_relative_project_path(resolved_artifacts.source_path)} because {compile_reason}",
        )

    def _compile_windows_runner(self, artifacts: CpuArtifacts) -> None:
        """Compile the Windows CPU runner with MSVC."""

        if self._find_vsdevcmd() is None:
            raise FileNotFoundError("VsDevCmd.bat was not found")

        compile_script_path = artifacts.build_dir / "build_gemv_cpu_windows.cmd"
        compile_script_path.write_text(
            "\n".join(
                [
                    "@echo off",
                    *_windows_vsdevcmd_setup_lines(),
                    "pushd \"%~dp0\"",
                    # `/fp:fast` gives the compiler freedom to reassociate the
                    # FP32 reduction, which is appropriate for this performance
                    # benchmark because correctness is already checked with
                    # tolerance-based validation rather than exact bit matches.
                    # `/MT` statically links the MSVC runtime so the checked-in
                    # benchmark executable can run on a clean Windows machine
                    # without requiring the Visual Studio toolchain or a VC++
                    # redistributable install.
                    f"cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc /Fe:{artifacts.executable_path.name} "
                    f"..\\{artifacts.source_path.name}",
                    "set \"BUILD_EXIT=%ERRORLEVEL%\"",
                    "popd",
                    "exit /b %BUILD_EXIT%",
                ]
            )
            + "\n",
            encoding="ascii",
        )

        # PowerShell's call operator handles the batch-file invocation more
        # reliably than routing it back through cmd.exe argument quoting.
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"& '{compile_script_path}'",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    def _compile_macos_runner(self, artifacts: CpuArtifacts) -> None:
        """Compile the macOS CPU runner with the system C++ compiler."""

        compiler = self._find_macos_cpp_compiler()
        if compiler is None:
            raise FileNotFoundError("clang++, c++, or g++ was not found")

        completed = subprocess.run(
                [
                    compiler,
                    f"../{artifacts.source_path.name}",
                    "-std=c++20",
                    "-O3",
                    "-ffast-math",
                    # Best-effort macOS self-containment: keep symbol exports
                    # narrow and strip dead code, even though Apple toolchains
                    # still require dynamic linkage to system runtime libraries.
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
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    def _can_build_for_artifacts(self, artifacts: CpuArtifacts) -> tuple[bool, str]:
        """Report whether the current machine can compile one platform's CPU runner."""

        if artifacts.platform_key == "windows":
            if self._find_vsdevcmd() is None:
                return False, "VsDevCmd.bat was not found; MSVC build environment is unavailable."
            return True, ""

        compiler = self._find_macos_cpp_compiler()
        if compiler is None:
            return False, "No C++ compiler (clang++, c++, or g++) was found on PATH; macOS CPU backend is unavailable."
        return True, f" with {Path(compiler).name}"

    def _macos_linkage_note(self, executable_path: Path) -> str:
        """Describe the runtime libraries the macOS CPU binary still depends on."""

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

        library_summary = ", ".join(dynamic_libraries)
        return (
            "macOS CPU build uses best-effort self-contained flags, but Apple toolchains still require "
            f"dynamic system libraries: {library_summary}"
        )

    @staticmethod
    def _find_macos_cpp_compiler() -> str | None:
        """Resolve a usable C++ compiler on macOS."""

        for candidate in ("clang++", "c++", "g++"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    @staticmethod
    def _find_vsdevcmd() -> Path | None:
        """Resolve the Visual Studio developer-command batch file."""

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
                    candidate = Path(resolved[0].strip())
                    if candidate.exists():
                        return candidate

        fallback_path = (
            Path(program_files_x86)
            / "Microsoft Visual Studio"
            / "18"
            / "BuildTools"
            / "Common7"
            / "Tools"
            / "VsDevCmd.bat"
        )
        if fallback_path.exists():
            return fallback_path
        return None
