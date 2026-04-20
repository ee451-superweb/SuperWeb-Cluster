"""DirectX 12 backend for the conv2d benchmark on Windows.

Use this module to probe non-NVIDIA Windows adapters, compile or reuse the
native DX12 runner, sweep DX12 launch settings, and convert runner output into
benchmark models used by the reporting pipeline.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

from compute_node.compute_methods.conv2d import (
    DX12_BUILD_DIR,
    DX12_EXECUTABLE_PATH,
    DX12_SOURCE_PATH,
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
from compute_node.performance_metrics.gemv.backends.windows_gpu_inventory import (
    detect_non_nvidia_windows_adapter,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
WINDOWS_DX12_RUNTIME_NOTE = (
    "Windows DX12 runner uses the system Direct3D 12 runtime and the installed graphics driver."
)
_MAX_LOG_TAIL_CHARS = 1200

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
    """Pull the benchmark-only detail fields out of a native runner's JSON."""
    return {key: metrics[key] for key in _RAW_REPORT_KEYS if key in metrics}


def _relative_project_path(path: Path) -> str:
    """Render one project-local path without an absolute machine prefix."""
    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    """Strip absolute-path prefixes from backend note text."""
    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    """Render one path for use in subprocess command lines."""
    return to_relative_cli_path(path, start=ROOT_DIR)


def _tail_text(text: str | None, *, limit: int = _MAX_LOG_TAIL_CHARS) -> str:
    """Return the sanitized tail of a long log or error string."""
    if not text:
        return ""
    normalized = sanitize_text(text.strip(), start=ROOT_DIR)
    if len(normalized) <= limit:
        return normalized
    return f"...{normalized[-limit:]}"


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


def _candidate_thread_group_sizes() -> list[int]:
    """Return DX12 thread-group-size candidates for autotune."""
    return [64, 128, 256, 512]


def _candidate_tile_sizes(c_out: int) -> list[int]:
    """Return DX12 tile-size candidates for autotune."""
    candidates = [value for value in (1, 2, 4, 8) if value <= c_out]
    return candidates or [1]


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed autotune repeat count for DX12 Conv2D benchmarks."""
    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed measurement repeat count for DX12 Conv2D benchmarks."""
    return DEFAULT_MEASUREMENT_REPEATS


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a compiled binary predates any relevant input files."""
    if not binary_path.exists():
        return True

    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


class Dx12Backend:
    """Compile, probe, and run the native spatial DX12 benchmark backend."""

    name = "dx12"

    def diagnostic_context(self, spec: BenchmarkSpec | None = None) -> dict[str, object]:
        """Return structured tuning context for status logging."""
        c_out = 1 if spec is None else spec.c_out
        adapter_name, _adapter_message = detect_non_nvidia_windows_adapter()
        return {
            "adapter_name": adapter_name,
            "thread_group_size_candidates": _candidate_thread_group_sizes(),
            "tile_size_candidates": _candidate_tile_sizes(c_out),
            "autotune_repeats": _autotune_repeats(spec),
            "measurement_repeats": _measurement_repeats(spec),
            "runner_path": str(DX12_EXECUTABLE_PATH),
            "source_path": str(DX12_SOURCE_PATH),
        }

    def probe(self) -> tuple[bool, str]:
        """Check whether this machine can run the spatial DX12 backend."""
        if os.name != "nt":
            return False, "DX12 backend is only available on Windows."
        if not DX12_SOURCE_PATH.exists():
            return False, f"missing DX12 runner source at {_relative_project_path(DX12_SOURCE_PATH)}"

        adapter_name, adapter_message = detect_non_nvidia_windows_adapter()
        if adapter_name is None:
            return False, adapter_message

        build_inputs = [DX12_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()

        if DX12_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(DX12_EXECUTABLE_PATH, build_inputs):
                return (
                    True,
                    f"DX12 backend available for non-NVIDIA adapter {adapter_name!r} via "
                    f"{_relative_project_path(DX12_EXECUTABLE_PATH)}.",
                )
            if not toolchain_available:
                return (
                    True,
                    f"DX12 backend available for {adapter_name!r} via {_relative_project_path(DX12_EXECUTABLE_PATH)}, "
                    f"but the local MSVC build environment is unavailable ({toolchain_message}), so the existing binary will be used.",
                )
            return (
                True,
                f"DX12 backend available for {adapter_name!r}. Existing binary is older than the source, so it will be rebuilt.",
            )

        if not toolchain_available:
            return False, toolchain_message

        return True, f"DX12 backend available for {adapter_name!r}; the runner will be compiled from source."

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
        """Run autotune plus final measurement for the DX12 backend."""
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
            notes.append(WINDOWS_DX12_RUNTIME_NOTE)
        except (OSError, subprocess.CalledProcessError) as exc:
            details = ""
            if isinstance(exc, subprocess.CalledProcessError):
                details = (exc.stderr or exc.stdout or "").strip()
            notes.append(_sanitize_note(f"failed to prepare DX12 runner: {details or exc}"))
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
        thread_group_sizes = _candidate_thread_group_sizes()
        tile_sizes = _candidate_tile_sizes(spec.c_out)
        autotune_repeats = _autotune_repeats(spec)
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else _measurement_repeats(spec)
        )

        notes.append(f"thread-group-size search order: {thread_group_sizes}")
        notes.append(f"tile size search order: {tile_sizes}")
        notes.append(f"autotune_repeats_per_config: {autotune_repeats}")
        notes.append(f"measurement_repeats_for_best_config: {final_measurement_repeats}")

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                phase="autotune",
                thread_group_sizes=thread_group_sizes,
                tile_sizes=tile_sizes,
                autotune_repeats=autotune_repeats,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
            )
        except subprocess.TimeoutExpired:
            notes.append("DX12 benchmark timed out")
            return BackendResult(self.name, False, None, None, None, [], notes)
        except subprocess.CalledProcessError as exc:
            details = _tail_text(exc.stderr) or _tail_text(exc.stdout) or _sanitize_note(str(exc))
            notes.append(details)
            return BackendResult(self.name, False, None, None, None, [], notes)
        except Exception as exc:
            notes.append(_sanitize_note(str(exc)))
            return BackendResult(self.name, False, None, None, None, [], notes)

        invalid_reason = self._validate_metrics(autotune_metrics)
        if invalid_reason is not None:
            notes.append(invalid_reason)
            return BackendResult(self.name, False, None, None, None, [], notes)

        try:
            metrics = self._run_runner(
                executable_path,
                measurement_spec,
                measurement_dataset,
                phase="measurement",
                thread_group_sizes=[int(autotune_metrics["thread_group_size"])],
                tile_sizes=[int(autotune_metrics["tile_size"])],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
            )
        except subprocess.TimeoutExpired:
            notes.append("DX12 benchmark timed out")
            return BackendResult(self.name, False, None, None, None, [], notes)
        except subprocess.CalledProcessError as exc:
            details = _tail_text(exc.stderr) or _tail_text(exc.stdout) or _sanitize_note(str(exc))
            notes.append(details)
            return BackendResult(self.name, False, None, None, None, [], notes)
        except Exception as exc:
            notes.append(_sanitize_note(str(exc)))
            return BackendResult(self.name, False, None, None, None, [], notes)

        invalid_reason = self._validate_metrics(metrics)
        if invalid_reason is not None:
            notes.append(invalid_reason)
            return BackendResult(self.name, False, None, None, None, [], notes)

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
            "thread_group_size": int(autotune_metrics["thread_group_size"]),
            "tile_size": int(autotune_metrics["tile_size"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
            "accumulation_precision": "fp32",
        }
        measurement_config = {
            "thread_group_size": int(metrics["thread_group_size"]),
            "tile_size": int(metrics["tile_size"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
            "accumulation_precision": "fp32",
        }
        trial_notes: list[str] = ["accumulation_precision=fp32"]
        if "device_name" in metrics:
            trial_notes.append(f"device={metrics['device_name']}")
        if "adapter_kind" in metrics:
            trial_notes.append(f"adapter_preference={metrics['adapter_kind']}")
        if "kernel_layout" in metrics:
            trial_notes.append(f"kernel_layout={metrics['kernel_layout']}")
        if "static_input_heap" in metrics:
            trial_notes.append(f"static_input_heap={metrics['static_input_heap']}")
        if "dispatches_per_repeat" in metrics:
            trial_notes.append(f"dispatches_per_repeat={metrics['dispatches_per_repeat']}")
        if "setup_wall_clock_latency_seconds" in metrics:
            notes.append(
                "DX12 setup_wall_clock_latency_seconds="
                f"{float(metrics['setup_wall_clock_latency_seconds']):.6f}"
            )
        if "static_upload_wall_clock_latency_seconds" in metrics:
            notes.append(
                "DX12 static_upload_wall_clock_latency_seconds="
                f"{float(metrics['static_upload_wall_clock_latency_seconds']):.6f}"
            )

        autotune_trial = TrialRecord(
            backend=self.name,
            config=autotune_config,
            wall_clock_latency_seconds=float(autotune_metrics["autotune_wall_clock_latency_seconds"]),
            effective_gflops=float(autotune_metrics["autotune_effective_gflops"]),
            checksum=str(autotune_metrics["autotune_checksum"]),
            score=autotune_score,
            notes=list(trial_notes),
        )
        trial = TrialRecord(
            backend=self.name,
            config=measurement_config,
            wall_clock_latency_seconds=float(metrics["measurement_wall_clock_latency_seconds"]),
            effective_gflops=float(metrics["measurement_effective_gflops"]),
            checksum=str(metrics["measurement_checksum"]),
            score=measurement_score,
            notes=list(trial_notes),
        )
        raw_report = {
            "autotune": _extract_raw_report(autotune_metrics),
            "measurement": _extract_raw_report(metrics),
        }
        return BackendResult(
            backend=self.name,
            available=True,
            selected_config=dict(measurement_config),
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
        phase: str,
        thread_group_sizes: list[int],
        tile_sizes: list[int],
        autotune_repeats: int,
        measurement_repeats: int,
        timeout_seconds: float,
        verbose: bool = False,
    ) -> dict[str, object]:
        """Invoke the native DX12 runner and parse its JSON metrics.

        Args:
            executable_path: Runner binary to execute.
            spec: Benchmark spec being measured.
            dataset: Dataset layout read by the runner.
            phase: Human-readable benchmark phase label.
            thread_group_sizes: DX12 thread-group-size candidates to sweep.
            tile_sizes: Tile-size candidates to sweep.
            autotune_repeats: Repeat count for autotune trials.
            measurement_repeats: Repeat count for the selected configuration.
            timeout_seconds: Subprocess timeout for the native runner.

        Returns:
            The parsed JSON metrics emitted by the native runner.
        """
        command = [
            str(executable_path),
            "--input",
            _relative_cli_path(dataset.input_path),
            "--weight",
            _relative_cli_path(dataset.weight_path),
            "--h",
            str(spec.h),
            "--w",
            str(spec.w),
            "--cin",
            str(spec.c_in),
            "--cout",
            str(spec.c_out),
            "--k",
            str(spec.k),
            "--pad",
            str(spec.pad),
            "--stride",
            str(spec.stride),
            "--thread-group-sizes",
            ",".join(str(value) for value in thread_group_sizes),
            "--tile-sizes",
            ",".join(str(value) for value in tile_sizes),
            "--autotune-repeats",
            str(autotune_repeats),
            "--measurement-repeats",
            str(measurement_repeats),
        ]
        if verbose:
            command.append("--verbose")
        emit_status(
            "method.conv2d.backend.native_runner.start",
            status="running",
            method="conv2d",
            backend=self.name,
            phase=phase,
            command=command,
            timeout_seconds=timeout_seconds,
            autotune_repeats=autotune_repeats,
            measurement_repeats=measurement_repeats,
            candidate_thread_group_sizes=thread_group_sizes,
            candidate_tile_sizes=tile_sizes,
        )
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=ROOT_DIR,
            )
        except OSError:
            emit_status(
                "method.conv2d.backend.native_runner.error",
                status="failed",
                method="conv2d",
                backend=self.name,
                phase=phase,
                command=command,
            )
            raise

        stderr_chunks: list[str] = []
        stdout_chunks: list[str] = []

        def _pump_stderr() -> None:
            for line in process.stderr:
                stderr_chunks.append(line)
                if verbose:
                    sys.stderr.write(line)
                    sys.stderr.flush()

        def _pump_stdout() -> None:
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
            stderr_pump.join(timeout=5.0)
            stdout_pump.join(timeout=5.0)
            emit_status(
                "method.conv2d.backend.native_runner.timeout",
                status="failed",
                method="conv2d",
                backend=self.name,
                phase=phase,
                command=command,
                timeout_seconds=timeout_seconds,
            )
            raise

        stderr_pump.join(timeout=5.0)
        stdout_pump.join(timeout=5.0)
        stdout_data = "".join(stdout_chunks)
        stderr_data = "".join(stderr_chunks)

        if return_code != 0:
            emit_status(
                "method.conv2d.backend.native_runner.error",
                status="failed",
                method="conv2d",
                backend=self.name,
                phase=phase,
                command=command,
                returncode=return_code,
                stderr_tail=_tail_text(stderr_data),
                stdout_tail=_tail_text(stdout_data),
            )
            raise subprocess.CalledProcessError(
                return_code, command, output=stdout_data, stderr=stderr_data
            )

        emit_status(
            "method.conv2d.backend.native_runner.complete",
            status="running",
            method="conv2d",
            backend=self.name,
            phase=phase,
            command=command,
        )
        return json.loads(stdout_data)

    def _validate_metrics(self, metrics: dict[str, object]) -> str | None:
        """Validate the required numeric fields returned by the DX12 runner."""
        required_keys = (
            "autotune_wall_clock_latency_seconds",
            "measurement_wall_clock_latency_seconds",
            "autotune_effective_gflops",
            "measurement_effective_gflops",
            "autotune_checksum",
            "measurement_checksum",
            "thread_group_size",
            "tile_size",
        )
        for key in required_keys:
            if key not in metrics:
                return f"DX12 runner output is missing required key {key!r}."

        autotune_seconds = float(metrics["autotune_wall_clock_latency_seconds"])
        measurement_seconds = float(metrics["measurement_wall_clock_latency_seconds"])
        autotune_gflops = float(metrics["autotune_effective_gflops"])
        measurement_gflops = float(metrics["measurement_effective_gflops"])
        measurement_checksum = str(metrics["measurement_checksum"])

        if autotune_seconds <= 0.0 or measurement_seconds <= 0.0:
            return "DX12 runner reported a non-positive runtime."
        if not math.isfinite(autotune_gflops) or not math.isfinite(measurement_gflops):
            return "DX12 runner reported a non-finite GFLOPS value."
        if autotune_gflops <= 0.0 or measurement_gflops <= 0.0:
            return "DX12 runner reported a non-positive GFLOPS value."
        if measurement_checksum == "chk_0":
            return "DX12 runner produced an all-zero checksum on the benchmark dataset."
        return None

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> Path:
        """Ensure the native DX12 runner exists and return its path."""
        executable_path, _note = self._resolve_executable_path(force_rebuild=force_rebuild)
        return executable_path

    def _resolve_executable_path(self, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Resolve or build the DX12 runner executable for this machine."""
        DX12_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        build_inputs = [DX12_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()

        if force_rebuild:
            if not toolchain_available:
                raise FileNotFoundError(
                    "force rebuild requested for the DX12 runner, but the local MSVC build environment "
                    f"is unavailable ({toolchain_message})"
                )
        elif DX12_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(DX12_EXECUTABLE_PATH, build_inputs):
                return (
                    DX12_EXECUTABLE_PATH,
                    f"using prebuilt Windows DX12 runner at {_relative_project_path(DX12_EXECUTABLE_PATH)}",
                )

            if not toolchain_available:
                return (
                    DX12_EXECUTABLE_PATH,
                    f"using prebuilt Windows DX12 runner at {_relative_project_path(DX12_EXECUTABLE_PATH)} because "
                    f"the source or build recipe is newer but the local MSVC build environment is unavailable "
                    f"({toolchain_message})",
                )

        if not toolchain_available:
            raise FileNotFoundError(toolchain_message)

        self._compile_runner()
        if not DX12_EXECUTABLE_PATH.exists():
            raise FileNotFoundError(
                f"DX12 runner build completed without producing {_relative_project_path(DX12_EXECUTABLE_PATH)}"
            )

        return (
            DX12_EXECUTABLE_PATH,
            f"compiled Windows DX12 runner from {_relative_project_path(DX12_SOURCE_PATH)}"
            + (" because rebuild was explicitly requested" if force_rebuild else ""),
        )

    def _compile_runner(self) -> None:
        """Compile the DX12 runner through a generated Windows batch script."""
        if self._find_vsdevcmd() is None:
            raise FileNotFoundError("VsDevCmd.bat was not found")

        compile_script_path = DX12_BUILD_DIR / "build_conv2d_dx12_runner.cmd"
        compile_script_path.write_text(
            "\n".join(
                [
                    "@echo off",
                    *_windows_vsdevcmd_setup_lines(),
                    "pushd \"%~dp0\"",
                    (
                        "cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc "
                        f"/Fe:{DX12_EXECUTABLE_PATH.name} ..\\{DX12_SOURCE_PATH.name} "
                        "d3d12.lib dxgi.lib d3dcompiler.lib dxguid.lib"
                    ),
                    "set \"BUILD_EXIT=%ERRORLEVEL%\"",
                    "popd",
                    "exit /b %BUILD_EXIT%",
                ]
            )
            + "\n",
            encoding="ascii",
        )

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

    def _toolchain_status(self) -> tuple[bool, str]:
        """Return whether the local MSVC build environment is available."""
        if self._find_vsdevcmd() is None:
            return False, "VsDevCmd.bat was not found; MSVC build environment is unavailable."
        return True, "VsDevCmd.bat is available."

    @staticmethod
    def _find_vsdevcmd() -> Path | None:
        """Locate the Visual Studio developer-command script on Windows."""
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
