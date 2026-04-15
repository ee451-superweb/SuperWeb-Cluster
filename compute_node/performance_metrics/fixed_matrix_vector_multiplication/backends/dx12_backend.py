"""DirectX 12 backend for the fixed matrix-vector benchmark on Windows.

This backend is the first Windows-native GPU path that does not depend on
CUDA. It is aimed primarily at integrated GPUs such as Radeon 780M, so the
native runner prefers the minimum-power DXGI adapter instead of the
high-performance adapter.

The Python layer keeps the same responsibilities as the CPU/CUDA/Metal
backends:

- prepare or rebuild the native runner when needed
- invoke it with a small autotune surface
- parse the JSON metrics into the shared benchmark model
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from compute_node.compute_methods.fixed_matrix_vector_multiplication import (
    DX12_BUILD_DIR,
    DX12_EXECUTABLE_PATH,
    DX12_SOURCE_PATH,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.backends.windows_gpu_inventory import (
    detect_non_nvidia_windows_adapter,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.scoring import linear_time_score
from compute_node.performance_metrics.path_utils import sanitize_text, to_relative_cli_path, to_relative_string

ROOT_DIR = Path(__file__).resolve().parents[1]
WINDOWS_DX12_RUNTIME_NOTE = (
    "Windows DX12 runner uses the system Direct3D 12 runtime and the installed graphics driver; "
    "it does not require CUDA, ROCm, or the Windows SDK at runtime."
)


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


def _candidate_thread_group_sizes() -> list[int]:
    """Return the DX12 thread-group sizes to sweep."""

    return [256, 512]


def _candidate_rows_per_thread() -> list[int]:
    """Return how many output rows one thread should compute."""

    return [1, 2]


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the autotune repeat count for the DX12 benchmark.

    The Radeon 780M path showed noticeably more run-to-run noise than CUDA
    during the very short 3-repeat surface search, so DX12 uses a slightly
    longer short-run phase to avoid selecting a config that only wins on one
    lucky burst.
    """

    return max(DEFAULT_AUTOTUNE_REPEATS, 8)


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed sustained-measurement repeat count for every DX12 benchmark."""

    return DEFAULT_MEASUREMENT_REPEATS


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a built artifact predates any source or build-recipe input."""

    if not binary_path.exists():
        return True

    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


def _detect_non_nvidia_windows_adapter() -> tuple[str | None, str]:
    """Keep the old local helper name as a small wrapper around shared routing."""

    return detect_non_nvidia_windows_adapter()


class Dx12Backend:
    """Compile and invoke the DX12 autotuning runner on Windows."""

    name = "dx12"

    def probe(self) -> tuple[bool, str]:
        """Check whether this machine looks capable of running the DX12 backend."""

        if os.name != "nt":
            return False, "DX12 backend is only available on Windows."
        if not DX12_SOURCE_PATH.exists():
            return False, f"missing DX12 runner source at {_relative_project_path(DX12_SOURCE_PATH)}"
        adapter_name, adapter_message = _detect_non_nvidia_windows_adapter()
        if adapter_name is None:
            return False, adapter_message

        build_inputs = [DX12_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()

        if DX12_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(DX12_EXECUTABLE_PATH, build_inputs):
                return (
                    True,
                    f"DX12 backend available via Windows runner at {_relative_project_path(DX12_EXECUTABLE_PATH)}. "
                    f"The runtime will target the non-NVIDIA adapter {adapter_name!r} and prefers the minimum-power "
                    "D3D12 adapter so integrated GPUs can be benchmarked without CUDA.",
                )

            if not toolchain_available:
                return (
                    True,
                    f"DX12 backend available via Windows runner at {_relative_project_path(DX12_EXECUTABLE_PATH)}. "
                    f"The source or build recipe is newer, but the local MSVC toolchain is unavailable "
                    f"({toolchain_message}), so the existing runner will be used on {adapter_name!r}.",
                )

            return (
                True,
                f"DX12 backend available. Existing runner at {_relative_project_path(DX12_EXECUTABLE_PATH)} is older "
                f"than the source or build recipe, so {_relative_project_path(DX12_SOURCE_PATH)} will be rebuilt for "
                f"the non-NVIDIA adapter {adapter_name!r}.",
            )

        if not toolchain_available:
            return False, toolchain_message
        return (
            True,
            f"DX12 backend available. Binary is missing, so {_relative_project_path(DX12_SOURCE_PATH)} will be "
            f"compiled with MSVC and linked against the Windows Direct3D 12 system libraries for {adapter_name!r}.",
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
    ) -> BackendResult:
        """Run the DX12 executable once and return the best configuration it found."""

        if spec.accumulation_precision != "fp32":
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=[
                    f"DX12 backend currently supports only fp32 accumulation; requested {spec.accumulation_precision}."
                ],
            )

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
        rows_per_thread_values = _candidate_rows_per_thread()
        autotune_repeats = _autotune_repeats(spec)
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else _measurement_repeats(spec)
        )

        notes.append(f"thread-group-size search order: {thread_group_sizes}")
        notes.append(f"rows-per-thread search order: {rows_per_thread_values}")
        notes.append(f"autotune_repeats_per_config: {autotune_repeats}")
        notes.append(f"measurement_repeats_for_best_config: {final_measurement_repeats}")

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                thread_group_sizes=thread_group_sizes,
                rows_per_thread_values=rows_per_thread_values,
                autotune_repeats=autotune_repeats,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except subprocess.TimeoutExpired:
            notes.append("DX12 benchmark timed out")
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
            stdout = (exc.stdout or "").strip()
            notes.append(_sanitize_note(stderr if stderr else (stdout if stdout else str(exc))))
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
            metrics = self._run_runner(
                executable_path,
                measurement_spec,
                measurement_dataset,
                thread_group_sizes=[int(autotune_metrics["thread_group_size"])],
                rows_per_thread_values=[int(autotune_metrics["rows_per_thread"])],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except subprocess.TimeoutExpired:
            notes.append("DX12 benchmark timed out")
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
            stdout = (exc.stdout or "").strip()
            notes.append(_sanitize_note(stderr if stderr else (stdout if stdout else str(exc))))
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
        if "nvidia" in str(metrics.get("device_name") or "").lower():
            notes.append(
                "DX12 runner selected an NVIDIA adapter; this backend is intentionally reserved for non-NVIDIA GPUs."
            )
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )
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
            "rows_per_thread": int(autotune_metrics["rows_per_thread"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
            "accumulation_precision": "fp32",
        }
        config = {
            "thread_group_size": int(metrics["thread_group_size"]),
            "rows_per_thread": int(metrics["rows_per_thread"]),
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
        if "vector_upload_wall_clock_latency_seconds" in metrics:
            notes.append(
                "DX12 vector_upload_wall_clock_latency_seconds="
                f"{float(metrics['vector_upload_wall_clock_latency_seconds']):.6f}"
            )

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
        return BackendResult(
            backend=self.name,
            available=True,
            selected_config=dict(config),
            autotune_trial=autotune_trial,
            best_trial=trial,
            trials=[autotune_trial, trial],
            notes=notes,
        )

    def _run_runner(
        self,
        executable_path: Path,
        spec: BenchmarkSpec,
        dataset: DatasetLayout,
        *,
        thread_group_sizes: list[int],
        rows_per_thread_values: list[int],
        autotune_repeats: int,
        measurement_repeats: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
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
            "--thread-group-sizes",
            ",".join(str(value) for value in thread_group_sizes),
            "--rows-per-thread",
            ",".join(str(value) for value in rows_per_thread_values),
            "--autotune-repeats",
            str(autotune_repeats),
            "--measurement-repeats",
            str(measurement_repeats),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=ROOT_DIR,
        )
        return json.loads(completed.stdout)

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> Path:
        """Return the DX12 runner path, compiling only when needed."""

        executable_path, _note = self._resolve_executable_path(force_rebuild=force_rebuild)
        return executable_path

    def _resolve_executable_path(self, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Prefer the current Windows binary and fall back to compiling when possible."""

        DX12_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        build_inputs = [DX12_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()

        if force_rebuild:
            if not toolchain_available:
                raise FileNotFoundError(
                    f"force rebuild requested for the DX12 runner, but the local MSVC build environment is unavailable "
                    f"({toolchain_message})"
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
        """Build the Windows DX12 runner with MSVC."""

        if self._find_vsdevcmd() is None:
            raise FileNotFoundError("VsDevCmd.bat was not found")

        compile_script_path = DX12_BUILD_DIR / "build_fmvm_dx12_runner.cmd"
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
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    def _toolchain_status(self) -> tuple[bool, str]:
        """Report whether the local MSVC toolchain is usable."""

        if self._find_vsdevcmd() is None:
            return False, "VsDevCmd.bat was not found; MSVC build environment is unavailable."
        return True, "VsDevCmd.bat is available."

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
