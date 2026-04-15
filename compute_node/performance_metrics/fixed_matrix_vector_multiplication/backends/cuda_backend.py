"""CUDA backend for the fixed matrix-vector benchmark.

This backend mirrors the CPU design:

- Python handles orchestration, compilation, and result parsing
- the CUDA executable handles loading the dataset once and sweeping multiple
  kernel configurations in memory

That split keeps the expensive dataset I/O out of the tuning loop and makes the
top-level benchmark code stay hardware-agnostic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from compute_node.compute_methods.fixed_matrix_vector_multiplication import (
    CUDA_BUILD_DIR,
    CUDA_DIR,
    CUDA_EXECUTABLE_PATH,
    CUDA_SOURCE_PATH,
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
    detect_nvidia_windows_adapter,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.scoring import linear_time_score
from compute_node.performance_metrics.path_utils import sanitize_text, to_relative_cli_path, to_relative_string

ROOT_DIR = Path(__file__).resolve().parents[1]
WINDOWS_PREBUILT_SMS = ("75", "80", "86", "89", "90", "120")
WINDOWS_CUDA_SELF_CONTAINED_NOTE = (
    "Windows CUDA runner is packaged as a self-contained executable: it statically links cudart and the MSVC runtime, "
    "so runtime needs only a compatible NVIDIA driver that provides nvcuda.dll."
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


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a built artifact predates any source or build-recipe input."""

    if not binary_path.exists():
        return True

    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


def _format_windows_sm_targets(targets: tuple[str, ...] | list[str] | None = None) -> str:
    """Render the baked-in Windows CUDA architectures for human-readable notes."""

    resolved_targets = WINDOWS_PREBUILT_SMS if targets is None else targets
    return ", ".join(f"sm_{value}" for value in resolved_targets)


def _supports_windows_prebuilt_capability(capability: str | None) -> bool:
    """Return whether the shipped Windows CUDA runner should cover one GPU."""

    return capability is None or capability in WINDOWS_PREBUILT_SMS


def _windows_gencode_args(capability: str | None) -> list[str]:
    """Return the fat-binary architecture list for the shipped Windows runner."""

    target_sms = _windows_compile_sm_targets(capability)

    args: list[str] = []
    for sm in target_sms:
        args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    return args


def _detect_nvcc_supported_sms() -> set[str] | None:
    """Ask nvcc which `sm_XX` codes it can compile for on this machine."""

    if shutil.which("nvcc") is None:
        return None

    completed = subprocess.run(
        ["nvcc", "--list-gpu-code"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None

    supported: set[str] = set()
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("sm_"):
            digits = stripped.removeprefix("sm_")
            if digits.isdigit():
                supported.add(digits)
    return supported or None


def _windows_compile_sm_targets(capability: str | None) -> list[str]:
    """Return the Windows SM targets that this machine should compile today.

    We want the checked-in runner to cover a broad set of GPUs, including
    `sm_120`, but we also do not want rebuilds to fail on an older local CUDA
    toolchain that cannot emit every requested architecture.
    """

    target_sms: list[str] = list(WINDOWS_PREBUILT_SMS)
    if capability is not None and capability not in target_sms:
        target_sms.append(capability)

    supported_sms = _detect_nvcc_supported_sms()
    if supported_sms is None:
        return target_sms

    filtered_targets = [sm for sm in target_sms if sm in supported_sms]
    if capability is not None and capability in supported_sms and capability not in filtered_targets:
        filtered_targets.append(capability)
    return filtered_targets or target_sms


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


def _candidate_block_sizes() -> list[int]:
    """Return the CUDA block sizes to sweep."""

    return [64, 128, 256, 512]


def _candidate_tile_sizes() -> list[int]:
    """Return the template tile sizes supported by the CUDA kernels."""

    return [1, 2, 4, 8]


def _candidate_transpose_modes() -> list[int]:
    """Try both the row-major input layout and a transposed GPU layout."""

    return [0, 1]


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed autotune repeat count for every CUDA benchmark."""

    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed sustained-measurement repeat count for every CUDA benchmark."""

    return DEFAULT_MEASUREMENT_REPEATS


def _detect_compute_capability() -> str | None:
    """Ask `nvidia-smi` for the first GPU's compute capability."""

    if shutil.which("nvidia-smi") is None:
        return None

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digits = stripped.replace(".", "")
        if digits.isdigit():
            return digits
    return None


class CudaBackend:
    """Compile and invoke the CUDA autotuning runner."""

    name = "cuda"

    def probe(self) -> tuple[bool, str]:
        """Check whether the current machine looks CUDA-capable."""

        if not CUDA_SOURCE_PATH.exists():
            return False, f"missing CUDA runner source at {_relative_project_path(CUDA_SOURCE_PATH)}"

        nvidia_adapter_name: str | None = None
        if os.name == "nt":
            nvidia_adapter_name, nvidia_adapter_message = detect_nvidia_windows_adapter()
            if nvidia_adapter_name is None:
                return False, nvidia_adapter_message

        capability = _detect_compute_capability()
        build_inputs = [CUDA_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()
        binary_is_stale = _binary_is_stale(CUDA_EXECUTABLE_PATH, build_inputs)

        if CUDA_EXECUTABLE_PATH.exists():
            if os.name == "nt" and not _supports_windows_prebuilt_capability(capability):
                if toolchain_available:
                    return (
                        True,
                        f"Detected GPU is sm_{capability}, which is outside the checked-in Windows CUDA runner target "
                        f"set ({_format_windows_sm_targets()}). The local CUDA toolchain is available, so the runner "
                        "will be rebuilt for this GPU.",
                    )
                return (
                    False,
                    f"Detected GPU is sm_{capability}, but the checked-in Windows CUDA runner only contains "
                    f"{_format_windows_sm_targets()} and the local CUDA toolchain is unavailable ({toolchain_message}).",
                )

            if not binary_is_stale:
                if os.name == "nt":
                    if capability is None:
                        return (
                            True,
                            f"CUDA backend available via self-contained Windows runner at "
                            f"{_relative_project_path(CUDA_EXECUTABLE_PATH)}. It targets {_format_windows_sm_targets()} "
                            "and needs only the NVIDIA driver at runtime.",
                        )
                    return (
                        True,
                        f"CUDA backend available via self-contained Windows runner at "
                        f"{_relative_project_path(CUDA_EXECUTABLE_PATH)}. Detected GPU is sm_{capability}; the runner "
                        f"targets {_format_windows_sm_targets()} and needs only the NVIDIA driver at runtime. "
                        f"Matched display adapter {nvidia_adapter_name!r}.",
                    )

                if capability is None:
                    return (
                        True,
                        f"CUDA backend available via prebuilt binary at {_relative_project_path(CUDA_EXECUTABLE_PATH)}. "
                        "Compute capability could not be queried.",
                    )
                return (
                    True,
                    f"CUDA backend available via prebuilt binary at {_relative_project_path(CUDA_EXECUTABLE_PATH)}. "
                    f"Detected GPU is sm_{capability}.",
                )

            if not toolchain_available:
                if os.name == "nt":
                    return (
                        True,
                        f"CUDA backend available via self-contained Windows runner at "
                        f"{_relative_project_path(CUDA_EXECUTABLE_PATH)}. The source or build recipe is newer, but the "
                        f"local CUDA toolchain is unavailable ({toolchain_message}), so the existing runner will be "
                        "used.",
                    )
                return (
                    True,
                    f"CUDA backend available via prebuilt binary at {_relative_project_path(CUDA_EXECUTABLE_PATH)}. "
                    f"The source or build recipe is newer, but the local CUDA toolchain is unavailable "
                    f"({toolchain_message}), so the existing binary will be used.",
                )

            if capability is None:
                return (
                    True,
                    f"CUDA backend available. Existing binary at {_relative_project_path(CUDA_EXECUTABLE_PATH)} is older "
                    f"than the source or build recipe, so {_relative_project_path(CUDA_SOURCE_PATH)} will be rebuilt.",
                )
            return (
                True,
                f"CUDA backend available. Existing binary at {_relative_project_path(CUDA_EXECUTABLE_PATH)} is older "
                f"than the source or build recipe, so {_relative_project_path(CUDA_SOURCE_PATH)} will be rebuilt for "
                f"detected GPU sm_{capability}.",
            )

        if not toolchain_available:
            return False, toolchain_message

        if os.name == "nt":
            if capability is None:
                return (
                    True,
                    "CUDA toolchain detected on Windows. Binary is missing, so a self-contained runner with a static "
                    f"CUDA runtime will be compiled for {_format_windows_sm_targets()}.",
                )
            return (
                True,
                f"CUDA toolchain detected on Windows. Binary is missing, so a self-contained runner with a static "
                f"CUDA runtime will be compiled for {_format_windows_sm_targets()} and detected GPU sm_{capability}.",
            )

        if capability is None:
            return True, "nvcc detected on PATH, but compute capability could not be queried."
        return True, f"nvcc detected on PATH; first GPU compute capability is sm_{capability}."

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
        """Run the CUDA executable once and return the best configuration it found."""

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
            if os.name == "nt":
                notes.append(WINDOWS_CUDA_SELF_CONTAINED_NOTE)
        except (OSError, subprocess.CalledProcessError) as exc:
            details = ""
            if isinstance(exc, subprocess.CalledProcessError):
                details = (exc.stderr or exc.stdout or "").strip()
            notes.append(_sanitize_note(f"failed to prepare CUDA runner: {details or exc}"))
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
        block_sizes = _candidate_block_sizes()
        tile_sizes = _candidate_tile_sizes()
        transpose_modes = _candidate_transpose_modes()
        autotune_repeats = _autotune_repeats(spec)
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else _measurement_repeats(spec)
        )

        notes.append(f"transpose search order: {transpose_modes}")
        notes.append(f"block size search order: {block_sizes}")
        notes.append(f"tile size search order: {tile_sizes}")
        notes.append(f"autotune_repeats_per_config: {autotune_repeats}")
        notes.append(f"measurement_repeats_for_best_config: {final_measurement_repeats}")

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                transpose_modes=transpose_modes,
                block_sizes=block_sizes,
                tile_sizes=tile_sizes,
                autotune_repeats=autotune_repeats,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except subprocess.TimeoutExpired:
            notes.append("CUDA benchmark timed out")
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

        selected_transpose = int(autotune_metrics["transpose"])
        selected_block_size = int(autotune_metrics["block_size"])
        selected_tile_size = int(autotune_metrics["tile_size"])

        try:
            metrics = self._run_runner(
                executable_path,
                measurement_spec,
                measurement_dataset,
                transpose_modes=[selected_transpose],
                block_sizes=[selected_block_size],
                tile_sizes=[selected_tile_size],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except subprocess.TimeoutExpired:
            notes.append("CUDA benchmark timed out")
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
            "transpose": bool(autotune_metrics["transpose"]),
            "block_size": int(autotune_metrics["block_size"]),
            "tile_size": int(autotune_metrics["tile_size"]),
            "accumulation_precision": str(autotune_metrics.get("accumulation_precision") or spec.accumulation_precision),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        config = {
            "transpose": bool(metrics["transpose"]),
            "block_size": int(metrics["block_size"]),
            "tile_size": int(metrics["tile_size"]),
            "accumulation_precision": str(metrics.get("accumulation_precision") or measurement_spec.accumulation_precision),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        trial_notes: list[str] = [f"accumulation_precision={config['accumulation_precision']}"]
        if "device_name" in metrics:
            trial_notes.append(f"device={metrics['device_name']}")
        if "compute_capability" in metrics:
            trial_notes.append(f"sm={metrics['compute_capability']}")

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
        transpose_modes: list[int],
        block_sizes: list[int],
        tile_sizes: list[int],
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
            "--accumulation-precision",
            spec.accumulation_precision,
            "--transpose-modes",
            ",".join(str(value) for value in transpose_modes),
            "--block-sizes",
            ",".join(str(value) for value in block_sizes),
            "--tile-sizes",
            ",".join(str(value) for value in tile_sizes),
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
        """Return the CUDA runner path, compiling only when needed."""

        executable_path, _note = self._resolve_executable_path(force_rebuild=force_rebuild)
        return executable_path

    def _resolve_executable_path(self, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Prefer the current OS binary and fall back to compiling when possible."""

        CUDA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        build_inputs = [CUDA_SOURCE_PATH, Path(__file__)]
        capability = _detect_compute_capability()
        toolchain_available, toolchain_message = self._toolchain_status()

        if force_rebuild:
            if not toolchain_available:
                raise FileNotFoundError(
                    f"force rebuild requested for the CUDA runner, but the local CUDA toolchain is unavailable "
                    f"({toolchain_message})"
                )
        elif CUDA_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(CUDA_EXECUTABLE_PATH, build_inputs):
                if os.name == "nt":
                    return (
                        CUDA_EXECUTABLE_PATH,
                        f"using prebuilt self-contained Windows CUDA runner at "
                        f"{_relative_project_path(CUDA_EXECUTABLE_PATH)} targeting {_format_windows_sm_targets()}",
                    )
                return (
                    CUDA_EXECUTABLE_PATH,
                    f"using prebuilt CUDA runner at {_relative_project_path(CUDA_EXECUTABLE_PATH)}",
                )

            if not toolchain_available:
                if os.name == "nt":
                    if not _supports_windows_prebuilt_capability(capability):
                        raise FileNotFoundError(
                            f"detected GPU is sm_{capability}, but the checked-in Windows CUDA runner only contains "
                            f"{_format_windows_sm_targets()} and the local CUDA toolchain is unavailable "
                            f"({toolchain_message})"
                        )
                    return (
                        CUDA_EXECUTABLE_PATH,
                        f"using prebuilt self-contained Windows CUDA runner at "
                        f"{_relative_project_path(CUDA_EXECUTABLE_PATH)} because the source or build recipe is newer "
                        f"but the local CUDA toolchain is unavailable ({toolchain_message})",
                    )

                return (
                    CUDA_EXECUTABLE_PATH,
                    f"using prebuilt CUDA runner at {_relative_project_path(CUDA_EXECUTABLE_PATH)} because the source "
                    f"or build recipe is newer but the local CUDA toolchain is unavailable ({toolchain_message})",
                )

        if not toolchain_available:
            raise FileNotFoundError(toolchain_message)

        compiled_targets = self._compile_runner(capability)

        if not CUDA_EXECUTABLE_PATH.exists():
            raise FileNotFoundError(
                f"CUDA runner build completed without producing {_relative_project_path(CUDA_EXECUTABLE_PATH)}"
            )

        if os.name == "nt":
            if capability is None:
                return (
                    CUDA_EXECUTABLE_PATH,
                    f"compiled self-contained Windows CUDA runner from {_relative_project_path(CUDA_SOURCE_PATH)} "
                    f"targeting {_format_windows_sm_targets(compiled_targets)}"
                    + (" because rebuild was explicitly requested" if force_rebuild else ""),
                )
            return (
                CUDA_EXECUTABLE_PATH,
                f"compiled self-contained Windows CUDA runner from {_relative_project_path(CUDA_SOURCE_PATH)} "
                f"targeting {_format_windows_sm_targets(compiled_targets)} and detected GPU sm_{capability}"
                + (" because rebuild was explicitly requested" if force_rebuild else ""),
            )

        if capability is None:
            return (
                CUDA_EXECUTABLE_PATH,
                f"compiled CUDA runner from {_relative_project_path(CUDA_SOURCE_PATH)}"
                + (" because rebuild was explicitly requested" if force_rebuild else ""),
            )
        return (
            CUDA_EXECUTABLE_PATH,
            f"compiled CUDA runner from {_relative_project_path(CUDA_SOURCE_PATH)} for detected GPU sm_{capability}"
            + (" because rebuild was explicitly requested" if force_rebuild else ""),
        )

    def _compile_runner(self, capability: str | None) -> list[str]:
        """Build the CUDA runner for the current platform."""

        if shutil.which("nvcc") is None:
            raise FileNotFoundError("nvcc was not found")

        if os.name == "nt":
            vsdevcmd_path = self._find_vsdevcmd()
            if vsdevcmd_path is None:
                raise FileNotFoundError("VsDevCmd.bat was not found for CUDA compilation")

            compile_targets = _windows_compile_sm_targets(capability)
            command_parts = [
                "nvcc",
                "..\\fmvm_cuda_runner.cu",
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-cudart",
                "static",
                "-Xcompiler",
                "\"/MT /EHsc\"",
                "-o",
                "fmvm_cuda_runner.exe",
                *_windows_gencode_args(capability),
            ]

            compile_script_path = CUDA_BUILD_DIR / "build_fmvm_cuda_runner.cmd"
            compile_script_path.write_text(
                "\n".join(
                    [
                        "@echo off",
                        *_windows_vsdevcmd_setup_lines(),
                        "pushd \"%~dp0\"",
                        " ".join(command_parts),
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
        else:
            compile_targets = [capability] if capability is not None else []
            command = [
                "nvcc",
                "../fmvm_cuda_runner.cu",
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-o",
                CUDA_EXECUTABLE_PATH.name,
            ]
            if capability is not None:
                command.append(f"-gencode=arch=compute_{capability},code=sm_{capability}")
            completed = subprocess.run(command, capture_output=True, text=True, cwd=CUDA_BUILD_DIR)

        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return compile_targets

    def _toolchain_status(self) -> tuple[bool, str]:
        """Report whether the local CUDA build toolchain is usable."""

        if shutil.which("nvcc") is None:
            return False, "nvcc was not found on PATH."

        if os.name == "nt":
            if self._find_vsdevcmd() is None:
                return False, "VsDevCmd.bat was not found; the MSVC build environment is unavailable."
            return True, "nvcc and VsDevCmd.bat are available."

        return True, "nvcc is available."

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
