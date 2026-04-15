"""Metal backend for the fixed matrix-vector benchmark on macOS.

This backend mirrors the CUDA design:

- Python handles orchestration, compilation, and result parsing
- the Metal executable handles loading the dataset once and sweeping multiple
  kernel configurations in memory

That split keeps the expensive dataset I/O out of the tuning loop and makes the
top-level benchmark code stay hardware-agnostic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from compute_node.compute_methods.fixed_matrix_vector_multiplication import (
    METAL_AIR_PATH,
    METAL_BUILD_DIR,
    METAL_DIR,
    METAL_EXECUTABLE_PATH,
    METAL_HOST_SOURCE_PATH,
    METAL_KERNEL_SOURCE_PATH,
    METAL_LIBRARY_PATH,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.performance_metrics.fixed_matrix_vector_multiplication.scoring import linear_time_score
from compute_node.performance_metrics.path_utils import sanitize_text, to_relative_cli_path, to_relative_string

ROOT_DIR = Path(__file__).resolve().parents[1]


def _relative_project_path(path: Path) -> str:
    """Show a project-local path without an absolute prefix."""

    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    """Remove project/home absolute prefixes from backend notes."""

    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    """Render a relative path for subprocess calls."""

    return to_relative_cli_path(path, start=ROOT_DIR)


def _candidate_block_sizes() -> list[int]:
    """Return the Metal threadgroup sizes to sweep."""

    return [32, 64, 128, 256, 512, 1024]


def _candidate_tile_sizes() -> list[int]:
    """Return the unroll factors supported by the Metal kernel."""

    return [1, 2, 4, 8, 16]


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed autotune repeat count for every Metal benchmark."""

    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed sustained-measurement repeat count for every Metal benchmark."""

    return DEFAULT_MEASUREMENT_REPEATS


def _find_xcrun_tool(tool_name: str) -> str | None:
    """Resolve one Apple developer tool through `xcrun`."""

    completed = subprocess.run(
        ["xcrun", "--find", tool_name],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    resolved = completed.stdout.strip()
    return resolved or None


def _sources_are_stale(outputs: list[Path], sources: list[Path]) -> bool:
    """Return whether any output is missing or older than any source."""

    if any(not output.exists() for output in outputs):
        return True

    newest_source = max(source.stat().st_mtime for source in sources)
    oldest_output = min(output.stat().st_mtime for output in outputs)
    return oldest_output < newest_source


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a built artifact predates any source or build-recipe input."""

    if not binary_path.exists():
        return True

    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


class MetalBackend:
    """Compile and invoke the Metal autotuning runner."""

    name = "metal"

    def probe(self) -> tuple[bool, str]:
        """Check whether the current machine looks Metal-capable."""

        if sys.platform != "darwin":
            return False, "Metal backend is only available on macOS."
        if not METAL_HOST_SOURCE_PATH.exists():
            return False, f"missing Metal host runner source at {_relative_project_path(METAL_HOST_SOURCE_PATH)}"
        if not METAL_KERNEL_SOURCE_PATH.exists():
            return False, f"missing Metal kernel source at {_relative_project_path(METAL_KERNEL_SOURCE_PATH)}"

        build_inputs = [METAL_HOST_SOURCE_PATH, METAL_KERNEL_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()

        if METAL_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(METAL_EXECUTABLE_PATH, build_inputs):
                return (
                    True,
                    f"Metal backend available via self-contained runner at "
                    f"{_relative_project_path(METAL_EXECUTABLE_PATH)}. No local toolchain is required at runtime.",
                )

            if not toolchain_available:
                return (
                    True,
                    f"Metal backend available via self-contained runner at "
                    f"{_relative_project_path(METAL_EXECUTABLE_PATH)}. The source or build recipe is newer, but the "
                    f"local Metal toolchain is unavailable ({toolchain_message}), so the existing runner will be used.",
                )

            return (
                True,
                f"Metal backend available. Existing self-contained runner at "
                f"{_relative_project_path(METAL_EXECUTABLE_PATH)} is older than the source or build recipe, so "
                f"{_relative_project_path(METAL_HOST_SOURCE_PATH)} will be rebuilt with an embedded metallib.",
            )

        if not toolchain_available:
            return False, toolchain_message
        return (
            True,
            "Metal toolchain detected via xcrun; binary is missing, so a self-contained Metal runner with an "
            "embedded metallib will be compiled.",
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
        """Run the Metal executable once and return the best configuration it found."""

        if spec.accumulation_precision != "fp32":
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=[
                    f"Metal backend currently supports only fp32 accumulation; requested {spec.accumulation_precision}."
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
            executable_path, build_note = self._compile_if_needed(force_rebuild=force_rebuild)
            notes.append(build_note)
            notes.append(self._runtime_note(executable_path))
        except (OSError, subprocess.CalledProcessError) as exc:
            details = ""
            if isinstance(exc, subprocess.CalledProcessError):
                details = (exc.stderr or exc.stdout or "").strip()
            notes.append(_sanitize_note(f"failed to compile Metal runner: {details or exc}"))
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
        autotune_repeats = _autotune_repeats(spec)
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else _measurement_repeats(spec)
        )

        notes.append(f"block size search order: {block_sizes}")
        notes.append(f"tile size search order: {tile_sizes}")
        notes.append(f"autotune_repeats_per_config: {autotune_repeats}")
        notes.append(f"measurement_repeats_for_best_config: {final_measurement_repeats}")

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                block_sizes=block_sizes,
                tile_sizes=tile_sizes,
                autotune_repeats=autotune_repeats,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except subprocess.TimeoutExpired:
            notes.append("Metal benchmark timed out")
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
                block_sizes=[int(autotune_metrics["block_size"])],
                tile_sizes=[int(autotune_metrics["tile_size"])],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
            )
        except subprocess.TimeoutExpired:
            notes.append("Metal benchmark timed out")
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
            "block_size": int(autotune_metrics["block_size"]),
            "tile_size": int(autotune_metrics["tile_size"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        config = {
            "block_size": int(metrics["block_size"]),
            "tile_size": int(metrics["tile_size"]),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        trial_notes: list[str] = []
        if "device_name" in metrics:
            trial_notes.append(f"device={metrics['device_name']}")
        if "thread_execution_width" in metrics:
            trial_notes.append(f"thread_execution_width={metrics['thread_execution_width']}")
        if "max_total_threads_per_threadgroup" in metrics:
            trial_notes.append(f"max_threads_per_group={metrics['max_total_threads_per_threadgroup']}")

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

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Build the self-contained Metal runner when sources changed."""

        METAL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        build_inputs = [METAL_HOST_SOURCE_PATH, METAL_KERNEL_SOURCE_PATH, Path(__file__)]
        if not force_rebuild and METAL_EXECUTABLE_PATH.exists() and not _binary_is_stale(METAL_EXECUTABLE_PATH, build_inputs):
            return (
                METAL_EXECUTABLE_PATH,
                f"using prebuilt self-contained Metal runner at {_relative_project_path(METAL_EXECUTABLE_PATH)}",
            )

        toolchain_available, toolchain_message = self._toolchain_status()
        if not toolchain_available:
            if not force_rebuild and METAL_EXECUTABLE_PATH.exists():
                return (
                    METAL_EXECUTABLE_PATH,
                    f"using prebuilt self-contained Metal runner at {_relative_project_path(METAL_EXECUTABLE_PATH)} "
                    f"because the source or build recipe is newer but the local Metal toolchain is unavailable "
                    f"({toolchain_message})",
                )
            raise FileNotFoundError(toolchain_message)

        subprocess.run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metal",
                "-c",
                "../fmvm_metal_kernels.metal",
                "-o",
                "fmvm_metal_kernels.air",
            ],
            cwd=METAL_BUILD_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metallib",
                "fmvm_metal_kernels.air",
                "-o",
                "fmvm_metal_kernels.metallib",
            ],
            cwd=METAL_BUILD_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "clang++",
                "../fmvm_metal_runner.mm",
                "-std=c++20",
                "-O3",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-Wl,-sectcreate,__DATA,__metallib,fmvm_metal_kernels.metallib",
                "-o",
                "fmvm_metal_runner",
            ],
            cwd=METAL_BUILD_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return (
            METAL_EXECUTABLE_PATH,
            f"compiled self-contained Metal runner from {_relative_project_path(METAL_HOST_SOURCE_PATH)} and "
            f"{_relative_project_path(METAL_KERNEL_SOURCE_PATH)} with an embedded metallib"
            + (" because rebuild was explicitly requested" if force_rebuild else ""),
        )

    def _toolchain_status(self) -> tuple[bool, str]:
        """Report whether the local Metal compiler toolchain is available."""

        missing_tools: list[str] = []
        for tool_name in ("metal", "metallib", "clang++"):
            if _find_xcrun_tool(tool_name) is None:
                missing_tools.append(tool_name)
        if missing_tools:
            return False, f"Metal toolchain is incomplete; missing tools: {', '.join(missing_tools)}."
        return True, "Metal toolchain detected via xcrun."

    def _runtime_note(self, executable_path: Path) -> str:
        """Describe the runtime dependencies of the self-contained Metal runner."""

        completed = subprocess.run(
            ["otool", "-L", str(executable_path)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return (
                "self-contained Metal runner embeds its metallib and no longer needs a sidecar kernel file at "
                "runtime."
            )

        dynamic_libraries: list[str] = []
        for line in completed.stdout.splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            library_path = stripped.split(" (compatibility version", 1)[0].strip()
            dynamic_libraries.append(Path(library_path).name or library_path)

        if not dynamic_libraries:
            return "self-contained Metal runner embeds its metallib and reported no explicit framework dependencies."

        return (
            "self-contained Metal runner embeds its metallib and depends only on macOS system frameworks: "
            + ", ".join(dynamic_libraries)
        )
