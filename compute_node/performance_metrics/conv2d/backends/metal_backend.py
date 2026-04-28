"""Metal backend for the conv2d benchmark on macOS.

This mirrors the GEMV Metal backend design:

- Python owns probing, compilation, and result parsing
- the Objective-C++ runner owns dataset loading plus the autotune sweep
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from compute_node.compute_methods.conv2d import (
    METAL_BUILD_DIR,
    METAL_EXECUTABLE_PATH,
    METAL_HOST_SOURCE_PATH,
    METAL_KERNEL_SOURCE_PATH,
)
from compute_node.performance_metrics.benchmark_status import emit_status
from compute_node.performance_metrics.conv2d.models import (
    DEFAULT_AUTOTUNE_REPEATS,
    DEFAULT_MEASUREMENT_REPEATS,
    BackendResult,
    BenchmarkSpec,
    DatasetLayout,
    TrialRecord,
)
from compute_node.performance_metrics.conv2d.scoring import linear_time_score
from compute_node.performance_metrics.path_utils import sanitize_text, to_relative_cli_path, to_relative_string

ROOT_DIR = Path(__file__).resolve().parents[1]

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
    """Render a project-local path without an absolute prefix."""

    return to_relative_string(path, start=ROOT_DIR)


def _sanitize_note(text: str) -> str:
    """Strip absolute machine-specific prefixes from notes."""

    return sanitize_text(text, start=ROOT_DIR)


def _relative_cli_path(path: Path) -> str:
    """Render a path for subprocess arguments."""

    return to_relative_cli_path(path, start=ROOT_DIR)


def _candidate_block_sizes() -> list[int]:
    """Return Metal threadgroup-size candidates."""

    return [32, 64, 128, 256, 512, 1024]


def _candidate_tile_sizes() -> list[int]:
    """Return Metal threadgroup-width candidates."""

    return [4, 8, 16, 32, 64]


def _find_xcrun_tool(tool_name: str) -> str | None:
    """Resolve one Apple developer tool through xcrun."""

    completed = subprocess.run(
        ["xcrun", "--find", tool_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None
    resolved = completed.stdout.strip()
    return resolved or None


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a built artifact predates any source or recipe input."""

    if not binary_path.exists():
        return True

    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


class MetalBackend:
    """Compile and invoke the conv2d Metal autotuning runner."""

    name = "metal"

    def diagnostic_context(self, _spec: BenchmarkSpec | None = None) -> dict[str, object]:
        """Return structured tuning context for status logging."""

        return {
            "implementation": "mpsgraph",
            "block_size_candidates": _candidate_block_sizes(),
            "tile_size_candidates": _candidate_tile_sizes(),
            "autotune_repeats": DEFAULT_AUTOTUNE_REPEATS,
            "measurement_repeats": DEFAULT_MEASUREMENT_REPEATS,
            "runner_path": str(METAL_EXECUTABLE_PATH),
        }

    def probe(self) -> tuple[bool, str]:
        """Check whether the current machine looks Metal-capable."""

        if sys.platform != "darwin":
            return False, "Metal backend is only available on macOS."
        if not METAL_HOST_SOURCE_PATH.exists():
            return False, f"missing Metal host runner source at {_relative_project_path(METAL_HOST_SOURCE_PATH)}"

        build_inputs = [METAL_HOST_SOURCE_PATH, Path(__file__)]
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
                f"{_relative_project_path(METAL_HOST_SOURCE_PATH)} will be rebuilt against "
                "MetalPerformanceShadersGraph.",
            )

        if not toolchain_available:
            return False, toolchain_message
        return (
            True,
            "Metal toolchain detected via xcrun; binary is missing, so a self-contained Metal runner backed by "
            "MetalPerformanceShadersGraph will be compiled.",
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
        """Run the Metal executable and return the best configuration it found."""

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
        final_measurement_repeats = (
            1 if (measurement_spec != spec or measurement_dataset != dataset)
            else DEFAULT_MEASUREMENT_REPEATS
        )
        autotune_output_channel_batch = max(1, int(spec.c_out))
        measurement_output_channel_batch = max(1, int(measurement_spec.c_out))

        block_sizes = _candidate_block_sizes()
        tile_sizes = _candidate_tile_sizes()
        notes.append("implementation: official Apple MPSGraph convolution2D")
        notes.append("launch-shape autotune hints are accepted for interface compatibility but ignored by MPSGraph")
        notes.append("wall-clock timings include MPSGraph graph construction and initial compilation cost")
        notes.append(f"autotune_output_channel_batch: {autotune_output_channel_batch}")
        notes.append(f"measurement_output_channel_batch: {measurement_output_channel_batch}")
        notes.append(f"autotune_repeats_per_run: {DEFAULT_AUTOTUNE_REPEATS}")
        notes.append(f"measurement_repeats_for_best_run: {final_measurement_repeats}")

        try:
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                block_sizes=block_sizes,
                tile_sizes=tile_sizes,
                output_channel_batch=autotune_output_channel_batch,
                autotune_repeats=DEFAULT_AUTOTUNE_REPEATS,
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
            )
            selected_config = self._selected_config(
                autotune_metrics,
                measurement_repeats=final_measurement_repeats,
            )
            if callable(phase_callback):
                phase_callback("final_measurement", selected_config)
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
                block_sizes=block_sizes,
                tile_sizes=tile_sizes,
                output_channel_batch=measurement_output_channel_batch,
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
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

        autotune_config = self._selected_config(
            autotune_metrics,
            measurement_repeats=int(autotune_metrics["measurement_repeats"]),
        )
        measurement_config = self._selected_config(
            metrics,
            measurement_repeats=int(metrics["measurement_repeats"]),
            autotune_repeats=int(autotune_metrics["autotune_repeats"]),
            trials_run=int(autotune_metrics["trials_run"]),
        )

        trial_notes: list[str] = []
        if "implementation" in metrics:
            trial_notes.append(f"implementation={metrics['implementation']}")
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
            config=measurement_config,
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
        block_sizes: list[int],
        tile_sizes: list[int],
        output_channel_batch: int,
        autotune_repeats: int,
        measurement_repeats: int,
        timeout_seconds: float,
        verbose: bool = False,
    ) -> dict[str, object]:
        """Invoke the native Metal runner and parse its JSON metrics."""

        command = [
            str(executable_path),
            "--mode",
            "benchmark",
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
            "--block-sizes",
            ",".join(str(value) for value in block_sizes),
            "--tile-sizes",
            ",".join(str(value) for value in tile_sizes),
            "--include-preparation-in-metrics",
            "1",
            "--output-channel-batch",
            str(output_channel_batch),
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
            command=command,
            timeout_seconds=timeout_seconds,
            autotune_repeats=autotune_repeats,
            measurement_repeats=measurement_repeats,
        )

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
            raise

        stderr_pump.join(timeout=5.0)
        stdout_pump.join(timeout=5.0)
        stdout_data = "".join(stdout_chunks)
        stderr_data = "".join(stderr_chunks)

        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, command, output=stdout_data, stderr=stderr_data
            )

        emit_status(
            "method.conv2d.backend.native_runner.complete",
            status="running",
            method="conv2d",
            backend=self.name,
            command=command,
        )
        return json.loads(stdout_data)

    def _compile_if_needed(self, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Build the self-contained Metal runner when sources changed."""

        METAL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        build_inputs = [METAL_HOST_SOURCE_PATH, Path(__file__)]
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
                "clang++",
                "../conv2d_metal_runner.mm",
                "-std=c++20",
                "-O3",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                "-framework",
                "MetalPerformanceShaders",
                "-framework",
                "MetalPerformanceShadersGraph",
                "-o",
                "conv2d_metal_runner",
            ],
            cwd=METAL_BUILD_DIR,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return (
            METAL_EXECUTABLE_PATH,
            f"compiled self-contained Metal runner from {_relative_project_path(METAL_HOST_SOURCE_PATH)} "
            "against MetalPerformanceShadersGraph"
            + (" because rebuild was explicitly requested" if force_rebuild else ""),
        )

    def _toolchain_status(self) -> tuple[bool, str]:
        """Report whether the local Metal compiler toolchain is available."""

        missing_tools: list[str] = []
        for tool_name in ("clang++",):
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
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            return "self-contained Metal runner relies only on Apple system frameworks at runtime."

        dynamic_libraries: list[str] = []
        for line in completed.stdout.splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            library_path = stripped.split(" (compatibility version", 1)[0].strip()
            dynamic_libraries.append(Path(library_path).name or library_path)

        if not dynamic_libraries:
            return "self-contained Metal runner reported no explicit framework dependencies."

        return (
            "self-contained Metal runner depends only on macOS system frameworks: "
            + ", ".join(dynamic_libraries)
        )

    def _selected_config(
        self,
        metrics: dict[str, object],
        *,
        measurement_repeats: int,
        autotune_repeats: int | None = None,
        trials_run: int | None = None,
    ) -> dict[str, object]:
        """Normalize runner metrics into one benchmark selected-config payload."""

        config: dict[str, object] = {}
        for key in (
            "implementation",
            "block_size",
            "tile_size",
            "shared_input",
            "transpose",
            "output_channel_batch",
        ):
            if key in metrics:
                config[key] = metrics[key]
        config["autotune_repeats"] = int(
            metrics.get("autotune_repeats") if autotune_repeats is None else autotune_repeats
        )
        config["measurement_repeats"] = int(measurement_repeats)
        config["trials_run"] = int(metrics.get("trials_run") if trials_run is None else trials_run)
        return config
