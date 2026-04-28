"""CUDA backend for the conv2d benchmark.

Use this module to probe CUDA availability, compile or reuse the native CUDA
runner, sweep launch parameters, and convert runner output into benchmark
models used by the reporting pipeline.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

from core.constants import DEFAULT_CONV2D_CUDA_COOLDOWN_MS
from supervision.compute_resource_policy import (
    build_conv2d_cuda_output_channel_batch_candidates,
)
from compute_node.compute_methods.conv2d import (
    CUDA_BUILD_DIR,
    CUDA_DIR,
    CUDA_EXECUTABLE_PATH,
    CUDA_SOURCE_PATH,
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
DESIRED_FATBIN_SMS = (
    "50", "52",          # Maxwell
    "60", "61",          # Pascal
    "70",                # Volta
    "75",                # Turing
    "80", "86", "87", "88",  # Ampere family variants
    "89",                # Ada
    "90",                # Hopper
    "100", "103", "110", "120", "121",  # Blackwell / post-Blackwell toolchain labels
)
PTX_FALLBACK_PREFERENCE = ("120", "110", "100", "90", "89", "86", "80", "75", "70", "61", "52", "50")
CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE: tuple[int, ...] | None = None
CONV2D_CUDA_COOLDOWN_MS = DEFAULT_CONV2D_CUDA_COOLDOWN_MS

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


def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    """Return whether a compiled binary predates any relevant input files.

    Args:
        binary_path: Runner binary that may need rebuilding.
        inputs: Source or recipe files that should be newer than the binary.

    Returns:
        ``True`` when the binary is missing or older than any input.
    """
    if not binary_path.exists():
        return True
    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime:
            return True
    return False


def _candidate_block_sizes() -> list[int]:
    """Return CUDA block-size candidates for autotune."""
    return [64, 128, 256, 512]


def _candidate_tile_sizes() -> list[int]:
    """Return CUDA tile-size candidates for autotune."""
    return [8, 16, 32]


def _candidate_transpose_modes() -> list[int]:
    """Return transpose-mode candidates supported by the CUDA runner."""
    return [0]


def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed autotune repeat count for CUDA Conv2D benchmarks."""
    return DEFAULT_AUTOTUNE_REPEATS


def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    """Return the fixed measurement repeat count for CUDA Conv2D benchmarks."""
    return DEFAULT_MEASUREMENT_REPEATS


def _candidate_output_channel_batches(spec: BenchmarkSpec | None = None) -> list[int]:
    """Return output-channel batch candidates for CUDA autotune.

    Args:
        spec: Optional benchmark spec whose output channels bound the search.

    Returns:
        Candidate output-channel batch sizes for the runner to sweep.
    """
    if CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE is not None:
        return [int(value) for value in CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_OVERRIDE]
    output_channels = 64 if spec is None else spec.c_out
    return build_conv2d_cuda_output_channel_batch_candidates(output_channels)


def _detect_cuda_device_name() -> str | None:
    """Query the first CUDA device name through ``nvidia-smi``."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    completed = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _detect_compute_capability() -> str | None:
    """Query the first CUDA device compute capability through ``nvidia-smi``."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    completed = subprocess.run(
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            digits = stripped.replace(".", "")
            if digits.isdigit():
                return digits
    return None


@lru_cache(maxsize=1)
def _detect_nvcc_supported_sms() -> set[str] | None:
    """Ask ``nvcc`` which ``sm_XX`` targets this toolchain supports."""
    if shutil.which("nvcc") is None:
        return None

    completed = subprocess.run(["nvcc", "--list-gpu-code"], capture_output=True, text=True, encoding="utf-8", errors="replace")
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


@lru_cache(maxsize=1)
def _detect_nvcc_supported_compute_arches() -> set[str] | None:
    """Ask ``nvcc`` which ``compute_XX`` PTX targets this toolchain supports."""
    if shutil.which("nvcc") is None:
        return None

    completed = subprocess.run(["nvcc", "--list-gpu-arch"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        return None

    supported: set[str] = set()
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("compute_"):
            digits = stripped.removeprefix("compute_")
            if digits.isdigit():
                supported.add(digits)
    return supported or None


def _sort_arch_values(values: set[str]) -> list[str]:
    """Sort architecture digits numerically instead of lexicographically."""
    return sorted(values, key=lambda value: int(value))


def _select_fatbin_sm_targets(capability: str | None) -> list[str]:
    """Choose SM targets that should be baked into the CUDA runner fatbin."""
    supported_sms = _detect_nvcc_supported_sms()
    if supported_sms is None:
        fallback = ["75", "80", "86", "89", "90"]
        if capability is not None and capability not in fallback:
            fallback.append(capability)
        return fallback

    ordered_targets = [sm for sm in DESIRED_FATBIN_SMS if sm in supported_sms]
    if capability is not None and capability in supported_sms and capability not in ordered_targets:
        ordered_targets.append(capability)
    if ordered_targets:
        return ordered_targets

    if capability is not None and capability in supported_sms:
        return [capability]
    return _sort_arch_values(supported_sms)


def _select_ptx_fallback_target() -> str | None:
    """Choose the PTX fallback target for forward-compatible execution."""
    supported_arches = _detect_nvcc_supported_compute_arches()
    if supported_arches is None:
        return None

    for arch in PTX_FALLBACK_PREFERENCE:
        if arch in supported_arches:
            return arch

    ordered = _sort_arch_values(supported_arches)
    return ordered[-1] if ordered else None


def _fatbin_target_summary(capability: str | None) -> tuple[list[str], str | None]:
    """Return both baked SM targets and the PTX fallback target."""
    return _select_fatbin_sm_targets(capability), _select_ptx_fallback_target()


def _fatbin_gencode_args(capability: str | None) -> list[str]:
    """Build ``nvcc`` ``-gencode`` arguments for the chosen CUDA targets."""
    args: list[str] = []
    sm_targets, ptx_target = _fatbin_target_summary(capability)
    for sm in sm_targets:
        args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    if ptx_target is not None:
        args.append(f"-gencode=arch=compute_{ptx_target},code=compute_{ptx_target}")
    return args


def _fatbin_note(capability: str | None) -> str:
    """Return a human-readable summary of the chosen CUDA fatbin targets."""
    sm_targets, ptx_target = _fatbin_target_summary(capability)
    parts = [f"fatbin SMs: {', '.join(f'sm_{sm}' for sm in sm_targets)}"] if sm_targets else []
    if ptx_target is not None:
        parts.append(f"PTX fallback: compute_{ptx_target}")
    supported_sms = _detect_nvcc_supported_sms()
    if supported_sms is not None:
        omitted = [sm for sm in DESIRED_FATBIN_SMS if sm not in supported_sms]
        if omitted:
            parts.append(
                "toolkit-limited omissions: " + ", ".join(f"sm_{sm}" for sm in omitted)
            )
    return "; ".join(parts)


class CudaBackend:
    """Compile, probe, and run the native spatial CUDA benchmark backend."""

    name = "cuda"

    def diagnostic_context(self, spec: BenchmarkSpec | None = None) -> dict[str, object]:
        """Return structured tuning context for status logging.

        Args:
            spec: Optional benchmark spec being prepared.

        Returns:
            A dictionary describing CUDA autotune candidate settings.
        """
        return {
            "device_name": _detect_cuda_device_name(),
            "compute_capability": _detect_compute_capability(),
            "block_size_candidates": _candidate_block_sizes(),
            "tile_size_candidates": _candidate_tile_sizes(),
            "transpose_modes": _candidate_transpose_modes(),
            "autotune_repeats": _autotune_repeats(spec),
            "measurement_repeats": _measurement_repeats(spec),
            "output_channel_batch_candidates": _candidate_output_channel_batches(spec),
            "cooldown_ms": CONV2D_CUDA_COOLDOWN_MS,
        }

    def probe(self) -> tuple[bool, str]:
        """Check whether this machine can run the spatial CUDA backend.

        Returns:
            A ``(available, message)`` tuple describing backend readiness.
        """
        if not CUDA_SOURCE_PATH.exists():
            return False, f"missing CUDA runner source at {_relative_project_path(CUDA_SOURCE_PATH)}"

        device_name = _detect_cuda_device_name()
        capability = _detect_compute_capability()
        if device_name is None:
            return False, "NVIDIA GPU was not detected via nvidia-smi."

        toolchain_available, toolchain_message = self._toolchain_status()
        build_inputs = [CUDA_SOURCE_PATH, Path(__file__)]

        if CUDA_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(CUDA_EXECUTABLE_PATH, build_inputs):
                capability_note = f" (detected sm_{capability})" if capability else ""
                return True, (
                    f"CUDA backend available for {device_name!r}{capability_note} via "
                    f"{_relative_project_path(CUDA_EXECUTABLE_PATH)}."
                )
            if toolchain_available:
                return True, (
                    f"CUDA backend available for {device_name!r}. Existing binary is older than the source, "
                    "so it will be rebuilt."
                )
            return True, (
                f"CUDA backend available for {device_name!r} via {_relative_project_path(CUDA_EXECUTABLE_PATH)}, "
                f"but the local CUDA toolchain is unavailable ({toolchain_message}), so the existing binary will be used."
            )

        if not toolchain_available:
            return False, (
                f"CUDA runner binary is missing at {_relative_project_path(CUDA_EXECUTABLE_PATH)} and the local "
                f"CUDA toolchain is unavailable ({toolchain_message})."
            )

        capability_note = f" (detected sm_{capability})" if capability else ""
        return True, f"CUDA backend available for {device_name!r}{capability_note}; the runner will be compiled from source."

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
        """Run autotune plus final measurement for the CUDA backend.

        Args:
            spec: Benchmark spec used for autotune.
            dataset: Dataset layout used for autotune.
            measurement_spec: Optional spec for the final measurement phase.
            measurement_dataset: Optional dataset for the final measurement phase.
            time_budget_seconds: Overall time budget for the backend run.
            force_rebuild: Whether to force recompiling the CUDA runner.

        Returns:
            A ``BackendResult`` describing availability and measured performance.
        """
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
        except Exception as exc:
            notes.append(_sanitize_note(f"failed to prepare CUDA runner: {exc}"))
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
            else _measurement_repeats(measurement_spec)
        )
        autotune_output_channel_batches = _candidate_output_channel_batches(spec)
        try:
            autotune_subprocess_started = time.perf_counter()
            autotune_metrics = self._run_runner(
                executable_path,
                spec,
                dataset,
                block_sizes=_candidate_block_sizes(),
                tile_sizes=_candidate_tile_sizes(),
                transpose_modes=_candidate_transpose_modes(),
                output_channel_batches=autotune_output_channel_batches,
                autotune_repeats=_autotune_repeats(spec),
                measurement_repeats=1,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
            )
            autotune_subprocess_wall_s = time.perf_counter() - autotune_subprocess_started
        except Exception as exc:
            notes.append(_sanitize_note(f"CUDA autotune failed: {exc}"))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        invalid_reason = self._validate_metrics(autotune_metrics)
        if invalid_reason is not None:
            notes.append(invalid_reason)
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
        selected_output_channel_batch = int(
            autotune_metrics.get("output_channel_batch") or measurement_spec.c_out
        )
        selected_config = {
            "transpose": bool(selected_transpose),
            "block_size": selected_block_size,
            "tile_size": selected_tile_size,
            "output_channel_batch": selected_output_channel_batch,
            "cooldown_ms": float(autotune_metrics.get("cooldown_ms", CONV2D_CUDA_COOLDOWN_MS)),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": final_measurement_repeats,
        }
        if callable(phase_callback):
            phase_callback("final_measurement", selected_config)

        try:
            measurement_subprocess_started = time.perf_counter()
            measurement_metrics = self._run_runner(
                executable_path,
                measurement_spec,
                measurement_dataset,
                block_sizes=[selected_block_size],
                tile_sizes=[selected_tile_size],
                transpose_modes=[selected_transpose],
                output_channel_batches=[selected_output_channel_batch],
                autotune_repeats=1,
                measurement_repeats=final_measurement_repeats,
                timeout_seconds=max(time_budget_seconds, 30.0),
                verbose=verbose,
            )
            measurement_subprocess_wall_s = time.perf_counter() - measurement_subprocess_started
        except Exception as exc:
            notes.append(_sanitize_note(f"CUDA runtime measurement failed: {exc}"))
            return BackendResult(
                backend=self.name,
                available=False,
                selected_config=None,
                autotune_trial=None,
                best_trial=None,
                trials=[],
                notes=notes,
            )

        invalid_reason = self._validate_metrics(measurement_metrics)
        if invalid_reason is not None:
            notes.append(invalid_reason)
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
            float(measurement_metrics["measurement_wall_clock_latency_seconds"]),
            ideal_seconds=measurement_spec.ideal_seconds,
            zero_score_seconds=measurement_spec.zero_score_seconds,
        )

        autotune_config = {
            "transpose": bool(autotune_metrics["transpose"]),
            "shared_input": bool(int(autotune_metrics.get("shared_input", 0))),
            "block_size": int(autotune_metrics["block_size"]),
            "tile_size": int(autotune_metrics["tile_size"]),
            "output_channel_batch": selected_output_channel_batch,
            "cooldown_ms": float(autotune_metrics.get("cooldown_ms", CONV2D_CUDA_COOLDOWN_MS)),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(autotune_metrics["measurement_repeats"]),
            "trials_run": int(autotune_metrics["trials_run"]),
        }
        measurement_config = {
            "transpose": bool(measurement_metrics["transpose"]),
            "shared_input": bool(int(measurement_metrics.get("shared_input", 0))),
            "block_size": int(measurement_metrics["block_size"]),
            "tile_size": int(measurement_metrics["tile_size"]),
            "output_channel_batch": int(measurement_metrics.get("output_channel_batch", selected_output_channel_batch)),
            "cooldown_ms": float(measurement_metrics.get("cooldown_ms", CONV2D_CUDA_COOLDOWN_MS)),
            "autotune_repeats": int(autotune_metrics["autotune_repeats"]),
            "measurement_repeats": int(measurement_metrics["measurement_repeats"]),
            "trials_run": int(measurement_metrics["trials_run"]),
        }
        trial_notes: list[str] = []
        if measurement_metrics.get("device_name"):
            trial_notes.append(f"device={measurement_metrics['device_name']}")

        autotune_trial = TrialRecord(
            self.name,
            autotune_config,
            float(autotune_metrics["autotune_wall_clock_latency_seconds"]),
            float(autotune_metrics["autotune_effective_gflops"]),
            str(autotune_metrics["autotune_checksum"]),
            autotune_score,
            list(trial_notes),
        )
        trial = TrialRecord(
            self.name,
            measurement_config,
            float(measurement_metrics["measurement_wall_clock_latency_seconds"]),
            float(measurement_metrics["measurement_effective_gflops"]),
            str(measurement_metrics["measurement_checksum"]),
            measurement_score,
            list(trial_notes),
        )
        notes.append(
            "native_subprocess_wall_s: "
            f"autotune_sweep={autotune_subprocess_wall_s:.6f}, "
            f"measurement_pass={measurement_subprocess_wall_s:.6f}, "
            f"sum={autotune_subprocess_wall_s + measurement_subprocess_wall_s:.6f}"
        )
        raw_report = {
            "autotune": _extract_raw_report(autotune_metrics),
            "measurement": _extract_raw_report(measurement_metrics),
        }
        return BackendResult(
            self.name,
            True,
            dict(measurement_config),
            autotune_trial,
            trial,
            [autotune_trial, trial],
            notes,
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
        transpose_modes: list[int],
        output_channel_batches: list[int],
        autotune_repeats: int,
        measurement_repeats: int,
        timeout_seconds: float,
        verbose: bool = False,
    ) -> dict[str, object]:
        """Invoke the native CUDA runner and parse its JSON metrics.

        Args:
            executable_path: Runner binary to execute.
            spec: Benchmark spec being measured.
            dataset: Dataset layout read by the runner.
            block_sizes: CUDA block-size candidates to sweep.
            tile_sizes: Tile-size candidates to sweep.
            transpose_modes: Transpose-mode candidates to sweep.
            output_channel_batches: Output-channel batch candidates to sweep.
            autotune_repeats: Repeat count for autotune trials.
            measurement_repeats: Repeat count for the selected configuration.
            timeout_seconds: Subprocess timeout for the native runner.

        Returns:
            The parsed JSON metrics emitted by the native runner.
        """
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
            "--output-channel-batches",
            ",".join(str(value) for value in output_channel_batches),
            "--cooldown-ms",
            str(CONV2D_CUDA_COOLDOWN_MS),
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
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc

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
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            stderr_pump.join(timeout=5.0)
            stdout_pump.join(timeout=5.0)
            raise RuntimeError(f"CUDA runner timed out after {timeout_seconds:.1f}s") from exc

        stderr_pump.join(timeout=5.0)
        stdout_pump.join(timeout=5.0)
        stdout_data = "".join(stdout_chunks)
        stderr_data = "".join(stderr_chunks)

        if return_code != 0:
            details = (stderr_data or stdout_data).strip()
            raise RuntimeError(details if details else f"CUDA runner exited with code {return_code}")

        emit_status(
            "method.conv2d.backend.native_runner.complete",
            status="running",
            method="conv2d",
            backend=self.name,
            command=command,
        )
        return json.loads(stdout_data)

    def _validate_metrics(self, metrics: dict[str, object]) -> str | None:
        """Validate the required numeric fields returned by the CUDA runner.

        Args:
            metrics: Parsed runner-output dictionary.

        Returns:
            ``None`` when valid, otherwise an error string describing the issue.
        """
        required_keys = (
            "autotune_wall_clock_latency_seconds",
            "measurement_wall_clock_latency_seconds",
            "autotune_effective_gflops",
            "measurement_effective_gflops",
            "autotune_checksum",
            "measurement_checksum",
            "block_size",
            "tile_size",
        )
        for key in required_keys:
            if key not in metrics:
                return f"CUDA runner output is missing required key {key!r}."

        autotune_seconds = float(metrics["autotune_wall_clock_latency_seconds"])
        measurement_seconds = float(metrics["measurement_wall_clock_latency_seconds"])
        autotune_gflops = float(metrics["autotune_effective_gflops"])
        measurement_gflops = float(metrics["measurement_effective_gflops"])
        measurement_checksum = str(metrics["measurement_checksum"])

        if autotune_seconds <= 0.0 or measurement_seconds <= 0.0:
            return "CUDA runner reported a non-positive runtime."
        if not math.isfinite(autotune_gflops) or not math.isfinite(measurement_gflops):
            return "CUDA runner reported a non-finite GFLOPS value."
        if autotune_gflops <= 0.0 or measurement_gflops <= 0.0:
            return "CUDA runner reported a non-positive GFLOPS value."
        if measurement_checksum == "chk_0":
            return (
                "CUDA runner produced an all-zero checksum on the benchmark dataset; "
                "treating this run as invalid instead of ranking it."
            )
        return None

    def _resolve_executable_path(self, *, force_rebuild: bool = False) -> tuple[Path, str]:
        """Resolve or build the CUDA runner executable for this machine.

        Args:
            force_rebuild: Whether to force recompiling the runner.

        Returns:
            A tuple of ``(executable_path, note)`` describing the resolution.
        """
        CUDA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        build_inputs = [CUDA_SOURCE_PATH, Path(__file__)]
        toolchain_available, toolchain_message = self._toolchain_status()
        capability = _detect_compute_capability()

        if force_rebuild:
            if not toolchain_available:
                raise FileNotFoundError(
                    f"force rebuild requested for the CUDA runner, but the local CUDA toolchain is unavailable "
                    f"({toolchain_message})"
                )
        elif CUDA_EXECUTABLE_PATH.exists():
            if not _binary_is_stale(CUDA_EXECUTABLE_PATH, build_inputs):
                return CUDA_EXECUTABLE_PATH, f"using CUDA runner at {_relative_project_path(CUDA_EXECUTABLE_PATH)}"
            if not toolchain_available:
                return (
                    CUDA_EXECUTABLE_PATH,
                    f"using CUDA runner at {_relative_project_path(CUDA_EXECUTABLE_PATH)} because the source is newer "
                    f"but the local CUDA toolchain is unavailable ({toolchain_message})",
                )

        if not toolchain_available:
            raise FileNotFoundError(toolchain_message)

        self._compile_runner()
        if not CUDA_EXECUTABLE_PATH.exists():
            raise FileNotFoundError(
                f"CUDA runner build completed without producing {_relative_project_path(CUDA_EXECUTABLE_PATH)}"
            )
        compile_note = f"compiled CUDA runner from {_relative_project_path(CUDA_SOURCE_PATH)}"
        fatbin_note = _fatbin_note(capability)
        if fatbin_note:
            compile_note = f"{compile_note}; {fatbin_note}"
        return CUDA_EXECUTABLE_PATH, compile_note

    def _compile_runner(self) -> None:
        """Compile the CUDA runner on the current operating system."""
        capability = _detect_compute_capability()
        if os.name == "nt":
            self._compile_windows_runner(capability)
            return

        completed = subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-O3",
                "--use_fast_math",
                "-Wno-deprecated-gpu-targets",
                "-o",
                str(CUDA_EXECUTABLE_PATH),
                str(CUDA_SOURCE_PATH),
                *_fatbin_gencode_args(capability),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=CUDA_BUILD_DIR,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    def _compile_windows_runner(self, capability: str | None) -> None:
        """Compile the CUDA runner on Windows through a generated batch script.

        Args:
            capability: Detected device compute capability, if known.

        Returns:
            ``None`` after the compile step finishes successfully.
        """
        vsdevcmd = self._find_vsdevcmd()
        if vsdevcmd is None:
            raise FileNotFoundError("VsDevCmd.bat was not found")

        compile_script_path = CUDA_BUILD_DIR / "build_conv2d_cuda_runner.cmd"
        compile_script_path.write_text(
            "\n".join(
                [
                    "@echo off",
                    f"call \"{vsdevcmd}\" -arch=x64 -host_arch=x64 >nul",
                    "if errorlevel 1 exit /b %errorlevel%",
                    "pushd \"%~dp0\"",
                    " ".join(
                        [
                            "nvcc",
                            "-std=c++17",
                            "-O3",
                            "--use_fast_math",
                            "-Wno-deprecated-gpu-targets",
                            "-cudart",
                            "static",
                            "-Xcompiler",
                            "\"/MT /EHsc\"",
                            "-o",
                            CUDA_EXECUTABLE_PATH.name,
                            f"..\\{CUDA_SOURCE_PATH.name}",
                            *_fatbin_gencode_args(capability),
                        ]
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
            ["cmd", "/c", str(compile_script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=CUDA_BUILD_DIR,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    def _toolchain_status(self) -> tuple[bool, str]:
        """Return whether the local CUDA build toolchain is available."""
        if shutil.which("nvcc") is None:
            return False, "nvcc was not found on PATH."
        if os.name == "nt" and self._find_vsdevcmd() is None:
            return False, "VsDevCmd.bat was not found; the MSVC build environment is unavailable."
        return True, "nvcc is available."

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
