"""Execute conv2d runtime tasks on the local compute node.

Use this module when a compute node receives a conv2d
``TaskAssign`` and needs to choose the best local backend, run the native
runner, and package the result back into a ``TaskResult``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from core.process_exit import classify_exit_code

from core.constants import (
    CONV2D_CLIENT_RESPONSE_STATS_ONLY,
    CONV2D_STATS_MAX_SAMPLES,
    DEFAULT_CONV2D_CUDA_COOLDOWN_MS,
    DX12_BACKEND_DISABLED_REASON,
    METHOD_CONV2D,
    STATUS_OK,
)
from adapters.process import python_utf8_command
from compute_node.compute_methods.conv2d.paths import (
    CPU_MACOS_EXECUTABLE_PATH,
    CPU_WINDOWS_EXECUTABLE_PATH,
    CUDA_EXECUTABLE_PATH,
    METAL_EXECUTABLE_PATH,
    CONV2D_METHOD_DIR,
)
from compute_node.performance_metrics.conv2d.models import BenchmarkSpec
from compute_node.performance_metrics.conv2d.config import (
    DATASET_DIR as TOP_LEVEL_DATASET_DIR,
    GENERATE_SCRIPT_PATH as TOP_LEVEL_GENERATE_SCRIPT_PATH,
    RESULT_PATH as TOP_LEVEL_RESULT_PATH,
)
from compute_node.input_matrix.conv2d import build_input_matrix_spec, normalize_size_variant
from compute_node.performance_metrics.performance_summary import RuntimeProcessorProfile, load_runtime_processor_inventory
from setup import active_python_path
from wire.internal_protocol.control_plane import Conv2dResultPayload
from wire.internal_protocol.transport import TaskAssign, TaskResult, TransferMode

_LOGGER = logging.getLogger(__name__)
_RUNNER_STDERR_TAIL_BYTES = 2048


def _tail_stream(payload: str | bytes | None, *, limit: int = _RUNNER_STDERR_TAIL_BYTES) -> str:
    """Return the trailing portion of a captured runner stream, safe to log.

    Native runners can produce many KB of progress output; only the tail
    typically matters when explaining a crash, and bounding the size keeps the
    log readable.
    """
    if payload is None:
        return "<none>"
    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="replace")
    else:
        text = payload
    text = text.strip()
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return f"...<truncated {len(text) - limit} bytes>...{text[-limit:]}"


class RunnerProcessError(RuntimeError):
    """A native runner exited nonzero (or timed out) and we captured its stderr.

    Why a custom type: the conv2d executor runs inside a ProcessPoolExecutor
    child where ``_LOGGER`` has no handlers, so any ``_LOGGER.error`` we emit
    in the child is silently dropped. The parent's ``drain_completed_tasks``
    only sees the exception via ``future.result()`` and stringifies it once
    into both the worker log and the upstream TASK_FAIL message — so the
    stderr tail must travel inside the exception's ``str()`` to reach the
    operator after the cluster exits. Plain ``CalledProcessError.__str__()``
    contains only the command line.
    """


def _format_runner_failure_message(
    *,
    method: str,
    backend_name: str,
    task: TaskAssign,
    returncode: int | None,
    stderr: str | bytes | None,
    stdout: str | bytes | None,
    elapsed_ms: int,
) -> str:
    classification = classify_exit_code(returncode)
    return (
        f"{method} native runner failed: backend={backend_name} "
        f"task_id={getattr(task, 'task_id', '?')} elapsed_ms={elapsed_ms} "
        f"returncode={returncode} cause=\"{classification}\" "
        f"stderr_tail={_tail_stream(stderr)!r} stdout_tail={_tail_stream(stdout)!r}"
    )


def _format_runner_timeout_message(
    *,
    method: str,
    backend_name: str,
    task: TaskAssign,
    timeout: float | None,
    stderr: str | bytes | None,
    stdout: str | bytes | None,
) -> str:
    return (
        f"{method} native runner timed out: backend={backend_name} "
        f"task_id={getattr(task, 'task_id', '?')} timeout={float(timeout or 0.0):.1f}s "
        f"stderr_tail={_tail_stream(stderr)!r} stdout_tail={_tail_stream(stdout)!r}"
    )


def _log_runner_failure(
    *,
    method: str,
    backend_name: str,
    task: TaskAssign,
    returncode: int | None,
    stderr: str | bytes | None,
    stdout: str | bytes | None,
    elapsed_ms: int,
) -> None:
    """Emit a single ERROR line summarizing why a native runner exited nonzero.

    Kept for in-process callers; conv2d's ProcessPoolExecutor path can't rely
    on this reaching a file (no handlers in the child), so it uses the
    formatter directly to populate ``RunnerProcessError`` instead.
    """
    _LOGGER.error(
        "%s",
        _format_runner_failure_message(
            method=method,
            backend_name=backend_name,
            task=task,
            returncode=returncode,
            stderr=stderr,
            stdout=stdout,
            elapsed_ms=elapsed_ms,
        ),
    )


def _log_runner_timeout(
    *,
    method: str,
    backend_name: str,
    task: TaskAssign,
    timeout: float | None,
    stderr: str | bytes | None,
    stdout: str | bytes | None,
) -> None:
    """Emit a single ERROR line when a native runner blew its wall-clock budget."""
    _LOGGER.error(
        "%s",
        _format_runner_timeout_message(
            method=method,
            backend_name=backend_name,
            task=task,
            timeout=timeout,
            stderr=stderr,
            stdout=stdout,
        ),
    )


def _parse_compute_event_ms(stdout: str | bytes | None) -> int | None:
    """Extract the native runner's cudaEvent-bracketed kernel time (ms).

    Returns ``None`` when the runner stdout has no parseable
    ``compute_event_ms`` field, e.g. older runners that pre-date the
    dispatch/benchmark split or non-CUDA backends that don't emit JSON at all.
    """
    if not stdout:
        return None
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    try:
        payload = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    value = payload.get("compute_event_ms") if isinstance(payload, dict) else None
    if value is None:
        return None
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return None
    if ms < 0.0:
        return None
    return int(round(ms))


def _summarize_conv2d_slice_file(path: Path, *, max_samples: int) -> tuple[int, float, float, tuple[float, ...]]:
    """Stream a float32 conv2d slice file and return count, sum, sum-of-squares, and leading samples."""
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"conv2d slice file size is not a multiple of 4: {size}")
    element_count = size // 4
    sum_v = 0.0
    sum_sq = 0.0
    samples: list[float] = []
    remaining = max(0, int(max_samples))
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            values = np.frombuffer(chunk, dtype=np.float32)
            values64 = values.astype(np.float64)
            sum_v += float(values64.sum())
            sum_sq += float(np.dot(values64, values64))
            if remaining > 0:
                take = min(remaining, values.size)
                samples.extend(float(x) for x in values[:take])
                remaining -= take
    return element_count, sum_v, sum_sq, tuple(samples)

METHOD_DIR = CONV2D_METHOD_DIR
DEFAULT_CONV_RESULT_PATH = TOP_LEVEL_RESULT_PATH
DEFAULT_DATASET_DIR = TOP_LEVEL_DATASET_DIR
DATASET_GENERATE_SCRIPT_PATH = TOP_LEVEL_GENERATE_SCRIPT_PATH
DISABLED_CONV2D_BACKENDS = frozenset({"dx12"})
CONV2D_CUDA_COOLDOWN_MS = DEFAULT_CONV2D_CUDA_COOLDOWN_MS


def _best_backend_profile(
    result_path: Path,
    *,
    pinned_backend: str | None = None,
) -> RuntimeProcessorProfile | None:
    """Return the best usable processor profile from a method result file.

    Args:
        result_path: Conv2d benchmark result path for this compute node.
        pinned_backend: Optional backend name that restricts the inventory so
            only that backend's profile can be returned.

    Returns:
        The best non-disabled processor profile, or ``None``.
    """
    try:
        inventory = load_runtime_processor_inventory(
            result_path=result_path,
            method=METHOD_CONV2D,
            pinned_backend=pinned_backend,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    for processor in inventory.processors:
        if processor.hardware_type not in DISABLED_CONV2D_BACKENDS:
            return processor
    return None


def _best_backend_name(
    result_path: Path,
    *,
    pinned_backend: str | None = None,
) -> str:
    """Return the best usable backend name from the method result file.

    Args:
        result_path: Conv2d benchmark result path for this compute node.
        pinned_backend: Optional backend name to prefer; when set and valid,
            it is returned directly.

    Returns:
        The preferred backend name, falling back to ``cpu`` when unknown.
    """
    if pinned_backend is not None and pinned_backend not in DISABLED_CONV2D_BACKENDS:
        return pinned_backend
    profile = _best_backend_profile(result_path)
    if profile is not None:
        return profile.hardware_type
    if not result_path.exists():
        return "cpu"

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "cpu"

    methods = payload.get("methods")
    if isinstance(methods, dict):
        method_payload = methods.get(METHOD_CONV2D) or {}
        ranking = method_payload.get("ranking", [])
        for backend_name in ranking:
            name = str(backend_name)
            if name not in DISABLED_CONV2D_BACKENDS:
                return name
        return "cpu"

    ranking = payload.get("ranking", [])
    for backend_name in ranking:
        name = str(backend_name)
        if name not in DISABLED_CONV2D_BACKENDS:
            return name
    return "cpu"


def _runner_path(backend_name: str) -> Path:
    """Resolve the native runner executable path for one backend name.

    Args:
        backend_name: Backend name such as ``cpu`` or ``cuda``.

    Returns:
        The executable path for the requested backend.
    """
    if backend_name == "cpu":
        if sys.platform == "win32":
            return CPU_WINDOWS_EXECUTABLE_PATH
        if sys.platform == "darwin":
            return CPU_MACOS_EXECUTABLE_PATH
    if backend_name == "cuda":
        return CUDA_EXECUTABLE_PATH
    if backend_name == "dx12":
        raise ValueError(DX12_BACKEND_DISABLED_REASON)
    if backend_name == "metal":
        return METAL_EXECUTABLE_PATH
    raise ValueError(f"unsupported conv2d backend: {backend_name}")


def _prepare_runner_path(backend_name: str) -> Path:
    """Resolve one runtime runner path, compiling on demand when needed."""

    executable_path = _runner_path(backend_name)
    if executable_path.exists():
        return executable_path

    if backend_name == "cpu":
        from compute_node.performance_metrics.conv2d.backends.cpu_backend import CpuBackend

        executable_path, _note = CpuBackend()._resolve_executable_path(force_rebuild=False)
        return executable_path

    if backend_name == "cuda":
        from compute_node.performance_metrics.conv2d.backends.cuda_backend import CudaBackend

        executable_path, _note = CudaBackend()._resolve_executable_path(force_rebuild=False)
        return executable_path

    if backend_name == "metal":
        from compute_node.performance_metrics.conv2d.backends.metal_backend import MetalBackend

        executable_path, _note = MetalBackend()._compile_if_needed(force_rebuild=False)
        return executable_path

    return executable_path


def _spec_for_task(task: TaskAssign):
    """Resolve the workload spec and dataset variant for one task assignment."""
    return load_named_workload_spec(task.object_id, size=task.size)


def _size_from_object_id(object_id: str) -> str | None:
    """Infer one legacy workload size from a conv2d object id suffix."""

    normalized_object_id = object_id.strip().lower()
    for candidate in ("small", "mid", "large", "test", "medium", "runtime"):
        if normalized_object_id.endswith(f"/{candidate}"):
            return candidate
    return None


def load_named_workload_spec(object_id: str, size: str | None = None):
    """Resolve one supported conv2d object id into its workload spec.

    Args:
        object_id: Object id carried by the task assignment.
        size: Optional explicit workload size carried by the request/task.

    Returns:
        A ``(BenchmarkSpec, variant_name)`` pair for the requested object id.
    """
    variant = normalize_size_variant(size or _size_from_object_id(object_id), default="large")
    if variant not in {"small", "mid", "large"}:
        raise ValueError("conv2d requests require size small, mid, or large")
    dataset_spec = build_input_matrix_spec(default_variant=variant)
    benchmark_spec = BenchmarkSpec(
        name=dataset_spec.name,
        h=dataset_spec.h,
        w=dataset_spec.w,
        c_in=dataset_spec.c_in,
        c_out=dataset_spec.c_out,
        k=dataset_spec.k,
        pad=dataset_spec.pad,
        ideal_seconds=1.0,
        zero_score_seconds=10.0,
        stride=dataset_spec.stride,
    )
    return benchmark_spec, variant


def get_small_spec() -> BenchmarkSpec:
    """Return the canonical small conv2d runtime specification."""

    return load_named_workload_spec("conv2d/small", size="small")[0]


def get_mid_spec() -> BenchmarkSpec:
    """Return the canonical mid-sized conv2d runtime specification."""

    return load_named_workload_spec("conv2d/mid", size="mid")[0]


def get_large_spec() -> BenchmarkSpec:
    """Return the canonical large conv2d runtime specification."""

    return load_named_workload_spec("conv2d/large", size="large")[0]


def get_test_spec() -> BenchmarkSpec:
    """Return the legacy wrapper for the canonical small conv2d specification."""

    return get_small_spec()


def get_medium_spec() -> BenchmarkSpec:
    """Return the legacy wrapper for the canonical mid conv2d specification."""

    return get_mid_spec()


def get_runtime_spec() -> BenchmarkSpec:
    """Return the legacy wrapper for the canonical large conv2d specification."""

    return get_large_spec()


def _validate_task_against_spec(task: TaskAssign, spec) -> None:
    """Validate that a conv2d task matches the resolved workload spec.

    Args:
        task: Conv2d task assignment received from the main node.
        spec: Benchmark spec resolved from the task object id.

    Returns:
        ``None`` when the task is valid for local execution.
    """
    if task.method != METHOD_CONV2D:
        raise ValueError(f"unsupported task method: {task.method}")
    if task.start_oc < 0 or task.end_oc > spec.c_out or task.end_oc <= task.start_oc:
        raise ValueError("task output-channel range is invalid")
    if task.tensor_h and task.tensor_h != spec.h:
        raise ValueError(f"task tensor_h {task.tensor_h} does not match expected {spec.h}")
    if task.tensor_w and task.tensor_w != spec.w:
        raise ValueError(f"task tensor_w {task.tensor_w} does not match expected {spec.w}")
    if task.channels_in and task.channels_in != spec.c_in:
        raise ValueError(f"task channels_in {task.channels_in} does not match expected {spec.c_in}")
    if task.channels_out and task.channels_out != spec.c_out:
        raise ValueError(f"task channels_out {task.channels_out} does not match expected {spec.c_out}")
    if task.kernel_size and task.kernel_size != spec.k:
        raise ValueError(f"task kernel_size {task.kernel_size} does not match expected {spec.k}")
    if task.padding and task.padding != spec.pad:
        raise ValueError(f"task padding {task.padding} does not match expected {spec.pad}")
    if task.stride and task.stride != spec.stride:
        raise ValueError(f"task stride {task.stride} does not match expected {spec.stride}")

    expected_weight_bytes = spec.k * spec.k * spec.c_in * (task.end_oc - task.start_oc) * 4
    if len(task.weight_data) != expected_weight_bytes:
        raise ValueError(
            f"task weight byte size {len(task.weight_data)} does not match expected {expected_weight_bytes}"
        )


def _resolve_input_path(dataset_dir: Path, variant: str) -> Path:
    """Resolve the on-disk input path for one canonical or legacy dataset name."""

    normalized_variant = normalize_size_variant(variant, default="large")
    candidate = dataset_dir / f"{normalized_variant}_input.bin"
    if candidate.exists():
        return candidate
    legacy_variant = {"small": "test", "mid": "medium", "large": "runtime"}[normalized_variant]
    return dataset_dir / f"{legacy_variant}_input.bin"


def _ensure_dataset_ready(dataset_dir: Path, variant: str) -> None:
    """Generate the shared conv2d dataset if the compute node is missing it.

    Args:
        dataset_dir: Directory that should hold conv2d benchmark datasets.
        variant: Canonical named workload size needed by the current task.

    Returns:
        ``None`` after the required input files exist.
    """
    if _resolve_input_path(dataset_dir, variant).exists():
        return
    subprocess.run(
        python_utf8_command(
            active_python_path(),
            DATASET_GENERATE_SCRIPT_PATH,
            "--output-dir",
            dataset_dir,
            "--role",
            "compute",
        ),
        check=True,
        cwd=METHOD_DIR,
        timeout=600.0,
    )


class Conv2dTaskExecutor:
    """Execute one conv2d task on the local compute node."""

    def __init__(
        self,
        *,
        result_path: Path | None = None,
        dataset_root: Path | None = None,
        pinned_backend: str | None = None,
    ) -> None:
        """Store the benchmark result path and dataset root used at runtime.

        Args:
            result_path: Optional conv2d benchmark result path override.
            dataset_root: Optional conv2d dataset directory override.
            pinned_backend: Optional backend name restricting backend selection
                at task-dispatch time for dual-purpose peers.
        """
        self.result_path = DEFAULT_CONV_RESULT_PATH if result_path is None else Path(result_path)
        self.dataset_root = DEFAULT_DATASET_DIR if dataset_root is None else Path(dataset_root)
        self.pinned_backend = pinned_backend

    def execute_task(self, task: TaskAssign) -> TaskResult:
        """Execute one conv2d task and return its result payload.

        Args:
            task: Conv2d task assignment received from the main node.

        Returns:
            A ``TaskResult`` containing the output-channel slice bytes.
        """
        task_started_at = time.monotonic()
        computation_ms_total = 0
        spec, variant = _spec_for_task(task)
        _validate_task_against_spec(task, spec)
        _ensure_dataset_ready(self.dataset_root, variant)
        backend_profile = _best_backend_profile(
            self.result_path, pinned_backend=self.pinned_backend
        )
        backend_name = (
            backend_profile.hardware_type
            if backend_profile is not None
            else _best_backend_name(self.result_path, pinned_backend=self.pinned_backend)
        )
        best_config = {} if backend_profile is None else dict(backend_profile.best_config)
        executable_path = _prepare_runner_path(backend_name)
        if not executable_path.exists():
            raise FileNotFoundError(f"conv2d runner is missing: {executable_path}")

        input_path = _resolve_input_path(self.dataset_root, variant)
        if not input_path.exists():
            raise FileNotFoundError(f"missing generated conv2d input at {input_path}")

        disk_first_result = task.transfer_mode in (
            TransferMode.ARTIFACT_PREFERRED,
            TransferMode.ARTIFACT_REQUIRED,
        )

        with tempfile.TemporaryDirectory(prefix="superweb-spatial-task-") as temp_dir:
            temp_root = Path(temp_dir)
            weight_path = temp_root / "weight_slice.bin"
            weight_path.write_bytes(task.weight_data)
            output_path = temp_root / "output_slice.bin"
            if disk_first_result:
                output_fd, output_name = tempfile.mkstemp(prefix="superweb-spatial-output-", suffix=".bin")
                os.close(output_fd)
                output_path = Path(output_name)

            try:
                cmd = [
                    str(executable_path),
                    "--input",
                    str(input_path),
                    "--weight",
                    str(weight_path),
                    "--output",
                    str(output_path),
                    "--h",
                    str(spec.h),
                    "--w",
                    str(spec.w),
                    "--cin",
                    str(spec.c_in),
                    "--cout",
                    str(spec.c_out if backend_name == "cpu" else (task.end_oc - task.start_oc)),
                    "--k",
                    str(spec.k),
                    "--pad",
                    str(spec.pad),
                    "--stride",
                    str(spec.stride),
                    "--autotune-repeats",
                    "1",
                    "--measurement-repeats",
                    str(max(1, task.iteration_count)),
                ]
                if backend_name == "cpu":
                    configured_workers = int(
                        best_config.get("workers")
                        or best_config.get("requested_workers")
                        or (os.cpu_count() or 1)
                    )
                    worker_count = max(1, configured_workers)
                    cmd.extend(
                        [
                            "--start-oc",
                            str(task.start_oc),
                            "--end-oc",
                            str(task.end_oc),
                            "--workers",
                            str(worker_count),
                        ]
                    )
                elif backend_name == "cuda":
                    configured_batch = int(best_config.get("output_channel_batch") or 0)
                    resolved_batch = max(
                        1,
                        min(task.end_oc - task.start_oc, configured_batch or (task.end_oc - task.start_oc)),
                    )
                    cmd.extend(
                        [
                            "--output-channel-batch",
                            str(resolved_batch),
                            "--cooldown-ms",
                            str(CONV2D_CUDA_COOLDOWN_MS),
                        ]
                    )
                    cuda_block_size = int(best_config.get("block_size") or 0)
                    cuda_tile_size = int(best_config.get("tile_size") or 0)
                    if cuda_block_size > 0 and cuda_tile_size > 0:
                        cmd.extend(
                            [
                                "--block-sizes",
                                str(cuda_block_size),
                                "--tile-sizes",
                                str(cuda_tile_size),
                            ]
                        )
                    cuda_shared_input = best_config.get("shared_input")
                    if cuda_shared_input is not None:
                        cmd.extend(["--shared-input", str(int(cuda_shared_input))])
                elif backend_name == "metal":
                    block_size = int(best_config.get("block_size") or 256)
                    tile_size = int(best_config.get("tile_size") or 16)
                    slice_channels = task.end_oc - task.start_oc
                    output_channel_batch = int(best_config.get("output_channel_batch") or slice_channels)
                    cmd.extend(
                        [
                            "--block-sizes",
                            str(block_size),
                            "--tile-sizes",
                            str(tile_size),
                            "--output-channel-batch",
                            str(output_channel_batch),
                        ]
                    )

                subprocess_started_at = time.monotonic()
                try:
                    completed = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        cwd=METHOD_DIR,
                        timeout=900.0,
                    )
                except subprocess.CalledProcessError as exc:
                    message = _format_runner_failure_message(
                        method="conv2d",
                        backend_name=backend_name,
                        task=task,
                        returncode=exc.returncode,
                        stderr=exc.stderr,
                        stdout=exc.stdout,
                        elapsed_ms=int((time.monotonic() - subprocess_started_at) * 1000),
                    )
                    _LOGGER.error("%s", message)
                    raise RunnerProcessError(message) from exc
                except subprocess.TimeoutExpired as exc:
                    message = _format_runner_timeout_message(
                        method="conv2d",
                        backend_name=backend_name,
                        task=task,
                        timeout=exc.timeout,
                        stderr=exc.stderr,
                        stdout=exc.stdout,
                    )
                    _LOGGER.error("%s", message)
                    raise RunnerProcessError(message) from exc
                subprocess_wall_ms = max(
                    0, int((time.monotonic() - subprocess_started_at) * 1000)
                )
                compute_event_ms = _parse_compute_event_ms(completed.stdout)
                if compute_event_ms is not None:
                    # The CUDA runner's dispatch mode reports cudaEvent-bracketed
                    # GPU time. Using it instead of the subprocess wall keeps
                    # the supervisor's computation_ms aligned with what the
                    # benchmark's measurement pass measures, so dispatch-time
                    # per-channel throughput compares directly to the
                    # benchmark-reported capacity.
                    computation_ms_total += min(subprocess_wall_ms, compute_event_ms)
                else:
                    computation_ms_total += subprocess_wall_ms
                if not output_path.exists():
                    raise RuntimeError(
                        f"{backend_name} conv2d runner completed without writing {output_path.name}: "
                        f"{(completed.stdout or '').strip()}"
                    )

                output_size = output_path.stat().st_size
                conv_spec = BenchmarkSpec(
                    name="task",
                    h=spec.h,
                    w=spec.w,
                    c_in=spec.c_in,
                    c_out=task.end_oc - task.start_oc,
                    k=spec.k,
                    pad=spec.pad,
                    ideal_seconds=1.0,
                    zero_score_seconds=10.0,
                    stride=spec.stride,
                )
                expected_output_bytes = conv_spec.output_bytes
                if output_size != expected_output_bytes:
                    raise ValueError(
                        f"conv2d runtime output has {output_size} bytes, expected {expected_output_bytes}"
                    )

                conv2d_task_payload = task.conv2d_payload
                client_response_mode = (
                    int(conv2d_task_payload.client_response_mode) if conv2d_task_payload is not None else 0
                )
                stats_max_samples = (
                    int(conv2d_task_payload.stats_max_samples) if conv2d_task_payload is not None else 0
                )
                stats_only = client_response_mode == CONV2D_CLIENT_RESPONSE_STATS_ONLY
                if stats_only:
                    effective_max_samples = stats_max_samples if stats_max_samples > 0 else CONV2D_STATS_MAX_SAMPLES
                    stats_count, stats_sum, stats_sum_squares, stats_samples = _summarize_conv2d_slice_file(
                        output_path,
                        max_samples=effective_max_samples,
                    )
                    expected_length = conv_spec.output_h * conv_spec.output_w * (task.end_oc - task.start_oc)
                    if stats_count != expected_length:
                        raise ValueError(
                            f"conv2d slice element count {stats_count} does not match expected {expected_length}"
                        )
                    if disk_first_result:
                        output_path.unlink(missing_ok=True)
                    output_bytes = b""
                    local_result_path = ""
                else:
                    output_bytes = b"" if disk_first_result else output_path.read_bytes()
                    local_result_path = str(output_path) if disk_first_result else ""
                    stats_count = 0
                    stats_sum = 0.0
                    stats_sum_squares = 0.0
                    stats_samples: tuple[float, ...] = ()
            except Exception:
                if disk_first_result:
                    output_path.unlink(missing_ok=True)
                raise

        output_length = conv_spec.output_h * conv_spec.output_w * (task.end_oc - task.start_oc)
        wall_ms = max(0, int((time.monotonic() - task_started_at) * 1000))
        peripheral_ms = max(0, wall_ms - computation_ms_total)
        if stats_only:
            result_payload = Conv2dResultPayload(
                start_oc=task.start_oc,
                end_oc=task.end_oc,
                output_h=conv_spec.output_h,
                output_w=conv_spec.output_w,
                output_length=output_length,
                output_vector=b"",
                result_artifact_id="",
                stats_element_count=stats_count,
                stats_sum=stats_sum,
                stats_sum_squares=stats_sum_squares,
                stats_samples=stats_samples,
            )
            return TaskResult(
                request_id=task.request_id,
                node_id=task.node_id,
                task_id=task.task_id,
                timestamp_ms=task.timestamp_ms,
                status_code=STATUS_OK,
                iteration_count=task.iteration_count,
                result_payload=result_payload,
                local_result_path="",
                computation_ms=computation_ms_total,
                peripheral_ms=peripheral_ms,
            )
        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=STATUS_OK,
            local_result_path=local_result_path,
            row_start=0,
            row_end=0,
            output_length=output_length,
            output_vector=output_bytes,
            iteration_count=task.iteration_count,
            start_oc=task.start_oc,
            end_oc=task.end_oc,
            output_h=conv_spec.output_h,
            output_w=conv_spec.output_w,
            computation_ms=computation_ms_total,
            peripheral_ms=peripheral_ms,
        )
