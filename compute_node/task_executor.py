"""Execute GEMV runtime tasks across the local compute-node processors.

Use this module when a compute node receives an GEMV ``TaskAssign`` and needs
to split the row range across its locally benchmarked processors.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from core.process_exit import classify_exit_code

from core.constants import DX12_BACKEND_DISABLED_REASON
from core.work_partition import partition_contiguous_range
from compute_node.compute_methods.gemv import (
    CUDA_EXECUTABLE_PATH,
    GEMV_METHOD_DIR,
)
from compute_node.input_matrix.gemv import (
    build_dataset_layout,
    build_input_matrix_spec,
    dataset_prefix_for_size,
    dataset_is_generated,
    normalize_size_variant,
)
from compute_node.performance_metrics.gemv.config import DATASET_DIR as GEMV_DATASET_DIR
from compute_node.performance_metrics.performance_summary import RuntimeProcessorInventory, RuntimeProcessorProfile, load_runtime_processor_inventory
from core.constants import METHOD_GEMV
from wire.internal_protocol.transport import TaskAssign, TaskResult

ROOT_DIR = Path(__file__).resolve().parent
INPUT_MATRIX_GENERATED_DIR = GEMV_DATASET_DIR

_LOGGER = logging.getLogger(__name__)
_RUNNER_STDERR_TAIL_BYTES = 2048


def _tail_stream(payload: str | bytes | None, *, limit: int = _RUNNER_STDERR_TAIL_BYTES) -> str:
    """Return the trailing portion of a captured runner stream, safe to log."""
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


def _parse_compute_event_ms(stdout: str | bytes | None) -> int | None:
    """Return `compute_event_ms` from a runner's JSON stdout, or None.

    The gemv runners in task mode emit a JSON record with
    `compute_event_ms = wall_clock_latency_seconds × iteration_count × 1000`,
    i.e. the steady-state kernel time aggregated across all iterations, with
    file I/O and process startup excluded. The executor prefers this over
    subprocess wall-clock so the reported computation_ms scales with
    iteration_count instead of being dominated by one-time matrix load.
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


def _log_gemv_runner_failure(
    *,
    backend_name: str,
    task_id: str,
    row_start: int,
    row_end: int,
    returncode: int | None,
    stderr: str | bytes | None,
    stdout: str | bytes | None,
) -> None:
    """Emit a single ERROR line summarizing a nonzero gemv runner exit.

    Why: the gemv native runner runs in a worker subprocess; without explicit
    capture, its CalledProcessError propagates with no log line and the
    operator only sees that the worker disappeared.
    """
    classification = classify_exit_code(returncode)
    _LOGGER.error(
        "gemv native runner failed: backend=%s task_id=%s rows=[%d, %d) returncode=%s cause=\"%s\" "
        "stderr_tail=%r stdout_tail=%r",
        backend_name,
        task_id,
        row_start,
        row_end,
        returncode,
        classification,
        _tail_stream(stderr),
        _tail_stream(stdout),
    )


def _log_gemv_runner_timeout(
    *,
    backend_name: str,
    task_id: str,
    row_start: int,
    row_end: int,
    timeout: float | None,
    stderr: str | bytes | None,
    stdout: str | bytes | None,
) -> None:
    """Emit a single ERROR line when the gemv runner blew its wall-clock budget."""
    _LOGGER.error(
        "gemv native runner timed out: backend=%s task_id=%s rows=[%d, %d) timeout=%.1fs "
        "stderr_tail=%r stdout_tail=%r",
        backend_name,
        task_id,
        row_start,
        row_end,
        float(timeout or 0.0),
        _tail_stream(stderr),
        _tail_stream(stdout),
    )


@dataclass(frozen=True, slots=True)
class ProcessorTaskSlice:
    """Describe one local processor and the GEMV rows assigned to it."""

    processor: RuntimeProcessorProfile
    row_start: int
    row_end: int


class _Dx12ResidentRunner:
    """Hold a DX12 matrix resident across multiple local task slices."""

    def __init__(self, dataset_layout, spec, processors: tuple[RuntimeProcessorProfile, ...]) -> None:
        """Start the DX12 resident runner for all local DX12 processors.

        Args:
            dataset_layout: Runtime GEMV dataset layout shared by the compute node.
            spec: Runtime GEMV benchmark specification.
            processors: Local processor profiles that use the DX12 backend.
        """
        thread_group_sizes = sorted(
            {
                int(processor.best_config.get("thread_group_size") or 256)
                for processor in processors
                if processor.hardware_type == "dx12"
            }
        )
        if not thread_group_sizes:
            raise ValueError("DX12 resident runner requires at least one dx12 processor profile")

        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                str(DX12_EXECUTABLE_PATH),
                "--server",
                "1",
                "--matrix",
                str(dataset_layout.matrix_path),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--thread-group-sizes",
                ",".join(str(value) for value in thread_group_sizes),
            ],
            cwd=GEMV_METHOD_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        ready_line = self._read_response_line()
        if not ready_line.startswith("READY\t"):
            stderr = self._process.stderr.read().strip() if self._process.stderr is not None else ""
            raise RuntimeError(f"DX12 resident runner failed to start: {ready_line or stderr or 'unknown error'}")
        atexit.register(self.close)

    def run_slice(
        self,
        *,
        vector_path: Path,
        output_path: Path,
        row_start: int,
        row_end: int,
        thread_group_size: int,
        rows_per_thread: int,
        iteration_count: int,
    ) -> dict[str, object]:
        """Execute one GEMV slice through the resident DX12 runner.

        Args:
            vector_path: Temporary vector file for this task.
            output_path: Destination file for the slice output.
            row_start: Inclusive starting row for the slice.
            row_end: Exclusive ending row for the slice.
            thread_group_size: DX12 thread-group size to use.
            rows_per_thread: DX12 rows-per-thread setting.
            iteration_count: Number of times to repeat the math locally.

        Returns:
            Parsed JSON metrics reported by the DX12 resident runner.
        """
        with self._lock:
            self._ensure_running()
            command = "\t".join(
                [
                    "RUN",
                    str(vector_path),
                    str(output_path),
                    str(row_start),
                    str(row_end),
                    str(thread_group_size),
                    str(rows_per_thread),
                    str(iteration_count),
                ]
            )
            assert self._process.stdin is not None
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
            response_line = self._read_response_line()
            if response_line.startswith("OK\t"):
                return json.loads(response_line.split("\t", 1)[1])
            if response_line.startswith("ERR\t"):
                raise RuntimeError(response_line.split("\t", 1)[1])
            raise RuntimeError(f"unexpected DX12 resident runner response: {response_line}")

    def close(self) -> None:
        """Shut down the resident DX12 runner process if it is still alive."""
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
            except OSError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._process = None

    def _ensure_running(self) -> None:
        """Raise if the resident DX12 runner exited unexpectedly."""
        if self._process.poll() is not None:
            stderr = self._process.stderr.read().strip() if self._process.stderr is not None else ""
            raise RuntimeError(f"DX12 resident runner exited unexpectedly: {stderr or self._process.returncode}")

    def _read_response_line(self) -> str:
        """Read one protocol line from the resident DX12 runner."""
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read().strip() if self._process.stderr is not None else ""
            raise RuntimeError(f"DX12 resident runner closed its pipe unexpectedly: {stderr or 'no output'}")
        return line.rstrip("\r\n")


class GemvTaskExecutor:
    """Execute one fixed-matrix-vector task on the local compute node."""

    def __init__(
        self,
        inventory: RuntimeProcessorInventory | None = None,
        *,
        dataset_root: Path | None = None,
        pinned_backend: str | None = None,
    ) -> None:
        """Load the local processor inventory and runtime dataset paths.

        Args:
            inventory: Optional local processor inventory override.
            dataset_root: Optional runtime dataset directory override.
            pinned_backend: Optional backend name restricting the inventory
                when one is not supplied directly.
        """
        self.inventory = inventory or load_runtime_processor_inventory(
            pinned_backend=pinned_backend
        )
        self.dataset_root = INPUT_MATRIX_GENERATED_DIR if dataset_root is None else Path(dataset_root)
        self._dx12_runner = self._build_dx12_resident_runner()
        self._resolved_executable_paths: dict[str, Path] = {}

    def close(self) -> None:
        """Release any long-lived helper processes owned by the executor."""
        if self._dx12_runner is not None:
            self._dx12_runner.close()
            self._dx12_runner = None

    def execute_task(self, task: TaskAssign) -> TaskResult:
        """Run one assigned row slice and return its output bytes.

        Args:
            task: GEMV task assignment received from the main node.

        Returns:
            A ``TaskResult`` containing the merged GEMV output slice.
        """

        task_started_at = time.monotonic()
        spec, dataset_layout = self._resolve_task_dataset(task)
        self._validate_task(task, spec=spec, dataset_layout=dataset_layout)
        processor_slices = self._build_local_processor_slices(task)

        with tempfile.TemporaryDirectory(prefix="superweb-task-") as temp_dir:
            temp_root = Path(temp_dir)
            vector_path = temp_root / "x.bin"
            vector_path.write_bytes(task.vector_data)

            def _run_and_time(processor_slice):
                slice_started_at = time.monotonic()
                chunk, compute_event_ms = self._run_processor_slice(
                    processor_slice,
                    task.iteration_count,
                    vector_path,
                    temp_root,
                    spec=spec,
                    dataset_layout=dataset_layout,
                    task_id=task.task_id,
                )
                subprocess_wall_ms = max(0, int((time.monotonic() - slice_started_at) * 1000))
                # Prefer the runner-reported kernel time (per-iter latency ×
                # iteration_count) over subprocess wall; the latter is dominated
                # by one-time matrix load and would hide iteration_count scaling.
                if compute_event_ms is not None:
                    slice_ms = min(subprocess_wall_ms, compute_event_ms)
                else:
                    slice_ms = subprocess_wall_ms
                return processor_slice.row_start, chunk, slice_ms

            with ThreadPoolExecutor(max_workers=len(processor_slices), thread_name_prefix="local-processor") as executor:
                slice_results = list(executor.map(_run_and_time, processor_slices))

        slice_results.sort(key=lambda item: item[0])
        merged = b"".join(chunk for _, chunk, _ in slice_results)
        wall_ms = max(0, int((time.monotonic() - task_started_at) * 1000))
        computation_ms = max((slice_ms for _, _, slice_ms in slice_results), default=0)
        peripheral_ms = max(0, wall_ms - computation_ms)
        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=200,
            row_start=task.row_start,
            row_end=task.row_end,
            output_length=task.row_end - task.row_start,
            output_vector=merged,
            iteration_count=task.iteration_count,
            computation_ms=computation_ms,
            peripheral_ms=peripheral_ms,
        )

    def _resolve_task_dataset(self, task: TaskAssign):
        """Resolve the named GEMV dataset variant referenced by one task.

        Args:
            task: GEMV task assignment that may carry a canonical size string.

        Returns:
            A tuple of ``(spec, dataset_layout)`` for the requested task size.
        """
        variant = normalize_size_variant(getattr(task, "size", ""), default="large")
        spec = build_input_matrix_spec(default_variant=variant)
        dataset_layout = build_dataset_layout(
            self.dataset_root,
            prefix=dataset_prefix_for_size(variant, default="large"),
        )
        return spec, dataset_layout

    def _validate_task(self, task: TaskAssign, *, spec, dataset_layout) -> None:
        """Validate that an GEMV task matches the local runtime dataset.

        Args:
            task: GEMV task assignment to validate.
            spec: Resolved GEMV dataset specification for this task size.
            dataset_layout: Resolved GEMV dataset layout for this task size.

        Returns:
            ``None`` when the task is valid for local execution.
        """
        if task.method != METHOD_GEMV:
            raise ValueError(f"unsupported task method: {task.method}")
        if task.row_start < 0 or task.row_end > spec.rows or task.row_end <= task.row_start:
            raise ValueError("task row range is invalid")
        if task.vector_length != spec.cols:
            raise ValueError(f"task vector length {task.vector_length} does not match expected {spec.cols}")
        if len(task.vector_data) != task.vector_length * 4:
            raise ValueError("task vector byte size does not match vector_length")
        if task.iteration_count <= 0:
            raise ValueError("task iteration_count must be positive")
        if not dataset_is_generated(dataset_layout, spec):
            raise FileNotFoundError(
                f"missing generated input matrix at {dataset_layout.root_dir}; run compute_node/input_matrix/generate.py"
            )
        if not self.inventory.processors:
            raise RuntimeError("no locally registered processors are available for task execution")

    def _build_local_processor_slices(self, task: TaskAssign) -> list[ProcessorTaskSlice]:
        """Partition the GEMV row range across local processor profiles.

        Args:
            task: GEMV task assignment received from the main node.

        Returns:
            Processor-task slices ordered by local inventory.
        """
        partitions = partition_contiguous_range(
            task.row_start,
            task.row_end,
            [processor.effective_gflops for processor in self.inventory.processors],
        )
        slices: list[ProcessorTaskSlice] = []
        for partition, processor in zip(partitions, self.inventory.processors):
            if partition.end <= partition.start:
                continue
            slices.append(
                ProcessorTaskSlice(
                    processor=processor,
                    row_start=partition.start,
                    row_end=partition.end,
                )
            )
        return slices

    def _run_processor_slice(
        self,
        processor_slice: ProcessorTaskSlice,
        iteration_count: int,
        vector_path: Path,
        temp_root: Path,
        *,
        spec,
        dataset_layout,
        task_id: str = "?",
    ) -> tuple[bytes, int | None]:
        """Run one processor-specific GEMV slice.

        Args:
            processor_slice: Local processor slice to execute.
            iteration_count: Number of times to repeat the local computation.
            vector_path: Temporary vector file for this task.
            temp_root: Temporary directory for per-slice outputs.
            spec: Resolved GEMV dataset specification for this task size.
            dataset_layout: Resolved GEMV dataset layout for this task size.

        Returns:
            A tuple of ``(raw_output_bytes, compute_event_ms)``. The second
            element is the runner-reported kernel time in milliseconds or
            ``None`` when the runner did not provide one (e.g. DX12 resident
            runner).
        """
        output_path = temp_root / (
            f"{processor_slice.processor.hardware_type}_{processor_slice.row_start}_{processor_slice.row_end}.bin"
        )
        compute_event_ms: int | None = None
        if processor_slice.processor.hardware_type == "dx12" and self._dx12_runner is not None:
            self._dx12_runner.run_slice(
                vector_path=vector_path,
                output_path=output_path,
                row_start=processor_slice.row_start,
                row_end=processor_slice.row_end,
                thread_group_size=int(processor_slice.processor.best_config.get("thread_group_size") or 256),
                rows_per_thread=int(processor_slice.processor.best_config.get("rows_per_thread") or 1),
                iteration_count=iteration_count,
            )
        else:
            command = self._build_runtime_command(
                processor_slice,
                iteration_count,
                vector_path,
                output_path,
                spec=spec,
                dataset_layout=dataset_layout,
            )
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=GEMV_METHOD_DIR,
                    timeout=300.0,
                )
            except subprocess.CalledProcessError as exc:
                _log_gemv_runner_failure(
                    backend_name=processor_slice.processor.hardware_type,
                    task_id=task_id,
                    row_start=processor_slice.row_start,
                    row_end=processor_slice.row_end,
                    returncode=exc.returncode,
                    stderr=exc.stderr,
                    stdout=exc.stdout,
                )
                raise
            except subprocess.TimeoutExpired as exc:
                _log_gemv_runner_timeout(
                    backend_name=processor_slice.processor.hardware_type,
                    task_id=task_id,
                    row_start=processor_slice.row_start,
                    row_end=processor_slice.row_end,
                    timeout=exc.timeout,
                    stderr=exc.stderr,
                    stdout=exc.stdout,
                )
                raise
            if not output_path.exists():
                stdout = (completed.stdout or "").strip()
                raise RuntimeError(
                    f"{processor_slice.processor.hardware_type} runtime executable completed without writing {output_path.name}: "
                    f"{stdout}"
                )
            compute_event_ms = _parse_compute_event_ms(completed.stdout)
        raw = output_path.read_bytes()
        expected_bytes = (processor_slice.row_end - processor_slice.row_start) * 4
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{processor_slice.processor.hardware_type} runtime output has {len(raw)} bytes, expected {expected_bytes}"
            )
        return raw, compute_event_ms

    def _build_dx12_resident_runner(self) -> _Dx12ResidentRunner | None:
        """Create the DX12 resident runner when the local inventory supports it.

        Returns:
            A resident DX12 runner instance, or ``None`` when not used.
        """
        dx12_processors = tuple(
            processor for processor in self.inventory.processors if processor.hardware_type == "dx12"
        )
        if not dx12_processors or sys.platform != "win32":
            return None
        return None

    def _build_runtime_command(
        self,
        processor_slice: ProcessorTaskSlice,
        iteration_count: int,
        vector_path: Path,
        output_path: Path,
        *,
        spec=None,
        dataset_layout=None,
    ) -> list[str]:
        """Build the native runtime command for one local processor slice.

        Args:
            processor_slice: Local processor slice to execute.
            iteration_count: Number of times to repeat the local computation.
            vector_path: Temporary vector file for this task.
            output_path: Destination file for the slice output.
            spec: Resolved GEMV dataset specification for this task size.
            dataset_layout: Resolved GEMV dataset layout for this task size.

        Returns:
            The subprocess command list for the selected backend.
        """
        if spec is None or dataset_layout is None:
            spec = build_input_matrix_spec(default_variant="large")
            dataset_layout = build_dataset_layout(
                self.dataset_root,
                prefix=dataset_prefix_for_size("large", default="large"),
            )
        processor = processor_slice.processor
        if processor.hardware_type == "cpu":
            executable_path = self._resolve_runtime_executable_path("cpu")
            configured_workers = int(
                processor.best_config.get("workers")
                or processor.best_config.get("requested_workers")
                or (os.cpu_count() or 1)
            )
            workers = max(1, configured_workers)
            tile_size = int(processor.best_config.get("tile_size") or spec.cols)
            accumulation_precision = str(processor.best_config.get("accumulation_precision") or "fp32")
            return [
                str(executable_path),
                "--matrix",
                str(dataset_layout.matrix_path),
                "--vector",
                str(vector_path),
                "--output",
                str(output_path),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--accumulation-precision",
                accumulation_precision,
                "--row-start",
                str(processor_slice.row_start),
                "--row-end",
                str(processor_slice.row_end),
                "--fixed-workers",
                str(workers),
                "--fixed-tile-size",
                str(tile_size),
                # Task mode uses iteration_count to repeat the same math locally
                # without resending the client request through the cluster.
                "--iteration-count",
                str(iteration_count),
            ]

        if processor.hardware_type == "metal":
            executable_path = self._resolve_runtime_executable_path("metal")
            block_size = int(processor.best_config.get("block_size") or 256)
            tile_size = int(processor.best_config.get("tile_size") or 1)
            slice_rows = processor_slice.row_end - processor_slice.row_start
            row_chunk_size = int(processor.best_config.get("row_chunk_size") or slice_rows)
            return [
                str(executable_path),
                "--matrix",
                str(dataset_layout.matrix_path),
                "--vector",
                str(vector_path),
                "--output",
                str(output_path),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--row-start",
                str(processor_slice.row_start),
                "--row-end",
                str(processor_slice.row_end),
                "--block-sizes",
                str(block_size),
                "--tile-sizes",
                str(tile_size),
                "--row-chunk-size",
                str(row_chunk_size),
                "--autotune-repeats",
                "1",
                "--iteration-count",
                str(iteration_count),
            ]

        if processor.hardware_type == "cuda":
            executable_path = self._resolve_runtime_executable_path("cuda")
            transpose = 1 if bool(processor.best_config.get("transpose")) else 0
            block_size = int(processor.best_config.get("block_size") or 256)
            tile_size = int(processor.best_config.get("tile_size") or 1)
            accumulation_precision = str(processor.best_config.get("accumulation_precision") or "fp32")
            return [
                str(executable_path),
                "--matrix",
                str(dataset_layout.matrix_path),
                "--vector",
                str(vector_path),
                "--output",
                str(output_path),
                "--rows",
                str(spec.rows),
                "--cols",
                str(spec.cols),
                "--accumulation-precision",
                accumulation_precision,
                "--row-start",
                str(processor_slice.row_start),
                "--row-end",
                str(processor_slice.row_end),
                "--fixed-transpose",
                str(transpose),
                "--fixed-block-size",
                str(block_size),
                "--fixed-tile-size",
                str(tile_size),
                # Task mode uses iteration_count to repeat the same math locally
                # without resending the client request through the cluster.
                "--iteration-count",
                str(iteration_count),
            ]

        if processor.hardware_type == "dx12":
            raise ValueError(DX12_BACKEND_DISABLED_REASON)

        raise ValueError(f"unsupported local processor type: {processor.hardware_type}")

    def _resolve_runtime_executable_path(self, hardware_type: str) -> Path:
        """Resolve one runtime runner path, compiling on demand when needed."""

        cached = self._resolved_executable_paths.get(hardware_type)
        if cached is not None:
            return cached

        if hardware_type == "cpu":
            from compute_node.performance_metrics.gemv.backends.cpu_backend import CpuBackend

            executable_path, _note = CpuBackend()._resolve_executable_path(force_rebuild=False)
        elif hardware_type == "metal":
            from compute_node.performance_metrics.gemv.backends.metal_backend import MetalBackend

            executable_path, _note = MetalBackend()._compile_if_needed(force_rebuild=False)
        elif hardware_type == "cuda":
            executable_path = CUDA_EXECUTABLE_PATH
        else:
            raise ValueError(f"unsupported local processor type: {hardware_type}")

        self._resolved_executable_paths[hardware_type] = executable_path
        return executable_path
