"""Execute GEMM runtime tasks on the local compute node.

Use this module when a compute node receives a GEMM ``TaskAssign`` and
needs to run a pre-built GEMM runner against its locally generated A and
B matrices, then return the assigned M-axis slice of C. Prefers the
cuBLAS runner when that binary is available; otherwise falls back to the
self-contained threaded CPU runner so hosts without an NVIDIA toolchain
can still carry GEMM work.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from core.constants import BACKEND_CPU, BACKEND_CUDA, METHOD_GEMM, STATUS_OK
from core.process_exit import classify_exit_code
from compute_node.compute_methods.gemm.paths import (
    CUDA_EXECUTABLE_PATH,
    GEMM_METHOD_DIR,
    current_cpu_executable_path,
)
from compute_node.input_matrix.gemm import (
    build_dataset_layout,
    build_spec,
    dataset_is_generated,
    dataset_prefix_for_size,
    normalize_size_variant,
)
from compute_node.performance_metrics.gemm.config import DATASET_DIR as DEFAULT_DATASET_DIR
from wire.internal_protocol.control_plane import GemmResultPayload
from wire.internal_protocol.transport import TaskAssign, TaskResult

_LOGGER = logging.getLogger(__name__)
_RUNNER_STDERR_TAIL_BYTES = 2048
_RUNNER_TIMEOUT_SECONDS = 900.0


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
    """Return ``compute_event_ms`` from a runner's JSON stdout, or None.

    Both the cuBLAS and the CPU runner emit a single-line JSON record whose
    ``compute_event_ms`` is the kernel-loop wall time (cudaEvent-bracketed
    for cuBLAS, ``steady_clock`` for CPU) aggregated across
    ``iteration_count`` calls. Using this over subprocess wall-clock keeps
    capacity math consistent across backends and methods.
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


@dataclass(slots=True)
class _ResolvedTask:
    """Resolved spec and dataset layout for one incoming GEMM task."""

    spec: object  # GemmSpec
    dataset_layout: object  # DatasetLayout


class GemmTaskExecutor:
    """Execute one GEMM task on the local compute node (cuBLAS or CPU)."""

    def __init__(
        self,
        *,
        dataset_root: Path | None = None,
        pinned_backend: str | None = None,
    ) -> None:
        """Store the dataset root and optional backend pin used at runtime.

        Args:
            dataset_root: Optional GEMM dataset directory override. When
                ``None``, the canonical dataset directory under
                ``compute_node/input_matrix/gemm/generated`` is used.
            pinned_backend: Optional ``"cpu"`` or ``"cuda"`` restriction.
                Peer subprocesses advertising a specific backend must not
                dispatch to a different backend's runner just because the
                binary happens to exist on the shared parent-host filesystem
                — without this pin a peer-cpu subprocess co-located with a
                peer-cuda subprocess would happily pick the cuBLAS runner,
                contend for the same GPU, and skew the ``cpu:N GFLOPS`` line
                the main node saw at registration time.
        """
        self.dataset_root = DEFAULT_DATASET_DIR if dataset_root is None else Path(dataset_root)
        self.pinned_backend = pinned_backend

    def execute_task(self, task: TaskAssign) -> TaskResult:
        """Run one assigned GEMM M-axis slice and return its output bytes.

        Args:
            task: GEMM task assignment received from the main node.

        Returns:
            A ``TaskResult`` containing the merged GEMM output slice.
        """
        task_started_at = time.monotonic()
        resolved = self._resolve_task(task)
        self._validate_task(task, resolved)

        executable_path, backend_label = self._select_runner_executable()

        # Each concurrent slice on this host gets its own temp directory so
        # peer subprocesses (peer-cuda, standalone cuda, peer-cpu all on the
        # same box in dual-purpose mode) can't clobber one another's C output
        # file. Earlier revisions derived the output path from task_id alone
        # — which collides across peers because task_id is request-scoped,
        # not worker-scoped — and the last writer on a heterogeneous cluster
        # (the CPU peer finishing its few-row slice after the CUDA peers)
        # would silently truncate the CUDA output, tripping the size-check
        # here and killing both CUDA workers via failover. Mirrors the
        # tempfile pattern gemv's task_executor already uses.
        with tempfile.TemporaryDirectory(prefix="gemm-task-") as temp_dir:
            output_path = Path(temp_dir) / "C.bin"
            command = self._build_runtime_command(
                task,
                spec=resolved.spec,
                dataset_layout=resolved.dataset_layout,
                output_path=output_path,
                executable_path=executable_path,
            )
            subprocess_started_at = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=GEMM_METHOD_DIR,
                    timeout=_RUNNER_TIMEOUT_SECONDS,
                )
            except subprocess.CalledProcessError as exc:
                _LOGGER.error(
                    "gemm native runner failed: task_id=%s rows=[%d, %d) returncode=%s cause=\"%s\" "
                    "stderr_tail=%r stdout_tail=%r",
                    task.task_id,
                    task.m_start,
                    task.m_end,
                    exc.returncode,
                    classify_exit_code(exc.returncode),
                    _tail_stream(exc.stderr),
                    _tail_stream(exc.stdout),
                )
                raise
            except subprocess.TimeoutExpired as exc:
                _LOGGER.error(
                    "gemm native runner timed out: task_id=%s rows=[%d, %d) timeout=%.1fs "
                    "stderr_tail=%r stdout_tail=%r",
                    task.task_id,
                    task.m_start,
                    task.m_end,
                    float(exc.timeout or 0.0),
                    _tail_stream(exc.stderr),
                    _tail_stream(exc.stdout),
                )
                raise
            subprocess_wall_ms = max(0, int((time.monotonic() - subprocess_started_at) * 1000))
            compute_event_ms = _parse_compute_event_ms(completed.stdout)
            # Prefer runner-reported kernel time so computation_ms scales with
            # iteration_count instead of being dominated by one-time A/B load.
            if compute_event_ms is not None:
                computation_ms = min(subprocess_wall_ms, compute_event_ms)
            else:
                computation_ms = subprocess_wall_ms

            if not output_path.exists():
                raise RuntimeError(
                    f"GEMM {backend_label} runner completed without writing {output_path.name}: "
                    f"{(completed.stdout or '').strip()}"
                )
            slice_rows = task.m_end - task.m_start
            expected_bytes = slice_rows * resolved.spec.n * 4
            output_bytes = output_path.read_bytes()
            if len(output_bytes) != expected_bytes:
                raise ValueError(
                    f"GEMM {backend_label} runner output has {len(output_bytes)} bytes, expected {expected_bytes}"
                )

        output_length = slice_rows * resolved.spec.n
        wall_ms = max(0, int((time.monotonic() - task_started_at) * 1000))
        peripheral_ms = max(0, wall_ms - computation_ms)
        result_payload = GemmResultPayload(
            m_start=task.m_start,
            m_end=task.m_end,
            output_length=output_length,
            output_vector=output_bytes,
        )
        return TaskResult(
            request_id=task.request_id,
            node_id=task.node_id,
            task_id=task.task_id,
            timestamp_ms=task.timestamp_ms,
            status_code=STATUS_OK,
            iteration_count=task.iteration_count,
            result_payload=result_payload,
            computation_ms=computation_ms,
            peripheral_ms=peripheral_ms,
        )

    def close(self) -> None:
        """Release any long-lived helpers owned by the executor (none currently)."""
        return None

    def _resolve_task(self, task: TaskAssign) -> _ResolvedTask:
        """Resolve the named GEMM dataset variant referenced by one task."""
        variant = normalize_size_variant(getattr(task, "size", ""), default="large")
        spec = build_spec(default_variant=variant)
        dataset_layout = build_dataset_layout(
            self.dataset_root,
            prefix=dataset_prefix_for_size(variant, default="large"),
        )
        return _ResolvedTask(spec=spec, dataset_layout=dataset_layout)

    def _validate_task(self, task: TaskAssign, resolved: _ResolvedTask) -> None:
        """Validate that a GEMM task matches the local runtime dataset."""
        if task.method != METHOD_GEMM:
            raise ValueError(f"unsupported task method: {task.method}")
        spec = resolved.spec
        if task.m and task.m != spec.m:
            raise ValueError(f"task m={task.m} does not match expected {spec.m}")
        if task.n and task.n != spec.n:
            raise ValueError(f"task n={task.n} does not match expected {spec.n}")
        if task.k and task.k != spec.k:
            raise ValueError(f"task k={task.k} does not match expected {spec.k}")
        if task.m_start < 0 or task.m_end > spec.m or task.m_end <= task.m_start:
            raise ValueError(
                f"task m range [{task.m_start}, {task.m_end}) is invalid for M={spec.m}"
            )
        if task.iteration_count <= 0:
            raise ValueError("task iteration_count must be positive")
        if not dataset_is_generated(resolved.dataset_layout, spec):
            raise FileNotFoundError(
                f"missing generated GEMM dataset at {resolved.dataset_layout.root_dir}; "
                f"run compute_node/input_matrix/generate.py --method gemm"
            )

    def _select_runner_executable(self) -> tuple[Path, str]:
        """Return ``(executable_path, backend_label)`` for the GEMM runner.

        Selection rules:

        - If ``pinned_backend == "cpu"``: always use the CPU runner. A peer
          process that advertises CPU capacity must never fall through to
          the cuBLAS binary even when it's present on the shared filesystem
          — doing so would dispatch against a GPU the peer is not supposed
          to be using, and the host's CPU GFLOPS line in the registry would
          no longer match the capacity actually being consumed.
        - If ``pinned_backend == "cuda"`` (or unpinned): prefer the cuBLAS
          binary when it's built (an order of magnitude faster than CPU).
          When the cuBLAS binary isn't present — host without NVIDIA
          toolchain, or first run before ``--rebuild`` — fall back to the
          CPU runner so non-NVIDIA hosts can still join the cluster as
          GEMM workers.
        """
        pinned = self.pinned_backend
        if pinned == BACKEND_CPU:
            try:
                cpu_path = current_cpu_executable_path()
            except RuntimeError as exc:
                raise FileNotFoundError(
                    f"no GEMM CPU runner is available: this worker is pinned to cpu but no "
                    f"CPU runner is configured for this platform ({exc})"
                ) from exc
            if not cpu_path.exists():
                raise FileNotFoundError(
                    f"no GEMM CPU runner is available: pinned to cpu but CPU binary is missing "
                    f"at {cpu_path}; run performance_metrics/gemm to build one"
                )
            return cpu_path, "cpu"

        if pinned == BACKEND_CUDA:
            if not CUDA_EXECUTABLE_PATH.exists():
                raise FileNotFoundError(
                    f"no GEMM CUDA runner is available: pinned to cuda but cuBLAS binary is "
                    f"missing at {CUDA_EXECUTABLE_PATH}; run performance_metrics/gemm with "
                    f"--rebuild to build one"
                )
            return CUDA_EXECUTABLE_PATH, "cuda"

        if CUDA_EXECUTABLE_PATH.exists():
            return CUDA_EXECUTABLE_PATH, "cuda"
        try:
            cpu_path = current_cpu_executable_path()
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"no GEMM runner is available: cuBLAS binary at {CUDA_EXECUTABLE_PATH} is missing "
                f"and no CPU runner is configured for this platform ({exc})"
            ) from exc
        if not cpu_path.exists():
            raise FileNotFoundError(
                f"no GEMM runner is available: cuBLAS binary is missing at {CUDA_EXECUTABLE_PATH} "
                f"and CPU fallback binary is missing at {cpu_path}; run "
                f"performance_metrics/gemm to build one"
            )
        return cpu_path, "cpu"

    def _build_runtime_command(
        self,
        task: TaskAssign,
        *,
        spec,
        dataset_layout,
        output_path: Path,
        executable_path: Path,
    ) -> list[str]:
        """Build the GEMM runner command line for one M-axis slice.

        The CPU and CUDA runners share the same CLI surface so dispatch code
        can ignore which backend is actually running.
        """
        return [
            str(executable_path),
            "--input-a",
            str(dataset_layout.a_path),
            "--input-b",
            str(dataset_layout.b_path),
            "--output",
            str(output_path),
            "--m",
            str(spec.m),
            "--n",
            str(spec.n),
            "--k",
            str(spec.k),
            "--m-start",
            str(task.m_start),
            "--m-end",
            str(task.m_end),
            # Task mode uses iteration_count to repeat the same math locally
            # without resending the client request through the cluster.
            "--iteration-count",
            str(task.iteration_count),
            "--mode",
            "dispatch",
        ]
