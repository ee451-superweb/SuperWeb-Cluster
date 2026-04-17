"""Assign task slices to workers and collect their results.

Use this module when the main node needs to send ``TASK_ASSIGN`` messages,
wait for ``TASK_ACCEPT`` and ``TASK_RESULT``, and fetch large results through
the artifact data plane.
"""

from __future__ import annotations

import hashlib
import re
import socket
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

from app.constants import (
    MAIN_NODE_NAME,
    RUNTIME_MSG_ARTIFACT_RELEASE,
    METHOD_SPATIAL_CONVOLUTION,
    RUNTIME_MSG_TASK_ACCEPT,
    RUNTIME_MSG_TASK_ASSIGN,
    RUNTIME_MSG_TASK_FAIL,
    RUNTIME_MSG_TASK_RESULT,
)
from app.trace_utils import trace_function
from compute_node.compute_methods.spatial_convolution import DATASET_GENERATE_SCRIPT_PATH as SPATIAL_DATASET_GENERATE_SCRIPT_PATH
from compute_node.compute_methods.spatial_convolution.executor import load_named_workload_spec
from main_node.dispatcher import WorkerTaskSlice
from main_node.runtime_mailbox import RuntimeConnectionMailbox
from wire.internal_protocol.runtime_transport import (
    FixedMatrixVectorTaskPayload,
    MessageKind,
    SpatialConvolutionTaskPayload,
    TransferMode,
    build_artifact_release,
    build_task_assign,
    describe_message_kind,
    recv_message,
    send_message,
)


class WorkerTaskExchange:
    """Encapsulate one worker-task assignment round trip."""

    def __init__(self, *, config, spatial_dataset_dir: Path, remove_worker_connection, artifact_manager=None) -> None:
        """Store runtime services used during worker task exchange.

        Args:
            config: Main-node runtime configuration with timeout values.
            spatial_dataset_dir: Directory that holds shared conv2d datasets.
            remove_worker_connection: Callback that removes failed workers.
            artifact_manager: Optional artifact manager for large results.
        """
        self.config = config
        self.spatial_dataset_dir = spatial_dataset_dir
        self._remove_worker_connection = remove_worker_connection
        self.artifact_manager = artifact_manager
        self._artifact_workers: dict[str, tuple[str, str]] = {}
        self._artifact_workers_lock = threading.Lock()

    def ensure_spatial_convolution_dataset_ready(self) -> None:
        """Generate the shared conv2d dataset if the main node is missing it.

        Use this before slicing spatial-convolution weights so the scheduler can
        always read the canonical runtime and test datasets from disk.

        Args:
            None.

        Returns:
            ``None`` after the required dataset files exist.
        """
        runtime_input = self.spatial_dataset_dir / "runtime_input.bin"
        runtime_weight = self.spatial_dataset_dir / "runtime_weight.bin"
        test_input = self.spatial_dataset_dir / "test_input.bin"
        test_weight = self.spatial_dataset_dir / "test_weight.bin"
        if runtime_input.exists() and runtime_weight.exists() and test_input.exists() and test_weight.exists():
            return

        subprocess.run(
            [
                sys.executable,
                str(SPATIAL_DATASET_GENERATE_SCRIPT_PATH),
                "--output-dir",
                str(self.spatial_dataset_dir),
                "--role",
                "main",
                "--include-runtime-weight",
            ],
            check=True,
            timeout=1800.0,
        )

    def spatial_weight_slice(self, *, variant: str, spec, start_oc: int, end_oc: int) -> bytes:
        """Read the requested output-channel slice from a conv2d weight file.

        Use this while building ``TASK_ASSIGN`` for spatial convolution so each
        worker receives only the weight channels it is supposed to compute.

        Args:
            variant: Dataset variant name such as ``test`` or ``runtime``.
            spec: Workload specification describing tensor dimensions.
            start_oc: Inclusive starting output-channel index.
            end_oc: Exclusive ending output-channel index.

        Returns:
            The raw bytes covering the requested weight-channel slice.
        """
        weight_path = self.spatial_dataset_dir / f"{variant}_weight.bin"
        if not weight_path.exists():
            raise FileNotFoundError(f"missing spatial_convolution weight file at {weight_path}")
        weight_data = weight_path.read_bytes()
        bytes_per_output_channel = spec.k * spec.k * spec.c_in * 4
        return weight_data[start_oc * bytes_per_output_channel:end_oc * bytes_per_output_channel]

    def max_spatial_channels_per_task(self, spec) -> int:
        """Estimate how many output channels fit in one inline task payload.

        Use this when partitioning conv2d work for control-plane delivery so the
        sliced weight bytes stay under the runtime message-size budget.

        Args:
            spec: Workload specification describing kernel and channel counts.

        Returns:
            The maximum number of output channels that fit in one task payload.
        """
        bytes_per_channel = spec.k * spec.k * spec.c_in * 4
        if bytes_per_channel <= 0:
            raise ValueError("spatial_convolution weight slice size must be positive")
        payload_budget = max(self.config.max_message_size - 4096, 1)
        return max(1, payload_budget // bytes_per_channel)

    def _wait_for_task_message(self, connection, task_id: str):
        """Wait for the next task-scoped message from one worker.

        Use this helper so newer mailbox-based routing and older direct socket
        reads share one call site inside the task exchange flow.

        Args:
            connection: Worker connection being waited on.
            task_id: Task id expected from the worker.

        Returns:
            The received runtime message, or ``None`` on EOF.
        """
        mailbox = getattr(connection, "mailbox", None)
        if isinstance(mailbox, RuntimeConnectionMailbox):
            return mailbox.wait_for_task_message(task_id, timeout=None)

        poll_interval = max(self.config.runtime_socket_timeout, 1.0)
        previous_timeout = connection.sock.gettimeout()
        try:
            connection.sock.settimeout(poll_interval)
            while True:
                try:
                    return recv_message(connection.sock, max_size=self.config.max_message_size)
                except socket.timeout:
                    continue
        finally:
            try:
                connection.sock.settimeout(previous_timeout)
            except OSError:
                pass

    def _resolve_connection_lock(self, lock, fallback=None):
        """Return a usable context manager for worker task/I/O locks.

        Args:
            lock: Preferred lock-like object.
            fallback: Optional fallback lock-like object.

        Returns:
            The first candidate that supports ``with``, otherwise ``nullcontext()``.
        """
        lock_type = type(lock)
        if callable(getattr(lock_type, "__enter__", None)) and callable(getattr(lock_type, "__exit__", None)):
            return lock
        fallback_type = type(fallback)
        if callable(getattr(fallback_type, "__enter__", None)) and callable(getattr(fallback_type, "__exit__", None)):
            return fallback
        return nullcontext()

    def _record_artifact_owner(self, assignment: WorkerTaskSlice) -> None:
        """Remember which worker is expected to publish one artifact id.

        Use this before sending ``TASK_ASSIGN`` so later artifact results can be
        validated against the worker that originally received the assignment.

        Args:
            assignment: Worker slice assignment carrying the artifact id.

        Returns:
            ``None`` after the owner map has been updated.
        """
        with self._artifact_workers_lock:
            self._artifact_workers[assignment.artifact_id] = (
                assignment.connection.peer_id,
                assignment.connection.runtime_id,
            )

    def _pop_artifact_owner(self, artifact_id: str) -> tuple[str, str] | None:
        """Remove and return the remembered owner for one artifact id.

        Use this during task cleanup so the artifact-owner table only keeps
        active assignments that still might report results.

        Args:
            artifact_id: Artifact id allocated for one worker slice.

        Returns:
            The remembered ``(peer_id, runtime_id)`` tuple, if present.
        """
        with self._artifact_workers_lock:
            return self._artifact_workers.pop(artifact_id, None)

    def _validate_artifact_owner(self, assignment: WorkerTaskSlice) -> None:
        """Verify that an artifact result came from the expected worker.

        Use this after a worker reports ``result_artifact`` so the main node can
        reject stale or spoofed artifact ids before fetching data.

        Args:
            assignment: Worker slice assignment being validated.

        Returns:
            ``None`` when the artifact owner matches the assignment.
        """
        with self._artifact_workers_lock:
            owner = self._artifact_workers.get(assignment.artifact_id)
        expected_owner = (assignment.connection.peer_id, assignment.connection.runtime_id)
        if owner != expected_owner:
            raise ValueError(
                "artifact owner mismatch for "
                f"{assignment.artifact_id}: expected {expected_owner}, got {owner}"
            )

    def _send_artifact_release(self, assignment: WorkerTaskSlice, *, already_holds_io_lock: bool = False) -> None:
        """Tell a worker it may delete a fetched temporary artifact.

        Use this after the main node finishes fetching a large task result so
        the worker can clean up its local artifact store immediately.

        Args:
            assignment: Worker slice whose artifact should be released.
            already_holds_io_lock: Whether the caller already holds the
                connection I/O lock and should avoid reacquiring it.

        Returns:
            ``None`` after ``ARTIFACT_RELEASE`` has been sent.
        """
        sock = assignment.connection.sock
        release_lock = (
            nullcontext()
            if already_holds_io_lock
            else self._resolve_connection_lock(getattr(assignment.connection, "io_lock", None))
        )
        with release_lock:
            send_message(
                sock,
                build_artifact_release(
                    node_id=assignment.connection.runtime_id,
                    task_id=assignment.task_id,
                    artifact_id=assignment.artifact_id,
                ),
            )
        print(
            f"{RUNTIME_MSG_ARTIFACT_RELEASE} to {assignment.connection.node_name} "
            f"id={assignment.connection.runtime_id} task_id={assignment.task_id} "
            f"artifact_id={assignment.artifact_id}",
            flush=True,
        )

    def _artifact_fetch_path(self, artifact_id: str) -> Path:
        """Return one deterministic local path for a fetched worker artifact.

        Use this before downloading a worker artifact so the main node can fetch
        disk-first results directly into its artifact workspace without keeping
        the full payload in memory.

        Args:
            artifact_id: Worker-owned artifact identifier being fetched.

        Returns:
            A local path under the main-node artifact root directory.
        """
        if self.artifact_manager is None:
            raise RuntimeError("artifact manager is required to allocate local artifact paths")
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact_id).strip("._") or "artifact"
        digest = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()[:12]
        return self.artifact_manager.root_dir / f"fetch-{normalized[:80]}-{digest}.bin"

    def _select_transfer_mode(self, method: str) -> TransferMode:
        """Choose the preferred result-transfer mode for one method.

        Use this while building ``TASK_ASSIGN`` so large conv2d workloads bias
        toward artifact transfer while small FMVM tasks stay inline by default.

        Args:
            method: Logical compute method name.

        Returns:
            The transfer-mode enum to place in the task assignment.
        """
        if method == METHOD_SPATIAL_CONVOLUTION:
            return TransferMode.ARTIFACT_REQUIRED
        return TransferMode.INLINE_PREFERRED

    @trace_function
    def run_worker_task_slice(self, request, assignment: WorkerTaskSlice):
        """Execute one worker slice from assignment through result retrieval.

        Use this inside the main-node request path to send one task slice to a
        worker, wait for accept/result messages, and fetch artifacts if needed.

        Args:
            request: Original client request that owns the slice.
            assignment: One worker slice chosen by the dispatcher.

        Returns:
            The normalized ``TaskResult`` reported by that worker slice.
        """
        sock = assignment.connection.sock
        task_lock = self._resolve_connection_lock(
            getattr(assignment.connection, "task_lock", None),
            fallback=getattr(assignment.connection, "io_lock", None),
        )
        io_lock = self._resolve_connection_lock(getattr(assignment.connection, "io_lock", None))
        send_lock = nullcontext() if io_lock is task_lock else io_lock
        artifact_timeout_ms = int(max(self.config.compute_artifact_ttl_seconds, 0.0) * 1000)
        build_kwargs = dict(
            request_id=request.request_id,
            node_id=assignment.connection.runtime_id,
            task_id=assignment.task_id,
            method=request.method,
            object_id=request.object_id,
            stream_id=request.stream_id,
            iteration_count=request.iteration_count,
            transfer_mode=self._select_transfer_mode(request.method),
            artifact_id=assignment.artifact_id,
            artifact_timeout_ms=artifact_timeout_ms,
        )
        if request.method == METHOD_SPATIAL_CONVOLUTION:
            spatial_spec, variant = load_named_workload_spec(request.object_id)
            build_kwargs.update(
                task_payload=SpatialConvolutionTaskPayload(
                    start_oc=assignment.start_oc,
                    end_oc=assignment.end_oc,
                    tensor_h=spatial_spec.h,
                    tensor_w=spatial_spec.w,
                    channels_in=spatial_spec.c_in,
                    channels_out=spatial_spec.c_out,
                    kernel_size=spatial_spec.k,
                    padding=spatial_spec.pad,
                    stride=spatial_spec.stride,
                    weight_data=self.spatial_weight_slice(
                        variant=variant,
                        spec=spatial_spec,
                        start_oc=assignment.start_oc,
                        end_oc=assignment.end_oc,
                    ),
                ),
            )
        else:
            build_kwargs.update(
                task_payload=FixedMatrixVectorTaskPayload(
                    row_start=assignment.row_start,
                    row_end=assignment.row_end,
                    vector_length=request.vector_length,
                    vector_data=request.vector_data,
                )
            )
        task_assign = build_task_assign(**build_kwargs)
        self._record_artifact_owner(assignment)

        try:
            with task_lock:
                with send_lock:
                    send_message(sock, task_assign)
                task_scope = (
                    f"oc={assignment.start_oc}:{assignment.end_oc}"
                    if request.method == METHOD_SPATIAL_CONVOLUTION
                    else f"rows={assignment.row_start}:{assignment.row_end}"
                )
                print(
                    f"{RUNTIME_MSG_TASK_ASSIGN} to {assignment.connection.node_name} "
                    f"id={assignment.connection.runtime_id} "
                    f"task_id={assignment.task_id} artifact_id={assignment.artifact_id} "
                    f"method={request.method} {task_scope} "
                    f"iteration_count={request.iteration_count} "
                    f"transfer_mode={task_assign.task_assign.transfer_mode.name.lower()}",
                    flush=True,
                )

                accept_message = self._wait_for_task_message(
                    assignment.connection,
                    assignment.task_id,
                )
                if accept_message is None:
                    raise ConnectionError("worker closed the TCP session before TASK_ACCEPT")
                if accept_message.kind != MessageKind.TASK_ACCEPT or accept_message.task_accept is None:
                    raise ValueError(
                        f"expected {RUNTIME_MSG_TASK_ACCEPT}, got {describe_message_kind(accept_message.kind)}"
                    )
                task_accept = accept_message.task_accept
                if (
                    task_accept.request_id != request.request_id
                    or task_accept.task_id != assignment.task_id
                    or task_accept.node_id != assignment.connection.runtime_id
                ):
                    raise ValueError("received TASK_ACCEPT for the wrong request or task id")

                result_message = self._wait_for_task_message(
                    assignment.connection,
                    assignment.task_id,
                )
                if result_message is None:
                    raise ConnectionError("worker closed the TCP session before TASK_RESULT")
                if result_message.kind == MessageKind.TASK_FAIL and result_message.task_fail is not None:
                    task_fail = result_message.task_fail
                    if (
                        task_fail.request_id != request.request_id
                        or task_fail.task_id != assignment.task_id
                        or task_fail.node_id != assignment.connection.runtime_id
                    ):
                        raise ValueError("received TASK_FAIL for the wrong request or task id")
                    raise RuntimeError(task_fail.error_message or "worker reported TASK_FAIL")
                if result_message.kind != MessageKind.TASK_RESULT or result_message.task_result is None:
                    raise ValueError(
                        f"expected {RUNTIME_MSG_TASK_RESULT}, got {describe_message_kind(result_message.kind)}"
                    )
                task_result = result_message.task_result
                if (
                    task_result.request_id != request.request_id
                    or task_result.task_id != assignment.task_id
                    or task_result.node_id != assignment.connection.runtime_id
                ):
                    raise ValueError("received TASK_RESULT for the wrong request or task id")
                if task_result.result_artifact is not None:
                    if self.artifact_manager is None:
                        raise RuntimeError("artifact manager is required to fetch large task results")
                    self._validate_artifact_owner(assignment)
                    if task_result.result_artifact.artifact_id != assignment.artifact_id:
                        raise ValueError(
                            "worker reported unexpected artifact id "
                            f"{task_result.result_artifact.artifact_id}; expected {assignment.artifact_id}"
                        )
                    if task_result.result_artifact.producer_node_id != assignment.connection.runtime_id:
                        raise ValueError(
                            "worker reported artifact from unexpected producer "
                            f"{task_result.result_artifact.producer_node_id}; expected {assignment.connection.runtime_id}"
                        )
                    local_result_path = ""
                    output_vector = b""
                    if request.method == METHOD_SPATIAL_CONVOLUTION:
                        fetch_path = self._artifact_fetch_path(assignment.artifact_id)
                        self.artifact_manager.fetch_to_file(
                            task_result.result_artifact,
                            fetch_path,
                            timeout=max(
                                self.config.artifact_transfer_timeout,
                                self.config.runtime_socket_timeout,
                            ),
                        )
                        local_result_path = str(fetch_path)
                    else:
                        output_vector = self.artifact_manager.fetch_bytes(
                            task_result.result_artifact,
                            timeout=max(
                                self.config.artifact_transfer_timeout,
                                self.config.runtime_socket_timeout,
                            ),
                        )
                    self._send_artifact_release(
                        assignment,
                        already_holds_io_lock=(io_lock is task_lock),
                    )
                    task_result = type(task_result)(
                        request_id=task_result.request_id,
                        node_id=task_result.node_id,
                        task_id=task_result.task_id,
                        timestamp_ms=task_result.timestamp_ms,
                        status_code=task_result.status_code,
                        iteration_count=task_result.iteration_count,
                        result_artifact=task_result.result_artifact,
                        local_result_path=local_result_path,
                        row_start=task_result.row_start,
                        row_end=task_result.row_end,
                        output_length=task_result.output_length,
                        output_vector=output_vector,
                        start_oc=task_result.start_oc,
                        end_oc=task_result.end_oc,
                        output_h=task_result.output_h,
                        output_w=task_result.output_w,
                        result_artifact_id=task_result.result_artifact_id,
                    )
                return task_result
        except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
            self._remove_worker_connection(assignment.connection, str(exc))
            raise
        finally:
            self._pop_artifact_owner(assignment.artifact_id)
