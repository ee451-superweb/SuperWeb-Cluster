"""Own compute-node task state, message handling, and idle refresh helpers.

Use this module when the compute-node runtime should mostly orchestrate startup
and shutdown while dedicated helpers manage in-flight tasks and idle worker
performance updates.
"""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from app.constants import (
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
    METHOD_SPATIAL_CONVOLUTION,
    RUNTIME_MSG_ARTIFACT_RELEASE,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_HEARTBEAT_OK,
    RUNTIME_MSG_TASK_ACCEPT,
    RUNTIME_MSG_TASK_ASSIGN,
    RUNTIME_MSG_TASK_FAIL,
    RUNTIME_MSG_TASK_RESULT,
    RUNTIME_MSG_WORKER_UPDATE,
    STATUS_ACCEPTED,
    STATUS_INTERNAL_ERROR,
)
from common.types import ComputePerformanceSummary
from wire.internal_protocol.runtime_transport import (
    FixedMatrixVectorResultPayload,
    MessageKind,
    NodeStatus,
    SpatialConvolutionResultPayload,
    TransferMode,
    build_heartbeat_ok,
    build_task_accept,
    build_task_fail,
    build_task_result,
    build_worker_update,
)


def _method_display_name(method: str) -> str:
    """Return a short human-readable label for one method name."""

    if method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
        return "fmvm"
    if method == METHOD_SPATIAL_CONVOLUTION:
        return "conv2d"
    return method


def format_compute_performance_summary(performance: ComputePerformanceSummary | None) -> str:
    """Format one abstract performance summary for human-readable logs.

    Args:
        performance: Optional performance summary to render.

    Returns:
        A short string such as ``fmvm=cuda:125.000GFLOPS conv2d=cpu:24.000GFLOPS``.
    """

    if performance is None:
        return "<none>"
    if performance.method_summaries:
        parts: list[str] = []
        for method_summary in performance.method_summaries:
            ranked = method_summary.ranked_hardware
            if not ranked:
                parts.append(f"{_method_display_name(method_summary.method)}=<unavailable>")
                continue
            ranked_text = ",".join(
                f"{item.hardware_type}:{item.effective_gflops:.3f}GFLOPS"
                for item in ranked
            )
            parts.append(f"{_method_display_name(method_summary.method)}={ranked_text}")
        return " ".join(parts) if parts else "<none>"
    if performance.ranked_hardware:
        ranked_text = ",".join(
            f"{item.hardware_type}:{item.effective_gflops:.3f}GFLOPS"
            for item in performance.ranked_hardware
        )
        return f"{_method_display_name(METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION)}={ranked_text}"
    return "<none>"


class WorkerTaskRuntimeService:
    """Track active tasks and translate worker-side task events into runtime messages."""

    def __init__(self, *, config, logger, node_name: str) -> None:
        """Store runtime settings and initialize active-task bookkeeping.

        Args:
            config: Compute-node runtime configuration with wire-size and TTL policy.
            logger: Logger used for warning paths.
            node_name: Human-readable name of this compute node.
        """
        self.config = config
        self.logger = logger
        self.node_name = node_name
        self._active_tasks: dict[str, str] = {}
        self._active_tasks_lock = threading.Lock()
        self._completed_task_count = 0

    def has_active_or_pending_tasks(self, pending_tasks: dict[str, tuple[object, object]]) -> bool:
        """Report whether the worker still has running or queued work.

        Args:
            pending_tasks: Map from task id to ``(task, future)`` pairs.

        Returns:
            ``True`` when either active-task bookkeeping or pending futures is non-empty.
        """
        with self._active_tasks_lock:
            return bool(self._active_tasks) or bool(pending_tasks)

    def mark_active_task(self, task) -> None:
        """Record one accepted task so heartbeat replies can report it as active.

        Args:
            task: Parsed ``TASK_ASSIGN`` payload that was just accepted.

        Returns:
            ``None`` after the task id is visible in status snapshots.
        """
        with self._active_tasks_lock:
            self._active_tasks[task.task_id] = task.method

    def clear_active_task(self, task_id: str) -> None:
        """Forget one task after its result or failure has been reported.

        Args:
            task_id: Task identifier to remove from active bookkeeping.

        Returns:
            ``None`` after the task is no longer considered active.
        """
        with self._active_tasks_lock:
            self._active_tasks.pop(task_id, None)

    def describe_active_tasks(self) -> str:
        """Summarize the worker's current task pool for operator logs.

        Args:
            None.

        Returns:
            A short text string describing the active tasks or idle state.
        """
        with self._active_tasks_lock:
            if not self._active_tasks:
                return "current_task=<idle>"
            return "current_task=" + ",".join(
                f"{task_id}:{method}" for task_id, method in sorted(self._active_tasks.items())
            )

    def snapshot_active_task_ids(self) -> tuple[str, ...]:
        """Return a stable snapshot of current active task ids for heartbeat replies.

        Args:
            None.

        Returns:
            A sorted tuple of currently active task ids.
        """
        with self._active_tasks_lock:
            return tuple(sorted(self._active_tasks))

    def get_node_status(self) -> NodeStatus:
        """Return the worker's coarse busy/idle state for heartbeat replies.

        Args:
            None.

        Returns:
            ``NodeStatus.BUSY`` while tasks are active, otherwise ``NodeStatus.IDLE``.
        """
        with self._active_tasks_lock:
            return NodeStatus.BUSY if self._active_tasks else NodeStatus.IDLE

    def artifact_ttl_seconds_for_task(self, task) -> float | None:
        """Resolve how long one task-produced artifact should stay on disk locally.

        Args:
            task: Completed task whose protocol TTL hint may be set.

        Returns:
            TTL seconds, or ``None`` when the artifact should not expire automatically.
        """
        if getattr(task, "artifact_timeout_ms", 0) > 0:
            return max(task.artifact_timeout_ms / 1000.0, 0.0)
        if self.config.compute_artifact_ttl_seconds > 0:
            return self.config.compute_artifact_ttl_seconds
        return None

    def submit_task(
        self,
        *,
        task,
        process_pool: ProcessPoolExecutor | None,
        thread_pool: ThreadPoolExecutor | None,
        task_executor,
        subprocess_entrypoint,
    ):
        """Queue one accepted task on the configured execution backend.

        Args:
            task: Accepted task payload to execute.
            process_pool: Optional process pool used for isolated execution.
            thread_pool: Optional thread pool used with an injected executor.
            task_executor: In-process task executor used with ``thread_pool``.
            subprocess_entrypoint: Callable used when dispatching to ``process_pool``.

        Returns:
            A future representing the queued task execution.
        """
        if process_pool is not None:
            return process_pool.submit(subprocess_entrypoint, task)
        assert thread_pool is not None
        assert task_executor is not None
        return thread_pool.submit(task_executor.execute_task, task)

    def drain_completed_tasks(
        self,
        *,
        session,
        assigned_node_id: str,
        pending_tasks: dict[str, tuple[object, object]],
        artifact_manager,
    ) -> None:
        """Turn completed task futures into ``TASK_RESULT`` or ``TASK_FAIL`` messages.

        Args:
            session: Active worker session back to the main node.
            assigned_node_id: Main-node-assigned worker id.
            pending_tasks: Map from task id to ``(task, future)`` pairs.
            artifact_manager: Optional artifact manager for large outputs.

        Returns:
            ``None`` after every finished future has been flushed upstream.
        """
        completed_task_ids = [
            task_id
            for task_id, (_task, future) in pending_tasks.items()
            if future.done()
        ]
        for task_id in completed_task_ids:
            task, future = pending_tasks.pop(task_id)
            self.clear_active_task(task.task_id)
            self._completed_task_count += 1
            try:
                task_result = future.result()
            except (OSError, RuntimeError, ValueError) as exc:
                session.send(
                    build_task_fail(
                        request_id=task.request_id,
                        node_id=assigned_node_id,
                        task_id=task.task_id,
                        status_code=STATUS_INTERNAL_ERROR,
                        error_message=str(exc),
                    )
                )
                print(
                    f"{RUNTIME_MSG_TASK_FAIL} from {self.node_name} "
                    f"id={assigned_node_id} task_id={task.task_id} error={exc}",
                    flush=True,
                )
                continue

            result_artifact = task_result.result_artifact
            output_vector = task_result.output_vector
            local_result_path = task_result.local_result_path
            if artifact_manager is not None and result_artifact is None:
                if local_result_path:
                    result_artifact = artifact_manager.register_existing_file(
                        Path(local_result_path),
                        producer_node_id=assigned_node_id,
                        artifact_id=task.artifact_id or task.task_id,
                        ttl_seconds=self.artifact_ttl_seconds_for_task(task),
                        delete_local_path=True,
                    )
                    output_vector = b""
                elif (
                    len(output_vector) > self.config.max_message_size
                    or task.transfer_mode == TransferMode.ARTIFACT_REQUIRED
                ):
                    result_artifact = artifact_manager.publish_bytes(
                        output_vector,
                        producer_node_id=assigned_node_id,
                        artifact_id=task.artifact_id or task.task_id,
                        ttl_seconds=self.artifact_ttl_seconds_for_task(task),
                    )
                    output_vector = b""

            session.send(
                build_task_result(
                    request_id=task_result.request_id,
                    node_id=assigned_node_id,
                    task_id=task_result.task_id,
                    status_code=task_result.status_code,
                    iteration_count=task_result.iteration_count,
                    result_payload=(
                        SpatialConvolutionResultPayload(
                            start_oc=task_result.start_oc,
                            end_oc=task_result.end_oc,
                            output_h=task_result.output_h,
                            output_w=task_result.output_w,
                            output_length=task_result.output_length,
                            output_vector=output_vector,
                            result_artifact_id=task_result.result_artifact_id or (
                                result_artifact.artifact_id if result_artifact is not None else ""
                            ),
                        )
                        if task.method == METHOD_SPATIAL_CONVOLUTION
                        else FixedMatrixVectorResultPayload(
                            row_start=task_result.row_start,
                            row_end=task_result.row_end,
                            output_length=task_result.output_length,
                            output_vector=output_vector,
                        )
                    ),
                    result_artifact=result_artifact,
                )
            )
            result_scope = (
                f"oc={task_result.start_oc}:{task_result.end_oc}"
                if task.method == METHOD_SPATIAL_CONVOLUTION
                else f"rows={task_result.row_start}:{task_result.row_end}"
            )
            print(
                f"{RUNTIME_MSG_TASK_RESULT} from {self.node_name} "
                f"id={assigned_node_id} task_id={task.task_id} "
                f"method={task.method} {result_scope} "
                f"iteration_count={task_result.iteration_count} "
                f"artifact_id={(result_artifact.artifact_id if result_artifact is not None else '')}",
                flush=True,
            )

    def handle_heartbeat_message(self, *, session, assigned_node_id: str, heartbeat_state, heartbeat) -> None:
        """Reply to one incoming heartbeat using the worker's live task snapshot.

        Args:
            session: Active worker session back to the main node.
            assigned_node_id: Main-node-assigned worker id.
            heartbeat_state: Worker heartbeat state object updated by incoming heartbeats.
            heartbeat: Parsed heartbeat payload.

        Returns:
            ``None`` after one ``HEARTBEAT_OK`` message has been sent.
        """
        heartbeat_state.respond(heartbeat)
        session.send(
            build_heartbeat_ok(
                node_name=self.node_name,
                heartbeat_unix_time_ms=heartbeat.unix_time_ms,
                node_id=assigned_node_id,
                active_task_ids=self.snapshot_active_task_ids(),
                node_status=self.get_node_status(),
                completed_task_count=self._completed_task_count,
            )
        )
        print(
            f"{RUNTIME_MSG_HEARTBEAT} from {heartbeat.main_node_name} "
            f"at {heartbeat.unix_time_ms}",
            flush=True,
        )
        print(
            f"{RUNTIME_MSG_HEARTBEAT_OK} from {self.node_name} "
            f"for {heartbeat.unix_time_ms} {self.describe_active_tasks()}",
            flush=True,
        )

    def handle_artifact_release_message(self, *, assigned_node_id: str, artifact_release, artifact_manager) -> None:
        """Delete one locally published artifact after the main node releases it.

        Args:
            assigned_node_id: Main-node-assigned worker id.
            artifact_release: Parsed ``ARTIFACT_RELEASE`` payload.
            artifact_manager: Local artifact manager, if enabled.

        Returns:
            ``None`` after the release has been logged and applied when valid.
        """
        removed = False
        if artifact_release.node_id and artifact_release.node_id != assigned_node_id:
            self.logger.warning(
                "Ignoring artifact release for wrong node_id=%s expected=%s artifact_id=%s",
                artifact_release.node_id,
                assigned_node_id,
                artifact_release.artifact_id,
            )
        elif artifact_manager is not None:
            removed = artifact_manager.remove_artifact(artifact_release.artifact_id)
        print(
            f"{RUNTIME_MSG_ARTIFACT_RELEASE} for {assigned_node_id} "
            f"task_id={artifact_release.task_id} "
            f"artifact_id={artifact_release.artifact_id} "
            f"removed={removed}",
            flush=True,
        )

    def handle_task_assign_message(
        self,
        *,
        session,
        assigned_node_id: str,
        task,
        pending_tasks: dict[str, tuple[object, object]],
        process_pool: ProcessPoolExecutor | None,
        thread_pool: ThreadPoolExecutor | None,
        task_executor,
        subprocess_entrypoint,
        artifact_manager,
    ) -> None:
        """Accept one task, submit it for execution, and opportunistically flush completions.

        Args:
            session: Active worker session back to the main node.
            assigned_node_id: Main-node-assigned worker id.
            task: Parsed ``TASK_ASSIGN`` payload.
            pending_tasks: Map from task id to ``(task, future)`` pairs.
            process_pool: Optional isolated process pool.
            thread_pool: Optional thread pool used with an injected executor.
            task_executor: In-process task executor used with ``thread_pool``.
            subprocess_entrypoint: Callable used when dispatching to ``process_pool``.
            artifact_manager: Local artifact manager used when large outputs are published.

        Returns:
            ``None`` after the task has been queued and any immediately finished work flushed.
        """
        session.send(
            build_task_accept(
                request_id=task.request_id,
                node_id=assigned_node_id,
                task_id=task.task_id,
                status_code=STATUS_ACCEPTED,
            )
        )
        print(
            f"{RUNTIME_MSG_TASK_ACCEPT} from {self.node_name} "
            f"id={assigned_node_id} task_id={task.task_id} "
            f"artifact_id={task.artifact_id} "
            f"method={task.method} "
            f"rows={task.row_start}:{task.row_end} "
            f"oc={task.start_oc}:{task.end_oc} "
            f"iteration_count={task.iteration_count} "
            f"transfer_mode={task.transfer_mode.name.lower()}",
            flush=True,
        )
        pending_tasks[task.task_id] = (
            task,
            self.submit_task(
                task=task,
                process_pool=process_pool,
                thread_pool=thread_pool,
                task_executor=task_executor,
                subprocess_entrypoint=subprocess_entrypoint,
            ),
        )
        self.mark_active_task(task)
        print(
            f"{self.node_name} running {self.describe_active_tasks()}",
            flush=True,
        )
        self.drain_completed_tasks(
            session=session,
            assigned_node_id=assigned_node_id,
            pending_tasks=pending_tasks,
            artifact_manager=artifact_manager,
        )

    def handle_runtime_message(
        self,
        *,
        message,
        session,
        assigned_node_id: str,
        heartbeat_state,
        pending_tasks: dict[str, tuple[object, object]],
        process_pool: ProcessPoolExecutor | None,
        thread_pool: ThreadPoolExecutor | None,
        task_executor,
        subprocess_entrypoint,
        artifact_manager,
    ) -> bool:
        """Route one main-node runtime message to the appropriate worker-side handler.

        Args:
            message: Decoded runtime envelope received from the main node.
            session: Active worker session back to the main node.
            assigned_node_id: Main-node-assigned worker id.
            heartbeat_state: Worker heartbeat tracker updated by heartbeat traffic.
            pending_tasks: Map from task id to ``(task, future)`` pairs.
            process_pool: Optional isolated process pool.
            thread_pool: Optional thread pool used with an injected executor.
            task_executor: In-process task executor used with ``thread_pool``.
            subprocess_entrypoint: Callable used when dispatching to ``process_pool``.
            artifact_manager: Local artifact manager used when large outputs are published.

        Returns:
            ``True`` when the message was handled by this service, otherwise ``False``.
        """
        if message.kind == MessageKind.HEARTBEAT and message.heartbeat is not None:
            self.handle_heartbeat_message(
                session=session,
                assigned_node_id=assigned_node_id,
                heartbeat_state=heartbeat_state,
                heartbeat=message.heartbeat,
            )
            return True

        if message.kind == MessageKind.ARTIFACT_RELEASE and message.artifact_release is not None:
            self.handle_artifact_release_message(
                assigned_node_id=assigned_node_id,
                artifact_release=message.artifact_release,
                artifact_manager=artifact_manager,
            )
            return True

        if message.kind == MessageKind.TASK_ASSIGN and message.task_assign is not None:
            self.handle_task_assign_message(
                session=session,
                assigned_node_id=assigned_node_id,
                task=message.task_assign,
                pending_tasks=pending_tasks,
                process_pool=process_pool,
                thread_pool=thread_pool,
                task_executor=task_executor,
                subprocess_entrypoint=subprocess_entrypoint,
                artifact_manager=artifact_manager,
            )
            return True

        return False


class IdlePerformanceRefreshService:
    """Refresh and report abstract performance after a worker stays idle."""

    def __init__(self, *, config, logger, node_name: str) -> None:
        """Store runtime settings used by idle refresh polling.

        Args:
            config: Compute-node runtime configuration with idle refresh intervals.
            logger: Logger used for refresh-failure warnings.
            node_name: Human-readable name of this compute node.
        """
        self.config = config
        self.logger = logger
        self.node_name = node_name

    def advance(
        self,
        *,
        session,
        assigned_node_id: str,
        refresh_thread_pool: ThreadPoolExecutor | None,
        refresh_future,
        last_worker_update_at: float,
        has_active_or_pending_tasks: bool,
        refresh_callable,
    ):
        """Finish completed refresh work and launch a new idle refresh when due.

        Args:
            session: Active worker session back to the main node.
            assigned_node_id: Main-node-assigned worker id.
            refresh_thread_pool: Optional executor for idle refresh work.
            refresh_future: Currently running refresh future, if any.
            last_worker_update_at: Monotonic timestamp of the last worker update.
            has_active_or_pending_tasks: Whether the worker is currently busy.
            refresh_callable: Callable that produces a refreshed performance summary.

        Returns:
            A ``(refresh_future, last_worker_update_at)`` tuple with updated state.
        """
        if refresh_future is not None and refresh_future.done():
            completed_refresh = refresh_future
            refresh_future = None
            try:
                refreshed_performance = completed_refresh.result()
            except Exception as exc:
                self.logger.warning("Idle worker refresh failed: %s", exc)
                last_worker_update_at = time.monotonic()
            else:
                session.send(
                    build_worker_update(
                        node_id=assigned_node_id,
                        performance=refreshed_performance,
                    )
                )
                last_worker_update_at = time.monotonic()
                print(
                    f"{RUNTIME_MSG_WORKER_UPDATE} from {self.node_name} "
                    f"id={assigned_node_id} "
                    f"performance={format_compute_performance_summary(refreshed_performance)}",
                    flush=True,
                )

        if (
            refresh_thread_pool is not None
            and refresh_future is None
            and not has_active_or_pending_tasks
            and (time.monotonic() - last_worker_update_at) >= self.config.idle_worker_update_interval
        ):
            refresh_future = refresh_thread_pool.submit(refresh_callable)

        return refresh_future, last_worker_update_at
