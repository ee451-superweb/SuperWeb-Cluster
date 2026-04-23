"""Run the long-lived compute-node control loop for one worker process.

Use this module after discovery has resolved a main node and the worker needs to
register itself, accept tasks, keep heartbeats flowing, and publish artifacts.
"""

from __future__ import annotations

import logging
import multiprocessing
import signal
import socket
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

from adapters.audit_log import write_audit_event
from core.config import AppConfig
from core.tracing import trace_function
from core.hardware import collect_hardware_profile
from core.types import DiscoveryResult
from compute_node.task_router import TaskExecutionRouter
from compute_node.worker_handlers import build_default_method_handlers
from compute_node.heartbeat import WorkerHeartbeat
from compute_node.performance_metrics.performance_summary import load_compute_performance_summary, load_runtime_processor_inventory
from compute_node.worker_services import (
    WorkerTaskRuntimeService,
    format_compute_performance_summary,
)
from compute_node.session import WorkerSession
from transport.artifact_manager import ArtifactManager
from wire.internal_protocol.transport import (
    MessageKind,
    describe_message_kind,
)


def _configure_subprocess_worker_signals() -> None:
    """Prevent task worker subprocesses from handling terminal Ctrl+C directly.

    The supervisor owns SIGINT/SIGTERM in the parent process. When the terminal
    sends Ctrl+C to the whole process group, spawned task workers should stay
    quiet and let the parent coordinate shutdown instead of emitting their own
    ``KeyboardInterrupt`` tracebacks while blocked on the work queue.
    """

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        return


def _shutdown_task_process_pool(task_process_pool, *, has_in_flight_tasks: bool) -> None:
    """Shut down the task process pool without leaking idle worker state.

    Args:
        task_process_pool: Process-pool executor created for task execution.
        has_in_flight_tasks: Whether runtime shutdown still has unfinished tasks.

    Returns:
        ``None`` after the executor has been asked to stop and, when required,
        its worker processes have been terminated explicitly.
    """

    process_mapping = getattr(task_process_pool, "_processes", None)
    processes = tuple(process_mapping.values()) if isinstance(process_mapping, dict) else ()
    task_process_pool.shutdown(wait=not has_in_flight_tasks, cancel_futures=True)
    if not has_in_flight_tasks:
        return

    for process in processes:
        if getattr(process, "is_alive", lambda: False)():
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
    for process in processes:
        join = getattr(process, "join", None)
        if callable(join):
            join(timeout=1.0)


def _execute_task_in_subprocess(task, pinned_backend: str | None = None):
    """Use this helper when one task should run in its own worker process.

    Args: task decoded TaskAssign payload routed to the task executor.
        pinned_backend Optional backend the task runtime must restrict itself
        to so a dual-purpose peer only executes its assigned backend.
    Returns: The task-execution result produced by the method handler.
    """

    task_executor = TaskExecutionRouter(
        build_default_method_handlers(pinned_backend=pinned_backend)
    )
    try:
        return task_executor.execute_task(task)
    finally:
        task_executor.close()


class ComputeNodeRuntime:
    """Own the compute-node runtime session from register through shutdown."""

    @trace_function
    def __init__(
        self,
        config: AppConfig,
        main_node_host: str,
        main_node_port: int,
        logger: logging.Logger,
        should_stop: Callable[[], bool] | None = None,
        session_factory: Callable[..., WorkerSession] | None = None,
        task_executor_factory: Callable[..., TaskExecutionRouter] | None = None,
    ) -> None:
        """Create the compute-node runtime and its shared session/task state.

        Args: config runtime settings, main_node_host/port target endpoint, logger diagnostics sink, should_stop optional shutdown predicate, session_factory/task_executor_factory injectable test seams.
        Returns: None after the runtime stores its collaborators and counters.
        """
        self.config = config
        self.main_node_host = main_node_host
        self.main_node_port = main_node_port
        self.logger = logger
        self.should_stop = should_stop or (lambda: False)
        self.heartbeat_state = WorkerHeartbeat()
        self._session_factory = session_factory or WorkerSession
        self._task_executor_factory = task_executor_factory
        self.task_runtime_service = WorkerTaskRuntimeService(
            config=self.config,
            logger=self.logger,
            node_name=self.config.node_name,
        )

    @trace_function
    def _build_session(self) -> WorkerSession:
        """Use this when the runtime needs a fresh WorkerSession connection wrapper.

        Args: self runtime instance providing connection settings.
        Returns: A configured WorkerSession ready to connect to the main node.
        """
        return self._session_factory(
            self.main_node_host,
            self.main_node_port,
            connect_timeout=self.config.tcp_connect_timeout,
            socket_timeout=self.config.runtime_socket_timeout,
            max_message_size=self.config.max_message_size,
        )

    def _load_registration_profile(self):
        """Collect the local hardware and abstract performance summary for registration.

        Args:
            None.

        Returns:
            A ``(hardware, performance)`` tuple ready for ``REGISTER_WORKER``.
        """
        hardware = collect_hardware_profile(self.main_node_host, self.main_node_port)
        pinned_backend = self.config.pinned_backend
        legacy_inventory = load_runtime_processor_inventory(pinned_backend=pinned_backend)
        performance = legacy_inventory.to_legacy_summary()
        multi_method_summary = load_compute_performance_summary(pinned_backend=pinned_backend)
        if multi_method_summary.method_summaries:
            performance = multi_method_summary
        return hardware, performance

    def _build_task_execution_backend(self):
        """Create the task executor objects used by this worker runtime.

        Args:
            None.

        Returns:
            A ``(task_process_pool, task_thread_pool, task_executor)`` tuple.
        """
        task_process_pool = None
        task_thread_pool = None
        task_executor = None
        if self._task_executor_factory is None:
            spawn_context = multiprocessing.get_context("spawn")
            try:
                # Recycle the Python task worker after every task so large
                # task allocations return to the OS before we report the
                # result artifact back to the main node.
                task_process_pool = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=spawn_context,
                    max_tasks_per_child=1,
                    initializer=_configure_subprocess_worker_signals,
                )
            except TypeError:
                task_process_pool = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=spawn_context,
                    initializer=_configure_subprocess_worker_signals,
                )
        else:
            try:
                task_executor = self._task_executor_factory()
            except TypeError:
                # Backward compatibility for older tests/factories that still
                # expect one inventory-like constructor argument.
                task_executor = self._task_executor_factory(None)
            task_thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compute-task")
        return task_process_pool, task_thread_pool, task_executor

    def _build_artifact_manager(self, session: WorkerSession):
        """Create the local artifact manager used for large task results.

        Args:
            session: Connected worker session whose local socket address determines the advertised host.

        Returns:
            An ``ArtifactManager`` rooted under the compute-node artifact directory.
        """
        artifact_host = "127.0.0.1"
        if getattr(session, "sock", None) is not None:
            artifact_host = session.sock.getsockname()[0]
        return ArtifactManager(
            root_dir=Path(__file__).resolve().parent / "artifacts",
            public_host=artifact_host,
            chunk_size=self.config.artifact_chunk_size,
        )

    def _drain_completed_tasks(
        self,
        *,
        session: WorkerSession,
        assigned_node_id: str,
        pending_tasks: dict[str, tuple[object, object]],
        artifact_manager: ArtifactManager | None,
    ) -> None:
        """Use this inside the event loop to flush finished task futures back to the main node.

        Args: session current worker session, assigned_node_id worker runtime id, pending_tasks active future map, artifact_manager optional large-result publisher.
        Returns: None after completed tasks are turned into TASK_RESULT or TASK_FAIL messages.
        """
        self.task_runtime_service.drain_completed_tasks(
            session=session,
            assigned_node_id=assigned_node_id,
            pending_tasks=pending_tasks,
            artifact_manager=artifact_manager,
        )

    @trace_function
    def run(self) -> DiscoveryResult:
        """Run the compute-node event loop until shutdown or connection failure.

        Args: self runtime instance containing connection settings and task state.
        Returns: A DiscoveryResult describing whether the compute node stopped cleanly or failed to join/stay connected.
        """
        session = self._build_session()
        task_executor = None
        task_thread_pool = None
        task_process_pool = None
        artifact_manager = None
        pending_tasks: dict[str, tuple[object, object]] = {}
        try:
            hardware, performance = self._load_registration_profile()
            task_process_pool, task_thread_pool, task_executor = self._build_task_execution_backend()

            session.connect()
            register_ok = session.register(self.config.node_name, hardware, performance)
            assigned_node_id = register_ok.node_id or self.config.node_name
            artifact_manager = self._build_artifact_manager(session)

            write_audit_event(
                f"started as compute node assigned_node_id={assigned_node_id} "
                f"main_node={register_ok.main_node_ip}:{register_ok.main_node_port}",
                stdout=True,
                logger=self.logger,
            )
            write_audit_event(
                f"Connected to main node {register_ok.main_node_name} "
                f"at {register_ok.main_node_ip}:{register_ok.main_node_port} "
                f"assigned_node_id={assigned_node_id}",
                stdout=True,
                logger=self.logger,
            )
            write_audit_event(
                f"Registered compute node {self.config.node_name} "
                f"with cpu={hardware.logical_cpu_count} memory_bytes={hardware.memory_bytes}",
                stdout=True,
                logger=self.logger,
            )
            write_audit_event(
                "Reported compute methods "
                f"count={len(performance.method_summaries) or 1} "
                f"methods={[summary.method for summary in performance.method_summaries] or ['gemv']}",
                stdout=True,
                logger=self.logger,
            )
            write_audit_event(
                "Reported compute performance "
                f"{format_compute_performance_summary(performance)}",
                stdout=True,
                logger=self.logger,
            )

            while not self.should_stop():
                self._drain_completed_tasks(
                    session=session,
                    assigned_node_id=assigned_node_id,
                    pending_tasks=pending_tasks,
                    artifact_manager=artifact_manager,
                )
                try:
                    message = session.receive()
                except socket.timeout:
                    continue

                if message is None:
                    return DiscoveryResult(
                        success=False,
                        peer_address=self.main_node_host,
                        peer_port=self.main_node_port,
                        source="compute_node",
                        message="Main node closed the TCP session.",
                    )

                if self.task_runtime_service.handle_runtime_message(
                    message=message,
                    session=session,
                    assigned_node_id=assigned_node_id,
                    heartbeat_state=self.heartbeat_state,
                    pending_tasks=pending_tasks,
                    process_pool=task_process_pool,
                    thread_pool=task_thread_pool,
                    task_executor=task_executor,
                    subprocess_entrypoint=partial(
                        _execute_task_in_subprocess,
                        pinned_backend=self.config.pinned_backend,
                    ),
                    artifact_manager=artifact_manager,
                ):
                    continue

                self.logger.warning("Ignoring unexpected runtime message kind=%s", describe_message_kind(message.kind))

            return DiscoveryResult(
                success=True,
                peer_address=self.main_node_host,
                peer_port=self.main_node_port,
                source="compute_node",
                message="Compute-node runtime stopped.",
            )
        except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
            return DiscoveryResult(
                success=False,
                peer_address=self.main_node_host,
                peer_port=self.main_node_port,
                source="compute_node",
                message=f"Unable to join main-node TCP runtime: {exc}.",
            )
        finally:
            if task_process_pool is not None:
                _shutdown_task_process_pool(
                    task_process_pool,
                    has_in_flight_tasks=bool(pending_tasks),
                )
            if task_thread_pool is not None:
                task_thread_pool.shutdown(wait=True, cancel_futures=True)
            if task_executor is not None and hasattr(task_executor, "close"):
                task_executor.close()
            if artifact_manager is not None:
                artifact_manager.close()
            session.close()
