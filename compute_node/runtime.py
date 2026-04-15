"""Compute-node TCP runtime."""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable

from common.hardware import collect_hardware_profile
from common.types import DiscoveryResult
from compute_node.executor import TaskExecutionRouter
from compute_node.handlers import build_default_method_handlers
from compute_node.performance_summary import load_compute_performance_summary, load_runtime_processor_inventory
from compute_node.heartbeat import WorkerHeartbeat
from compute_node.session import WorkerSession
from app.config import AppConfig
from app.constants import (
    METHOD_SPATIAL_CONVOLUTION,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_HEARTBEAT_OK,
    RUNTIME_MSG_TASK_ACCEPT,
    RUNTIME_MSG_TASK_ASSIGN,
    RUNTIME_MSG_TASK_FAIL,
    RUNTIME_MSG_TASK_RESULT,
    STATUS_ACCEPTED,
    STATUS_INTERNAL_ERROR,
)
from wire.runtime import (
    MessageKind,
    build_heartbeat_ok,
    build_task_accept,
    build_task_fail,
    build_task_result,
    describe_message_kind,
)
from app.trace_utils import trace_function


class ComputeNodeRuntime:
    """Connect to the main node and stay attached to the runtime session."""

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
        self.config = config
        self.main_node_host = main_node_host
        self.main_node_port = main_node_port
        self.logger = logger
        self.should_stop = should_stop or (lambda: False)
        self.heartbeat_state = WorkerHeartbeat()
        self._session_factory = session_factory or WorkerSession
        self._task_executor_factory = task_executor_factory or (lambda: TaskExecutionRouter(build_default_method_handlers()))

    @trace_function
    def _build_session(self) -> WorkerSession:
        return self._session_factory(
            self.main_node_host,
            self.main_node_port,
            connect_timeout=self.config.tcp_connect_timeout,
            socket_timeout=self.config.runtime_socket_timeout,
            max_message_size=self.config.max_message_size,
        )

    @trace_function
    def run(self) -> DiscoveryResult:
        session = self._build_session()
        task_executor = None
        try:
            hardware = collect_hardware_profile(self.main_node_host, self.main_node_port)
            legacy_inventory = load_runtime_processor_inventory()
            performance = legacy_inventory.to_legacy_summary()
            multi_method_summary = load_compute_performance_summary()
            if multi_method_summary.method_summaries:
                performance = multi_method_summary
            try:
                task_executor = self._task_executor_factory()
            except TypeError:
                # Backward compatibility for older tests/factories that still
                # expect one inventory-like constructor argument.
                task_executor = self._task_executor_factory(None)
            session.connect()
            register_ok = session.register(self.config.node_name, hardware, performance)
            assigned_node_id = register_ok.node_id or self.config.node_name

            print(
                f"Connected to main node {register_ok.main_node_name} "
                f"at {register_ok.main_node_ip}:{register_ok.main_node_port} "
                f"assigned_node_id={assigned_node_id}",
                flush=True,
            )
            print(
                f"Registered compute node {self.config.node_name} "
                f"with cpu={hardware.logical_cpu_count} memory_bytes={hardware.memory_bytes}",
                flush=True,
            )
            print(
                "Reported compute methods "
                f"count={len(performance.method_summaries) or 1} "
                f"methods={[summary.method for summary in performance.method_summaries] or ['fixed_matrix_vector_multiplication']}",
                flush=True,
            )

            while not self.should_stop():
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

                if message.kind == MessageKind.HEARTBEAT and message.heartbeat is not None:
                    self.heartbeat_state.respond(message.heartbeat)
                    session.send(
                        build_heartbeat_ok(
                            node_name=self.config.node_name,
                            heartbeat_unix_time_ms=message.heartbeat.unix_time_ms,
                        )
                    )
                    print(
                        f"{RUNTIME_MSG_HEARTBEAT} from {message.heartbeat.main_node_name} "
                        f"at {message.heartbeat.unix_time_ms}",
                        flush=True,
                    )
                    print(
                        f"{RUNTIME_MSG_HEARTBEAT_OK} from {self.config.node_name} "
                        f"for {message.heartbeat.unix_time_ms}",
                        flush=True,
                    )
                    continue

                if message.kind == MessageKind.TASK_ASSIGN and message.task_assign is not None:
                    task = message.task_assign
                    session.send(
                        build_task_accept(
                            request_id=task.request_id,
                            node_id=assigned_node_id,
                            task_id=task.task_id,
                            status_code=STATUS_ACCEPTED,
                        )
                    )
                    print(
                        f"{RUNTIME_MSG_TASK_ACCEPT} from {self.config.node_name} "
                        f"id={assigned_node_id} task_id={task.task_id} "
                        f"method={task.method} "
                        f"rows={task.row_start}:{task.row_end} "
                        f"oc={task.start_oc}:{task.end_oc} "
                        f"iteration_count={task.iteration_count}",
                        flush=True,
                    )
                    try:
                        task_result = task_executor.execute_task(task)
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
                            f"{RUNTIME_MSG_TASK_FAIL} from {self.config.node_name} "
                            f"id={assigned_node_id} task_id={task.task_id} error={exc}",
                            flush=True,
                        )
                        continue

                    session.send(
                        build_task_result(
                            request_id=task_result.request_id,
                            node_id=assigned_node_id,
                            task_id=task_result.task_id,
                            status_code=task_result.status_code,
                            row_start=task_result.row_start,
                            row_end=task_result.row_end,
                            output_vector=task_result.output_vector,
                            iteration_count=task_result.iteration_count,
                            start_oc=task_result.start_oc,
                            end_oc=task_result.end_oc,
                            output_h=task_result.output_h,
                            output_w=task_result.output_w,
                            result_artifact_id=task_result.result_artifact_id,
                        )
                    )
                    result_scope = (
                        f"oc={task_result.start_oc}:{task_result.end_oc}"
                        if task.method == METHOD_SPATIAL_CONVOLUTION
                        else f"rows={task_result.row_start}:{task_result.row_end}"
                    )
                    print(
                        f"{RUNTIME_MSG_TASK_RESULT} from {self.config.node_name} "
                        f"id={assigned_node_id} task_id={task.task_id} "
                        f"method={task.method} {result_scope} "
                        f"iteration_count={task_result.iteration_count}",
                        flush=True,
                    )
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
            if task_executor is not None and hasattr(task_executor, "close"):
                task_executor.close()
            session.close()


