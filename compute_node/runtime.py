"""Compute-node TCP runtime."""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable

from common.hardware import collect_hardware_profile
from common.types import DiscoveryResult
from compute_node.performance_summary import load_compute_performance_summary
from compute_node.heartbeat import WorkerHeartbeat
from compute_node.session import WorkerSession
from config import AppConfig
from constants import RUNTIME_MSG_HEARTBEAT, RUNTIME_MSG_HEARTBEAT_OK
from runtime_protocol import MessageKind, build_heartbeat_ok, describe_message_kind
from trace_utils import trace_function


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
    ) -> None:
        self.config = config
        self.main_node_host = main_node_host
        self.main_node_port = main_node_port
        self.logger = logger
        self.should_stop = should_stop or (lambda: False)
        self.heartbeat_state = WorkerHeartbeat()
        self._session_factory = session_factory or WorkerSession

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
        try:
            hardware = collect_hardware_profile(self.main_node_host, self.main_node_port)
            performance = load_compute_performance_summary()
            session.connect()
            register_ok = session.register(self.config.node_name, hardware, performance)

            print(
                f"Connected to main node {register_ok.main_node_name} "
                f"at {register_ok.main_node_ip}:{register_ok.main_node_port}",
                flush=True,
            )
            print(
                f"Registered compute node {self.config.node_name} "
                f"with cpu={hardware.logical_cpu_count} memory_bytes={hardware.memory_bytes}",
                flush=True,
            )
            print(
                "Reported compute backends "
                f"count={performance.hardware_count} "
                f"ranking={[f'{item.hardware_type}:{item.effective_gflops:.3f}' for item in performance.ranked_hardware]}",
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
            session.close()
