"""TCP session helpers for compute-node runtime traffic."""

from __future__ import annotations

import socket

from common.types import ComputePerformanceSummary, HardwareProfile
from app.constants import RUNTIME_MSG_REGISTER_OK
from wire.runtime import MessageKind, RegisterOk, RuntimeEnvelope, build_register_worker, recv_message, send_message
from app.trace_utils import trace_function


class WorkerSession:
    """Persistent TCP session to the main node."""

    @trace_function
    def __init__(
        self,
        main_node_host: str,
        main_node_port: int,
        *,
        connect_timeout: float,
        socket_timeout: float,
        max_message_size: int,
    ) -> None:
        self.main_node_host = main_node_host
        self.main_node_port = main_node_port
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout
        self.max_message_size = max_message_size
        self.sock: socket.socket | None = None

    @trace_function
    def connect(self) -> None:
        self.sock = socket.create_connection((self.main_node_host, self.main_node_port), timeout=self.connect_timeout)
        self.sock.settimeout(self.socket_timeout)

    @trace_function
    def register(
        self,
        node_name: str,
        hardware: HardwareProfile,
        performance: ComputePerformanceSummary,
    ) -> RegisterOk:
        if self.sock is None:
            raise RuntimeError("worker session is not connected")

        send_message(self.sock, build_register_worker(node_name, hardware, performance))
        response = recv_message(self.sock, max_size=self.max_message_size)
        if response is None:
            raise ConnectionError("main node closed the TCP session during registration")
        if response.kind != MessageKind.REGISTER_OK or response.register_ok is None:
            raise ValueError(f"expected {RUNTIME_MSG_REGISTER_OK}, got {response.kind.name}")
        return response.register_ok

    @trace_function
    def receive(self):
        if self.sock is None:
            raise RuntimeError("worker session is not connected")
        return recv_message(self.sock, max_size=self.max_message_size)

    @trace_function
    def send(self, message: RuntimeEnvelope) -> None:
        if self.sock is None:
            raise RuntimeError("worker session is not connected")
        send_message(self.sock, message)

    @trace_function
    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        except OSError:
            pass
        finally:
            self.sock = None


