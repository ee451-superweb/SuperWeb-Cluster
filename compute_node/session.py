"""Manage the compute node's TCP session to the main node.

Use this module when the compute node needs to connect, register, exchange
runtime messages, and eventually close the persistent runtime session.
"""

from __future__ import annotations

import socket

from core.types import ComputePerformanceSummary, HardwareProfile
from core.constants import RUNTIME_MSG_REGISTER_OK
from wire.internal_protocol.transport import (
    MessageKind,
    RegisterOk,
    RuntimeEnvelope,
    build_register_worker,
    recv_message,
    send_message,
)
from core.tracing import trace_function


class WorkerSession:
    """Own the persistent runtime TCP session to the main node."""

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
        """Store connection settings for the future runtime session.

        Args:
            main_node_host: Main-node host name or IP address.
            main_node_port: Main-node runtime TCP port.
            connect_timeout: Timeout used when opening the socket.
            socket_timeout: Timeout applied to subsequent socket operations.
            max_message_size: Maximum accepted runtime message size in bytes.
        """
        self.main_node_host = main_node_host
        self.main_node_port = main_node_port
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout
        self.max_message_size = max_message_size
        self.sock: socket.socket | None = None

    @trace_function
    def connect(self) -> None:
        """Open the TCP session to the main node and apply socket timeouts."""
        self.sock = socket.create_connection((self.main_node_host, self.main_node_port), timeout=self.connect_timeout)
        self.sock.settimeout(self.socket_timeout)

    @trace_function
    def register(
        self,
        node_name: str,
        hardware: HardwareProfile,
        performance: ComputePerformanceSummary,
    ) -> RegisterOk:
        """Register this worker session with the main node.

        Args:
            node_name: Human-readable worker name.
            hardware: Local hardware profile advertised at registration.
            performance: Abstract performance summary advertised at registration.

        Returns:
            The parsed ``REGISTER_OK`` payload returned by the main node.
        """
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
        """Receive one runtime message from the main node.

        Returns:
            The next decoded runtime envelope, or ``None`` on EOF.
        """
        if self.sock is None:
            raise RuntimeError("worker session is not connected")
        return recv_message(self.sock, max_size=self.max_message_size)

    @trace_function
    def send(self, message: RuntimeEnvelope) -> None:
        """Send one runtime message to the main node.

        Args:
            message: Runtime envelope to send across the TCP session.
        """
        if self.sock is None:
            raise RuntimeError("worker session is not connected")
        send_message(self.sock, message)

    @trace_function
    def close(self) -> None:
        """Close the worker session socket if it is still open."""
        if self.sock is None:
            return
        try:
            self.sock.close()
        except OSError:
            pass
        finally:
            self.sock = None


