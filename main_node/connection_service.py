"""Accept and register worker or client runtime TCP sessions.

Use this module when the main node needs to receive new runtime connections,
classify them as workers or clients, and hand them off to the registry.
"""

from __future__ import annotations

import socket
import threading

from adapters import network
from adapters.audit_log import write_audit_event
from core.constants import MAIN_NODE_NAME, STATUS_OK
from core.tracing import trace_function
from wire.internal_protocol.transport import (
    MessageKind,
    build_client_response,
    build_register_ok,
    describe_message_kind,
    recv_message,
    send_message,
)


class RuntimeConnectionService:
    """Own runtime-session registration and connection handoff."""

    def __init__(
        self,
        *,
        config,
        registry,
        format_reported_hardware,
        print_cluster_compute_capacity,
        serve_client_connection,
        cluster_counts,
        on_runtime_connection_registered=None,
        logger=None,
    ) -> None:
        """Capture registration helpers and runtime callbacks.

        Args:
            config: Main-node runtime configuration.
            registry: Registry that stores worker and client connections.
            format_reported_hardware: Formatter used for worker registration logs.
            print_cluster_compute_capacity: Callback that prints cluster capacity.
            serve_client_connection: Entry point for a registered client session.
            cluster_counts: Callback returning current worker and client counts.
            on_runtime_connection_registered: Optional post-registration hook.
            logger: Logger used for audit events and rejection warnings.
        """
        self.config = config
        self.registry = registry
        self.format_reported_hardware = format_reported_hardware
        self.print_cluster_compute_capacity = print_cluster_compute_capacity
        self.serve_client_connection = serve_client_connection
        self.cluster_counts = cluster_counts
        self.on_runtime_connection_registered = on_runtime_connection_registered
        self.logger = logger

    @trace_function
    def register_worker_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        main_node_ip: str,
        register_worker,
    ):
        """Register one worker connection and send REGISTER_OK back.

        Use this after parsing a ``REGISTER_WORKER`` message so the main node can
        assign a runtime id, store the socket, and acknowledge the worker.

        Args:
            client_sock: Connected runtime socket from the worker.
            addr: Remote address tuple reported by ``accept``.
            main_node_ip: Public main-node IP returned in ``REGISTER_OK``.
            register_worker: Parsed worker registration payload.

        Returns:
            The registry connection object created for the worker.
        """
        payload = register_worker
        connection = self.registry.register_worker(
            node_name=payload.node_name,
            peer_address=addr[0],
            peer_port=addr[1],
            hardware=payload.hardware,
            performance=payload.performance,
            sock=client_sock,
        )
        send_message(
            client_sock,
            build_register_ok(
                main_node_ip=main_node_ip,
                main_node_port=self.config.tcp_port,
                main_node_name=MAIN_NODE_NAME,
                node_id=connection.runtime_id,
            ),
        )
        write_audit_event(
            f"Registered compute node {connection.node_name} "
            f"id={connection.runtime_id} "
            f"from {connection.peer_address}:{connection.peer_port} "
            f"cpu={connection.hardware.logical_cpu_count} memory_bytes={connection.hardware.memory_bytes} "
            f"reported_hardware={connection.performance.hardware_count if connection.performance else 0} "
            f"ranking={self.format_reported_hardware(connection)}",
            stdout=True,
            logger=self.logger,
        )
        self.print_cluster_compute_capacity()
        if self.on_runtime_connection_registered is not None:
            self.on_runtime_connection_registered(connection)
        return connection

    @trace_function
    def register_client_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        client_join,
    ):
        """Register one client connection and start its session thread.

        Use this after parsing a ``CLIENT_JOIN`` message so the main node can
        assign a client id, acknowledge the join, and start request handling.

        Args:
            client_sock: Connected runtime socket from the client.
            addr: Remote address tuple reported by ``accept``.
            client_join: Parsed client join payload.

        Returns:
            The registry connection object created for the client.
        """
        connection = self.registry.register_client(
            node_name=client_join.client_name,
            peer_address=addr[0],
            peer_port=addr[1],
            sock=client_sock,
        )
        worker_count, client_count = self.cluster_counts()
        send_message(
            client_sock,
            build_client_response(
                request_id="join",
                status_code=STATUS_OK,
                worker_count=worker_count,
                client_count=client_count,
                client_id=connection.runtime_id,
            ),
        )
        write_audit_event(
            f"Registered client {connection.node_name} "
            f"id={connection.runtime_id} "
            f"from {connection.peer_address}:{connection.peer_port}",
            stdout=True,
            logger=self.logger,
        )
        if self.on_runtime_connection_registered is not None:
            self.on_runtime_connection_registered(connection)
        client_thread = threading.Thread(
            target=self.serve_client_connection,
            args=(connection,),
            name=f"superweb-client-{connection.node_name}",
            daemon=True,
        )
        client_thread.start()
        return connection

    @trace_function
    def register_runtime_connection(
        self,
        client_sock: socket.socket,
        addr: tuple[str, int],
        main_node_ip: str,
    ):
        """Receive and accept one worker or client registration.

        Use this right after ``accept`` so the first runtime message can decide
        whether the socket belongs to a worker or a client.

        Args:
            client_sock: Newly accepted runtime socket.
            addr: Remote address tuple reported by ``accept``.
            main_node_ip: Public main-node IP returned to workers.

        Returns:
            The registered connection object for the new peer.
        """

        client_sock.settimeout(self.config.runtime_socket_timeout)
        message = recv_message(client_sock, max_size=self.config.max_message_size)
        if message is None:
            raise ConnectionError("peer closed the TCP session before registration")

        if message.kind == MessageKind.REGISTER_WORKER and message.register_worker is not None:
            return self.register_worker_connection(client_sock, addr, main_node_ip, message.register_worker)
        if message.kind == MessageKind.CLIENT_JOIN and message.client_join is not None:
            return self.register_client_connection(client_sock, addr, message.client_join)

        raise ValueError(
            f"expected REGISTER_WORKER or CLIENT_JOIN, got {describe_message_kind(message.kind)}"
        )

    @trace_function
    def accept_runtime_connections(
        self,
        server_sock: socket.socket,
        main_node_ip: str,
        *,
        runtime_should_stop,
    ) -> None:
        """Accept and classify runtime sockets until shutdown.

        Use this in the main-node accept thread to keep the runtime port open
        for both worker registrations and client join requests.

        Args:
            server_sock: Listening TCP socket for runtime traffic.
            main_node_ip: Public main-node IP shared with new workers.
            runtime_should_stop: Callback that reports shutdown state.

        Returns:
            ``None`` after the accept loop exits.
        """

        while not runtime_should_stop():
            try:
                client_sock, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if runtime_should_stop():
                    return
                raise

            try:
                self.register_runtime_connection(client_sock, addr, main_node_ip)
            except (OSError, ValueError, ConnectionError) as exc:
                if self.logger is not None:
                    self.logger.warning(
                        "Rejected superweb-cluster runtime connection from %s:%s: %s",
                        addr[0],
                        addr[1],
                        exc,
                    )
                network.safe_close(client_sock)
