"""Own the runtime request loop for one registered client session.

Use this module when the main node needs to read ``CLIENT_REQUEST`` messages,
track client request state in the registry, and send responses back over the
runtime TCP session.
"""

from __future__ import annotations

import socket
from contextlib import nullcontext

from app.constants import METHOD_SPATIAL_CONVOLUTION, RUNTIME_MSG_CLIENT_REQUEST, RUNTIME_MSG_CLIENT_RESPONSE
from app.trace_utils import trace_function
from main_node.runtime_mailbox import RuntimeConnectionMailbox
from wire.internal_protocol.runtime_transport import MessageKind, describe_message_kind, recv_message, send_message


def _resolve_connection_lock(lock):
    """Return a usable context manager for connection I/O locks.

    Args:
        lock: Candidate lock-like object stored on a runtime connection.

    Returns:
        The original lock when it supports ``with``, otherwise ``nullcontext()``.
    """
    lock_type = type(lock)
    if callable(getattr(lock_type, "__enter__", None)) and callable(getattr(lock_type, "__exit__", None)):
        return lock
    return nullcontext()


class ClientSessionService:
    """Serve one registered client from join until disconnect."""

    def __init__(self, *, config, registry, build_client_response_for_request, remove_client_connection) -> None:
        """Store runtime dependencies needed to serve a client session.

        Args:
            config: Main-node runtime configuration with message-size limits.
            registry: Registry used to track per-client request state.
            build_client_response_for_request: Handler that executes one request.
            remove_client_connection: Callback that removes broken client sockets.
        """
        self.config = config
        self.registry = registry
        self.build_client_response_for_request = build_client_response_for_request
        self.remove_client_connection = remove_client_connection

    def describe_client_request(self, request) -> str:
        """Render a concise workload summary for client-request logs.

        Use this when printing incoming client requests so operators can see the
        major dimensions of the request without inspecting raw payload bytes.

        Args:
            request: Parsed client request dataclass.

        Returns:
            A short textual summary of the workload dimensions.
        """
        if request.method == METHOD_SPATIAL_CONVOLUTION:
            return (
                f"tensor={request.tensor_h}x{request.tensor_w} "
                f"channels={request.channels_in}->{request.channels_out} "
                f"kernel={request.kernel_size} pad={request.padding} stride={request.stride}"
            )
        return f"vector_length={request.vector_length}"

    @trace_function
    def serve_client_connection(self, connection, runtime_should_stop) -> None:
        """Run the request/response loop for one registered client.

        Use this in the background client thread created after ``CLIENT_JOIN`` so
        the main node can continuously handle requests on that socket.

        Args:
            connection: Registered client connection object.
            runtime_should_stop: Callback that reports shutdown state.

        Returns:
            ``None`` after the client disconnects or is removed.
        """
        mailbox = getattr(connection, "mailbox", None)
        while not runtime_should_stop():
            try:
                if isinstance(mailbox, RuntimeConnectionMailbox):
                    message = mailbox.wait_for_client_request(self.config.runtime_socket_timeout)
                else:
                    message = recv_message(connection.sock, max_size=self.config.max_message_size)
            except socket.timeout:
                continue
            except (OSError, ValueError, ConnectionError) as exc:
                self.remove_client_connection(connection, str(exc))
                return

            if message is None:
                self.remove_client_connection(connection, "client closed the TCP session")
                return

            if message.kind != MessageKind.CLIENT_REQUEST or message.client_request is None:
                print(
                    f"Ignoring unexpected client runtime message from {connection.node_name}: "
                    f"{describe_message_kind(message.kind)}",
                    flush=True,
                )
                continue

            request = message.client_request
            self.registry.mark_client_request(connection.peer_id)
            if hasattr(self.registry, "mark_client_request_state"):
                self.registry.mark_client_request_state(
                    connection.peer_id,
                    request_id=request.request_id,
                    method=request.method,
                )
            print(
                f"{RUNTIME_MSG_CLIENT_REQUEST} from {request.client_name} "
                f"request_id={request.request_id} method={request.method} "
                f"{self.describe_client_request(request)} iteration_count={request.iteration_count}",
                flush=True,
            )
            try:
                response = self.build_client_response_for_request(request)
                if response.client_response is not None:
                    response.client_response.client_id = connection.runtime_id
                with _resolve_connection_lock(getattr(connection, "io_lock", None)):
                    send_message(connection.sock, response)
                print(
                    f"{RUNTIME_MSG_CLIENT_RESPONSE} to {request.client_name} "
                    f"request_id={request.request_id} status_code={response.client_response.status_code}",
                    flush=True,
                )
            except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
                self.remove_client_connection(connection, str(exc))
                return
            finally:
                if hasattr(self.registry, "clear_client_request_state"):
                    self.registry.clear_client_request_state(
                        connection.peer_id,
                        request_id=request.request_id,
                    )
