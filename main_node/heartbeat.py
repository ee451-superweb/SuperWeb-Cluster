"""Coordinate runtime heartbeats between the main node and registered workers.

Use this module when the main node needs to probe worker liveness, wait for
`HEARTBEAT_OK` acknowledgements, and remove dead connections after retries.
"""

from __future__ import annotations

import logging
import socket
import time

from adapters.audit_log import write_audit_event
from core.constants import MAIN_NODE_NAME, RUNTIME_MSG_HEARTBEAT, RUNTIME_MSG_HEARTBEAT_OK
from core.tracing import trace_function
from main_node.mailbox import RuntimeConnectionMailbox
from wire.internal_protocol.transport import MessageKind, build_heartbeat, describe_message_kind, recv_message, send_message


class HeartbeatCoordinator:
    """Send periodic heartbeats and record worker liveness state."""

    def __init__(self, *, config, registry, remove_connection, logger=None) -> None:
        """Store runtime dependencies for future heartbeat rounds.

        Args:
            config: Main-node runtime configuration with timeout and retry limits.
            registry: Connection registry used to record heartbeat timestamps.
            remove_connection: Callback that evicts broken worker connections.
            logger: Logger used for heartbeat audit entries.
        """
        self.config = config
        self.registry = registry
        self._remove_connection = remove_connection
        self.logger = logger

    def _describe_connection_activity(self, connection, *, reported_task_ids: tuple[str, ...] = ()) -> str:
        """Summarize the worker's active task state for heartbeat logging.

        Use this when printing heartbeat traces so operator logs include either
        the worker-reported task ids or the main node's current local view.

        Args:
            connection: Registered worker connection being described.
            reported_task_ids: Task ids reported by the worker in HEARTBEAT_OK.

        Returns:
            A short human-readable suffix for log lines.
        """
        if reported_task_ids:
            return " active_task=" + ",".join(reported_task_ids)
        active_task_id = getattr(connection, "active_task_id", "")
        active_request_id = getattr(connection, "active_request_id", "")
        active_method = getattr(connection, "active_method", "")
        if active_task_id:
            return f" active_task={active_task_id} request_id={active_request_id or '<empty>'} method={active_method or '<empty>'}"
        if active_request_id:
            return f" active_request={active_request_id} method={active_method or '<empty>'}"
        return " active_task=<idle>"

    @trace_function
    def await_heartbeat_ok(self, connection, heartbeat_unix_time_ms: int) -> None:
        """Wait for one worker heartbeat acknowledgement and validate it.

        Use this immediately after sending a heartbeat so the main node can
        confirm the worker replied with the expected timestamp and worker id.

        Args:
            connection: Worker connection that should acknowledge the heartbeat.
            heartbeat_unix_time_ms: Timestamp carried by the outgoing heartbeat.

        Returns:
            ``None`` after registry state and logs have been updated.
        """
        mailbox = getattr(connection, "mailbox", None)
        if isinstance(mailbox, RuntimeConnectionMailbox):
            message = mailbox.wait_for_heartbeat_ok(
                heartbeat_unix_time_ms,
                timeout=self.config.runtime_socket_timeout,
            )
        else:
            message = recv_message(connection.sock, max_size=self.config.max_message_size)
        if message is None:
            raise ConnectionError("worker closed the TCP session during heartbeat")
        if message.kind != MessageKind.HEARTBEAT_OK or message.heartbeat_ok is None:
            raise ValueError(f"expected {RUNTIME_MSG_HEARTBEAT_OK}, got {describe_message_kind(message.kind)}")
        if message.heartbeat_ok.heartbeat_unix_time_ms != heartbeat_unix_time_ms:
            raise ValueError(
                "received HEARTBEAT_OK for unexpected heartbeat timestamp "
                f"{message.heartbeat_ok.heartbeat_unix_time_ms}"
            )
        if message.heartbeat_ok.node_id and message.heartbeat_ok.node_id != connection.runtime_id:
            raise ValueError(
                "received HEARTBEAT_OK for unexpected node id "
                f"{message.heartbeat_ok.node_id}; expected {connection.runtime_id}"
            )

        ack_time = (
            message.heartbeat_ok.received_unix_time_ms / 1000 if message.heartbeat_ok.received_unix_time_ms else time.time()
        )
        self.registry.mark_heartbeat(connection.peer_id, sent_at=ack_time)

    @trace_function
    def send_heartbeat_with_retry(self, connection) -> None:
        """Send one worker heartbeat with retry and eviction on failure.

        Use this for one worker during a heartbeat round. The method retries the
        configured number of times before removing the connection.

        Args:
            connection: Worker connection that should receive the heartbeat.

        Returns:
            ``None`` after the worker replies or is removed.
        """
        total_attempts = self.config.heartbeat_retry_count + 1
        last_error: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            heartbeat = build_heartbeat(MAIN_NODE_NAME)
            assert heartbeat.heartbeat is not None
            heartbeat_unix_time_ms = heartbeat.heartbeat.unix_time_ms

            try:
                with connection.io_lock:
                    send_message(connection.sock, heartbeat)
                self.await_heartbeat_ok(connection, heartbeat_unix_time_ms)
                return
            except (socket.timeout, OSError, ConnectionError, ValueError) as exc:
                last_error = exc
                if attempt < total_attempts:
                    write_audit_event(
                        f"{RUNTIME_MSG_HEARTBEAT} failure for {connection.node_name} "
                        f"at {connection.peer_address}:{connection.peer_port} "
                        f"attempt={attempt}/{total_attempts} after {exc}"
                        f"{self._describe_connection_activity(connection)}",
                        logger=self.logger,
                        level=logging.WARNING,
                    )
        failure_count = self.registry.record_heartbeat_failure(connection.peer_id)
        if failure_count is None:
            return
        write_audit_event(
            f"{RUNTIME_MSG_HEARTBEAT} failure for {connection.node_name} "
            f"at {connection.peer_address}:{connection.peer_port} "
            f"failure_count={failure_count}/{total_attempts} after {last_error}",
            logger=self.logger,
            level=logging.WARNING,
        )
        if failure_count >= total_attempts:
            self._remove_connection(
                connection,
                f"after {failure_count} consecutive heartbeat failures: {last_error}",
            )

    def _list_heartbeat_targets(self):
        """Return the connections that should receive the next heartbeat round.

        Use this helper so heartbeat rounds only target compute workers. Client
        sessions have their own polling path and should not receive runtime
        heartbeats from the main node.

        Args:
            None.

        Returns:
            A list of currently registered worker connections.
        """
        return self.registry.list_workers()

    @trace_function
    def send_heartbeat_once(self) -> None:
        """Probe every known runtime connection once.

        Use this from the main-node heartbeat loop to perform one full liveness
        sweep across the currently registered workers.

        Args:
            None.

        Returns:
            ``None`` after all targets have been processed.
        """
        targets = self._list_heartbeat_targets()
        if not targets:
            return
        for connection in targets:
            self.send_heartbeat_with_retry(connection)
