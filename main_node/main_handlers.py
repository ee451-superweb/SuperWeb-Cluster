"""Handle runtime-side worker/client messages and connection lifecycle on the main node.

Use this module when the main-node runtime wants one place to own socket-reader
threads, worker-update handling, client-info replies, and role-aware cleanup.
"""

from __future__ import annotations

import logging
import socket
import threading
from contextlib import nullcontext

from adapters import network
from adapters.audit_log import write_audit_event
from core.constants import (
    DEFAULT_CLIENT_INFO_TIMEOUT,
    RUNTIME_MSG_CLIENT_INFO_REPLY,
    RUNTIME_MSG_CLIENT_INFO_REQUEST,
    RUNTIME_MSG_WORKER_UPDATE,
    RUNTIME_ROLE_CLIENT,
    RUNTIME_ROLE_WORKER,
)
from core.tracing import trace_function
from wire.internal_protocol.transport import MessageKind, build_client_info_reply, recv_message, send_message


class RuntimeConnectionHandler:
    """Own main-node runtime message routing and connection cleanup."""

    def __init__(
        self,
        *,
        config,
        registry,
        format_reported_hardware,
        print_cluster_compute_capacity,
        runtime_should_stop,
        logger=None,
    ) -> None:
        """Store dependencies used while reading and maintaining runtime peers.

        Args:
            config: Main-node runtime configuration with socket timeouts.
            registry: Registry containing worker and client runtime peers.
            format_reported_hardware: Formatter used in worker-update logs.
            print_cluster_compute_capacity: Callback that prints cluster totals.
            runtime_should_stop: Callback returning whether shutdown is in progress.
            logger: Logger used for audit events and removal warnings.
        """
        self.config = config
        self.registry = registry
        self.format_reported_hardware = format_reported_hardware
        self.print_cluster_compute_capacity = print_cluster_compute_capacity
        self.runtime_should_stop = runtime_should_stop
        self.logger = logger

    @trace_function
    def start_runtime_connection_reader(self, connection) -> None:
        """Launch the background reader thread for one registered runtime peer.

        Args:
            connection: Registered worker or client connection that needs mailbox reads.

        Returns:
            ``None`` after the reader thread has been started.
        """
        thread = threading.Thread(
            target=self.runtime_connection_reader_loop,
            args=(connection,),
            name=f"runtime-reader-{connection.runtime_id}",
            daemon=True,
        )
        thread.start()

    @trace_function
    def runtime_connection_reader_loop(self, connection) -> None:
        """Drain one runtime socket into its mailbox until disconnect or shutdown.

        Args:
            connection: Registered runtime peer whose TCP stream should be read.

        Returns:
            ``None`` after the reader exits.
        """
        mailbox = getattr(connection, "mailbox", None)
        if mailbox is None:
            return

        while not self.runtime_should_stop():
            try:
                message = recv_message(connection.sock, max_size=self.config.max_message_size)
            except socket.timeout:
                continue
            except OSError as exc:
                mailbox.close(str(exc))
                if not self.runtime_should_stop():
                    self.remove_runtime_connection(connection, str(exc))
                return
            except (ValueError, ConnectionError) as exc:
                mailbox.close(str(exc))
                if not self.runtime_should_stop():
                    self.remove_runtime_connection(connection, str(exc))
                return

            if message is None:
                reason = "peer closed the TCP session"
                mailbox.close(reason)
                if not self.runtime_should_stop():
                    self.remove_runtime_connection(connection, reason)
                return

            try:
                if (
                    connection.role == RUNTIME_ROLE_WORKER
                    and message.kind == MessageKind.WORKER_UPDATE
                    and message.worker_update is not None
                ):
                    self.handle_worker_update(connection, message.worker_update)
                    continue
                if (
                    connection.role == RUNTIME_ROLE_CLIENT
                    and message.kind == MessageKind.CLIENT_INFO_REQUEST
                    and message.client_info_request is not None
                ):
                    self.handle_client_info_request(connection, message.client_info_request)
                    continue
            except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
                mailbox.close(str(exc))
                if not self.runtime_should_stop():
                    self.remove_runtime_connection(connection, str(exc))
                return

            mailbox.publish(message)

    def handle_worker_update(self, connection, worker_update) -> None:
        """Apply a worker's updated abstract performance summary to the registry.

        Args:
            connection: Worker connection that sent the update.
            worker_update: Parsed ``WORKER_UPDATE`` payload.

        Returns:
            ``None`` after registry state and logs are updated.
        """
        if worker_update.node_id and worker_update.node_id != connection.runtime_id:
            raise ValueError(
                f"received {RUNTIME_MSG_WORKER_UPDATE} for unexpected worker id "
                f"{worker_update.node_id}; expected {connection.runtime_id}"
            )
        updated = self.registry.update_worker_performance_by_runtime_id(
            connection.runtime_id,
            worker_update.performance,
        )
        if updated is None:
            raise ValueError(f"unable to apply {RUNTIME_MSG_WORKER_UPDATE} for unknown worker {connection.runtime_id}")
        write_audit_event(
            f"{RUNTIME_MSG_WORKER_UPDATE} from {connection.node_name} "
            f"id={connection.runtime_id} ranking={self.format_reported_hardware(updated)}",
            logger=self.logger,
        )
        self.print_cluster_compute_capacity()

    def handle_client_info_request(self, connection, client_info_request) -> None:
        """Reply to one client's periodic info poll with its active request ids.

        Args:
            connection: Client connection that sent the request.
            client_info_request: Parsed ``CLIENT_INFO_REQUEST`` payload.

        Returns:
            ``None`` after a matching ``CLIENT_INFO_REPLY`` has been sent.
        """
        client_id = client_info_request.client_id or connection.runtime_id
        if client_id != connection.runtime_id:
            raise ValueError(
                f"received {RUNTIME_MSG_CLIENT_INFO_REQUEST} for unexpected client id "
                f"{client_id}; expected {connection.runtime_id}"
            )
        active_task_ids = self.registry.get_client_active_task_ids(connection.peer_id)
        timeout_ms = int(DEFAULT_CLIENT_INFO_TIMEOUT * 1000)
        with self._resolve_connection_lock(getattr(connection, "io_lock", None)):
            send_message(
                connection.sock,
                build_client_info_reply(
                    client_id=connection.runtime_id,
                    request_timestamp_ms=client_info_request.timestamp_ms,
                    timeout_ms=timeout_ms,
                    has_active_tasks=bool(active_task_ids),
                    active_task_ids=active_task_ids,
                ),
            )
        write_audit_event(
            f"{RUNTIME_MSG_CLIENT_INFO_REPLY} to {connection.node_name} "
            f"id={connection.runtime_id} active_tasks={list(active_task_ids)}",
            logger=self.logger,
        )

    def _resolve_connection_lock(self, lock):
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

    @trace_function
    def remove_worker_connection(self, connection, reason: str) -> None:
        """Remove one worker connection, close its socket, and update cluster logs.

        Args:
            connection: Worker runtime peer being removed.
            reason: Human-readable reason for the removal.

        Returns:
            ``None`` after the worker is gone from the registry.
        """
        removed = self.registry.remove_worker(connection.peer_id)
        if removed is None:
            return
        removed.mailbox.close(reason)
        network.safe_close(removed.sock)
        write_audit_event(
            f"Removed compute node {connection.node_name} at {connection.peer_address}:{connection.peer_port}: {reason}",
            logger=self.logger,
            level=logging.WARNING,
        )
        self.print_cluster_compute_capacity()

    @trace_function
    def remove_client_connection(self, connection, reason: str) -> None:
        """Remove one client connection and close its socket.

        Args:
            connection: Client runtime peer being removed.
            reason: Human-readable reason for the removal.

        Returns:
            ``None`` after the client is gone from the registry.
        """
        removed = self.registry.remove_client(connection.peer_id)
        if removed is None:
            return
        removed.mailbox.close(reason)
        network.safe_close(removed.sock)
        write_audit_event(
            f"Removed client {connection.node_name} at {connection.peer_address}:{connection.peer_port}: {reason}",
            logger=self.logger,
            level=logging.WARNING,
        )

    def remove_runtime_connection(self, connection, reason: str) -> None:
        """Remove one runtime peer without the caller caring about its role.

        Args:
            connection: Worker or client runtime peer being removed.
            reason: Human-readable reason for the removal.

        Returns:
            ``None`` after role-specific cleanup has run.
        """
        role = getattr(connection, "role", RUNTIME_ROLE_WORKER)
        if role == RUNTIME_ROLE_CLIENT:
            self.remove_client_connection(connection, reason)
            return
        self.remove_worker_connection(connection, reason)

    @trace_function
    def close_runtime_connections(self) -> None:
        """Close and clear every runtime peer during main-node shutdown.

        Args:
            None.

        Returns:
            ``None`` after all registry connections have been closed.
        """
        for connection in self.registry.clear():
            connection.mailbox.close("main-node runtime stopping")
            network.safe_close(connection.sock)
