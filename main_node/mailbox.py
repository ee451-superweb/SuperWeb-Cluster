"""Route runtime messages for one TCP connection to the right waiting consumer.

Use this module when one runtime socket needs concurrent readers and waiters,
such as the main node waiting for task replies while also accepting client
requests and heartbeat acknowledgements on the same connection.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import defaultdict, deque

from wire.internal_protocol.transport import MessageKind


class RuntimeConnectionMailbox:
    """Store queued runtime messages for one connection in a thread-safe way."""

    def __init__(self) -> None:
        """Create empty queues for client requests, task replies, and heartbeat acks.

        Args: self runtime mailbox instance being initialized.
        Returns: None after the internal queues and close state are ready.
        """
        self._condition = threading.Condition()
        self._closed_reason: str | None = None
        self._client_requests = deque()
        self._task_messages: dict[str, deque] = defaultdict(deque)
        self._heartbeat_acks: dict[int, deque] = defaultdict(deque)

    def publish(self, message) -> None:
        """Use this when the reader loop receives a new envelope for this connection.

        Args: message decoded runtime envelope to append to the matching queue.
        Returns: None after the message is queued and waiters are notified.
        """
        with self._condition:
            if message.kind == MessageKind.CLIENT_REQUEST and message.client_request is not None:
                self._client_requests.append(message)
            elif message.kind == MessageKind.HEARTBEAT_OK and message.heartbeat_ok is not None:
                self._heartbeat_acks[message.heartbeat_ok.heartbeat_unix_time_ms].append(message)
            elif message.kind == MessageKind.TASK_ACCEPT and message.task_accept is not None:
                self._task_messages[message.task_accept.task_id].append(message)
            elif message.kind == MessageKind.TASK_FAIL and message.task_fail is not None:
                self._task_messages[message.task_fail.task_id].append(message)
            elif message.kind == MessageKind.TASK_RESULT and message.task_result is not None:
                self._task_messages[message.task_result.task_id].append(message)
            self._condition.notify_all()

    def close(self, reason: str) -> None:
        """Use this to wake all waiters after the socket or session becomes unusable.

        Args: reason human-readable close cause stored for future wait errors.
        Returns: None after the close reason is recorded and waiters are notified.
        """
        with self._condition:
            if self._closed_reason is None:
                self._closed_reason = reason
            self._condition.notify_all()

    def wait_for_client_request(self, timeout: float):
        """Use this on a client connection while waiting for the next CLIENT_REQUEST.

        Args: timeout maximum seconds to block before raising a socket timeout.
        Returns: The next queued client-request envelope for this connection.
        """
        return self._wait_for(
            timeout,
            lambda: self._client_requests.popleft() if self._client_requests else None,
            "client request",
        )

    def wait_for_task_message(self, task_id: str, timeout: float | None = None):
        """Use this after sending one task to wait for its accept/fail/result message.

        Args: task_id target task identifier and timeout maximum wait in seconds.
        Returns: The next queued task envelope that matches the requested task id.
        """
        return self._wait_for(
            timeout,
            lambda: self._pop_task_message(task_id),
            f"task message for {task_id}",
        )

    def wait_for_heartbeat_ok(self, heartbeat_unix_time_ms: int, timeout: float):
        """Use this after one heartbeat send to wait for the matching HEARTBEAT_OK.

        Args: heartbeat_unix_time_ms sent heartbeat timestamp and timeout in seconds.
        Returns: The queued heartbeat acknowledgement for that timestamp.
        """
        return self._wait_for(
            timeout,
            lambda: self._pop_heartbeat_ack(heartbeat_unix_time_ms),
            f"heartbeat ack for {heartbeat_unix_time_ms}",
        )

    def _pop_task_message(self, task_id: str):
        """Use this internal helper to remove one queued task message by task id.

        Args: task_id task identifier whose queue should be popped.
        Returns: One matching message, or None when no queued message is available.
        """
        queue = self._task_messages.get(task_id)
        if not queue:
            return None
        message = queue.popleft()
        if not queue:
            self._task_messages.pop(task_id, None)
        return message

    def _pop_heartbeat_ack(self, heartbeat_unix_time_ms: int):
        """Use this internal helper to remove one queued heartbeat ack by timestamp.

        Args: heartbeat_unix_time_ms sent heartbeat timestamp used as the key.
        Returns: One matching heartbeat acknowledgement, or None if absent.
        """
        queue = self._heartbeat_acks.get(heartbeat_unix_time_ms)
        if not queue:
            return None
        message = queue.popleft()
        if not queue:
            self._heartbeat_acks.pop(heartbeat_unix_time_ms, None)
        return message

    def _wait_for(self, timeout: float | None, pop_message, description: str):
        """Use this shared waiter when any mailbox consumer needs blocking queue access.

        Args: timeout max wait seconds, pop_message dequeue callback, and description for errors.
        Returns: The first message produced by pop_message before timeout or close.
        """
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._condition:
            while True:
                message = pop_message()
                if message is not None:
                    return message
                if self._closed_reason is not None:
                    raise ConnectionError(self._closed_reason)
                if deadline is None:
                    self._condition.wait(0.5)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise socket.timeout(f"timed out waiting for {description}")
                self._condition.wait(remaining)
