"""Shared control-plane enums and envelope types.

Use this module when code needs the common runtime-envelope wrapper or enum
values that are shared across internal and external control-plane messages.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wire.external_protocol.control_plane import (
        ClientRequestOk,
        ClientInfoReply,
        ClientInfoRequest,
        ClientJoin,
        ClientRequest,
        ClientResponse,
    )
    from wire.internal_protocol.control_plane import (
        ArtifactRelease,
        Heartbeat,
        HeartbeatOk,
        RegisterOk,
        RegisterWorker,
        TaskAccept,
        TaskAssign,
        TaskFail,
        TaskResult,
        WorkerUpdate,
    )


class MessageKind(enum.IntEnum):
    """Supported protobuf envelope types."""

    UNSPECIFIED = 0
    REGISTER_WORKER = 1
    REGISTER_OK = 2
    HEARTBEAT = 3
    HEARTBEAT_OK = 4
    CLIENT_JOIN = 5
    CLIENT_REQUEST = 6
    CLIENT_RESPONSE = 7
    TASK_ASSIGN = 8
    TASK_ACCEPT = 9
    TASK_FAIL = 10
    TASK_RESULT = 11
    ARTIFACT_RELEASE = 12
    WORKER_UPDATE = 13
    CLIENT_INFO_REQUEST = 14
    CLIENT_INFO_REPLY = 15
    CLIENT_REQUEST_OK = 16


class NodeStatus(enum.IntEnum):
    """Compute-node operational status reported in heartbeat responses."""

    UNKNOWN = 0
    IDLE = 1
    BUSY = 2
    ERROR = 3


class TransferMode(enum.IntEnum):
    """Main-node hint describing whether a task should prefer inline or artifact delivery."""

    UNSPECIFIED = 0
    INLINE_PREFERRED = 1
    ARTIFACT_PREFERRED = 2
    ARTIFACT_REQUIRED = 3


@dataclass(slots=True)
class RuntimeEnvelope:
    """Framed protobuf message used on the TCP runtime channel."""

    kind: MessageKind
    register_worker: RegisterWorker | None = None
    register_ok: RegisterOk | None = None
    heartbeat: Heartbeat | None = None
    heartbeat_ok: HeartbeatOk | None = None
    client_join: ClientJoin | None = None
    client_request_ok: ClientRequestOk | None = None
    client_request: ClientRequest | None = None
    client_response: ClientResponse | None = None
    task_assign: TaskAssign | None = None
    task_accept: TaskAccept | None = None
    task_fail: TaskFail | None = None
    task_result: TaskResult | None = None
    artifact_release: ArtifactRelease | None = None
    worker_update: WorkerUpdate | None = None
    client_info_request: ClientInfoRequest | None = None
    client_info_reply: ClientInfoReply | None = None
