"""Heartbeat tracking for compute-node runtime."""

from __future__ import annotations

from dataclasses import dataclass

from wire.runtime import Heartbeat
from app.trace_utils import trace_function


@dataclass(slots=True)
class WorkerHeartbeat:
    """Track the last heartbeat received from the scheduler."""

    last_heartbeat: Heartbeat | None = None

    @trace_function
    def respond(self, heartbeat: Heartbeat) -> None:
        self.last_heartbeat = heartbeat


