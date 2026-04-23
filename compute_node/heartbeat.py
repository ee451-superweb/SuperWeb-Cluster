"""Track the last scheduler heartbeat seen by the compute node.

Use this module when the compute node needs a tiny state holder for the most
recent heartbeat received from the main node.
"""

from __future__ import annotations

from dataclasses import dataclass

from wire.internal_protocol.transport import Heartbeat
from core.tracing import trace_function


@dataclass(slots=True)
class WorkerHeartbeat:
    """Track the last heartbeat received from the scheduler."""

    last_heartbeat: Heartbeat | None = None

    @trace_function
    def respond(self, heartbeat: Heartbeat) -> None:
        """Record the latest heartbeat received from the main node.

        Args:
            heartbeat: Heartbeat payload that just arrived from the scheduler.
        """
        self.last_heartbeat = heartbeat


