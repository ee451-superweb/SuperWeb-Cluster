"""Placeholder module for future standalone heartbeat monitoring logic.

Use this module only as a stub during development. The real runtime heartbeat
flow currently lives in ``main_node.heartbeat_service``.
"""


class HeartbeatMonitor:
    """Stub heartbeat monitor reserved for future scheduler refactors."""

    def poll(self) -> None:
        """Raise until a dedicated heartbeat monitor is implemented.

        Use this only as a placeholder entry point so callers fail fast instead
        of assuming a real background heartbeat monitor exists already.

        Args:
            None.

        Returns:
            This function never returns successfully.
        """
        raise NotImplementedError("Heartbeat monitoring is not implemented in the kickoff version.")
