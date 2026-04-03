"""Runtime state definitions."""

from enum import Enum


class RuntimeState(str, Enum):
    """Minimal supervisor states for the kickoff version."""

    INIT = "INIT"
    DISCOVERY = "DISCOVERY"
    COMPUTE_NODE = "COMPUTE_NODE"
    MAIN_NODE = "MAIN_NODE"
    MANUAL_INPUT = "MANUAL_INPUT"
    IDLE = "IDLE"
    SHUTDOWN = "SHUTDOWN"
