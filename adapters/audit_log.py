"""Helpers for routing operator-facing audit events to file logs and optional stdout."""

from __future__ import annotations

import logging
import sys

from core.constants import LOGGER_NAME

_AUDIT_LOGGER_NAME = f"{LOGGER_NAME}.audit"


def get_audit_logger(name: str = "runtime") -> logging.Logger:
    """Return one audit child logger underneath the project logger tree."""

    suffix = name.strip(".")
    if not suffix:
        return logging.getLogger(_AUDIT_LOGGER_NAME)
    return logging.getLogger(f"{_AUDIT_LOGGER_NAME}.{suffix}")


def write_audit_event(
    message: str,
    *,
    stdout: bool = False,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> None:
    """Write one audit event to the log file and optionally mirror it to stdout."""

    target_logger = logger if logger is not None else get_audit_logger()
    target_logger.log(level, message)
    if stdout:
        print(message, flush=True)


def write_diag_event(
    message: str,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log one diagnostic event and mirror it to the console in verbose mode.

    Use this for operator-facing checkpoint markers (``[DIAG] ...``) that should
    always land in the role log file and, when ``--verbose`` is active, also
    surface in the live console. We write to ``stderr`` (not ``stdout``) because
    on Windows the bootstrap's stdout is often block-buffered by a launcher
    wrapper so ``print(..., flush=True)`` can get swallowed; stderr stays
    unbuffered in the same terminal and is visible immediately.
    """

    # Local import so callsites do not need to import logging_setup; also avoids
    # a circular import during module initialization.
    from core.logging_setup import is_verbose

    target_logger = logger if logger is not None else get_audit_logger()
    target_logger.log(logging.INFO, message)
    if is_verbose():
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
