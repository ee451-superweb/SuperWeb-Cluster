"""Helpers for routing operator-facing audit events to file logs and optional stdout."""

from __future__ import annotations

import logging

from app.constants import LOGGER_NAME

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
