"""Audit logging placeholder."""

import logging


def get_audit_logger(name: str = "audit") -> logging.Logger:
    """Return a logger placeholder for future audit trails."""

    return logging.getLogger(name)
