"""Helpers for lightweight runtime function tracing."""

from __future__ import annotations

import functools
import logging
from typing import Callable, ParamSpec, TypeVar

from core.constants import LOGGER_NAME

P = ParamSpec("P")
R = TypeVar("R")


def trace_function(func: Callable[P, R]) -> Callable[P, R]:
    """Emit a visible runtime message whenever the wrapped function is entered."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Log or print one trace line before calling the wrapped function."""
        qualified_name = f"{func.__module__}.{func.__qualname__}"
        message = f"[TRACE] Entering {qualified_name}"

        # Only emit trace lines once logging is configured. Early bootstrap
        # should stay quiet on stdout so audit-log routing remains consistent.
        if logging.getLogger().handlers:
            logging.getLogger(LOGGER_NAME).info(message)

        return func(*args, **kwargs)

    return wrapper

