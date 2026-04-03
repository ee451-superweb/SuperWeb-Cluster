"""Helpers for lightweight runtime function tracing."""

from __future__ import annotations

import functools
import logging
from typing import Callable, ParamSpec, TypeVar

from constants import LOGGER_NAME

P = ParamSpec("P")
R = TypeVar("R")


def trace_function(func: Callable[P, R]) -> Callable[P, R]:
    """Emit a visible runtime message whenever the wrapped function is entered."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        qualified_name = f"{func.__module__}.{func.__qualname__}"
        message = f"[TRACE] Entering {qualified_name}"

        # If logging is already configured, keep trace output aligned with the rest
        # of the program. Otherwise fall back to print so early startup still shows
        # function entry order.
        if logging.getLogger().handlers:
            logging.getLogger(LOGGER_NAME).info(message)
        else:
            print(message, flush=True)

        return func(*args, **kwargs)

    return wrapper
