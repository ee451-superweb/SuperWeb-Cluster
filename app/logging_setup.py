"""Logging configuration helpers."""

import logging

from app.constants import LOGGER_NAME
from app.trace_utils import trace_function


@trace_function
def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure console logging and return the app logger."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)

