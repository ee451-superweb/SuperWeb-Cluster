"""Shared test-package setup for stable, low-noise unit-test runs."""

from __future__ import annotations

import logging

# Route trace_function() calls through logging instead of stdout/stderr during
# unit tests. The root logger still defaults to WARNING, so trace-level INFO
# messages stay quiet unless a test explicitly configures logging.
if not logging.getLogger().handlers:
    logging.getLogger().addHandler(logging.NullHandler())
