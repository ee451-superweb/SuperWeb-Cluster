"""Shared helpers for keeping unit tests deterministic and inexpensive."""

from __future__ import annotations

import os
import unittest

RUN_INTEGRATION_TESTS = os.environ.get("SUPERWEB_RUN_INTEGRATION_TESTS") == "1"


def require_integration(reason: str):
    """Skip a test unless explicitly opted into integration coverage."""

    return unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        f"{reason} Set SUPERWEB_RUN_INTEGRATION_TESTS=1 to enable.",
    )
