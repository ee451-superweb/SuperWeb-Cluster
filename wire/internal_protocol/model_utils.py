"""Shared helpers used by wire-model dataclasses.

Use this module when multiple wire-model files need small compatibility helpers
without redefining them in every control-plane or data-plane model file.
"""

from __future__ import annotations


def initvar_or_default(value, default):
    """Normalize ``InitVar`` shadow values used by compatibility constructors.

    Args:
        value: Raw field initializer value supplied during dataclass construction.
        default: Fallback value to use when the initializer is a property object.

    Returns:
        The caller-provided default when the initializer is a property object,
        otherwise the original value.
    """

    return default if isinstance(value, property) else value
