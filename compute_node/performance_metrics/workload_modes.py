"""Shared benchmark workload-mode constants and inclusion helpers.

Use this module when a benchmark entrypoint needs to decide which dataset sizes
belong to a requested workload mode such as ``small``, ``mid``, or ``full``.
"""

from __future__ import annotations

WORKLOAD_MODE_SMALL = "small"
WORKLOAD_MODE_MID = "mid"
WORKLOAD_MODE_MEDIUM = "medium"
WORKLOAD_MODE_LARGE = "large"
WORKLOAD_MODE_FULL = "full"
WORKLOAD_MODE_CUSTOM = "custom"

BENCHMARK_WORKLOAD_MODE_CHOICES = (
    WORKLOAD_MODE_SMALL,
    WORKLOAD_MODE_MID,
    WORKLOAD_MODE_MEDIUM,
    WORKLOAD_MODE_LARGE,
    WORKLOAD_MODE_FULL,
)


def uses_small_dataset(mode: str) -> bool:
    """Return whether the requested mode should include the small dataset.

    Args:
        mode: Requested workload mode string.

    Returns:
        ``True`` when the small dataset should be generated or benchmarked.
    """

    return mode in {WORKLOAD_MODE_SMALL, WORKLOAD_MODE_FULL}


def uses_medium_dataset(mode: str) -> bool:
    """Return whether the requested mode should include the mid-sized dataset.

    Args:
        mode: Requested workload mode string.

    Returns:
        ``True`` when the mid-sized dataset should be generated or benchmarked.
    """

    return mode in {WORKLOAD_MODE_MID, WORKLOAD_MODE_MEDIUM}


def uses_large_dataset(mode: str) -> bool:
    """Return whether the requested mode should include the large dataset.

    Args:
        mode: Requested workload mode string.

    Returns:
        ``True`` when the large dataset should be generated or benchmarked.
    """

    return mode in {WORKLOAD_MODE_LARGE, WORKLOAD_MODE_FULL}
