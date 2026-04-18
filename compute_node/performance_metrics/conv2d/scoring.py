"""Define the linear scoring rule used by conv2d benchmark reports.

Use this module when the conv2d benchmark needs to convert measured latency
into a bounded score comparable across backends for the same workload.
"""

from __future__ import annotations

MAX_SCORE = 1000.0


def linear_time_score(
    elapsed_seconds: float,
    *,
    ideal_seconds: float,
    zero_score_seconds: float,
    max_score: float = MAX_SCORE,
) -> float:
    """Map runtime to a linear score where lower runtime means higher score.

    Args:
        elapsed_seconds: Measured wall-clock runtime for one trial.
        ideal_seconds: Runtime that should receive the maximum score.
        zero_score_seconds: Runtime at or above which the score becomes zero.
        max_score: Upper bound of the scoring range.

    Returns:
        The linearly interpolated score for the measured runtime.
    """

    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    if ideal_seconds <= 0:
        raise ValueError("ideal_seconds must be positive")
    if zero_score_seconds <= ideal_seconds:
        raise ValueError("zero_score_seconds must be greater than ideal_seconds")

    if elapsed_seconds <= ideal_seconds:
        return max_score
    if elapsed_seconds >= zero_score_seconds:
        return 0.0

    span = zero_score_seconds - ideal_seconds
    remaining = zero_score_seconds - elapsed_seconds
    return max_score * (remaining / span)


def scoring_formula_description() -> str:
    """Return a human-readable explanation of the scoring rule.

    Args:
        None.

    Returns:
        A short formula string describing the scoring function.
    """

    return (
        "score = clamp(max_score * (zero_score_seconds - elapsed_seconds) / "
        "(zero_score_seconds - ideal_seconds), 0, max_score)"
    )
