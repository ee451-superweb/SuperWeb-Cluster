"""Linear scoring helpers for performance benchmarks."""

from __future__ import annotations

MAX_SCORE = 1000.0


def linear_time_score(
    elapsed_seconds: float,
    *,
    ideal_seconds: float,
    zero_score_seconds: float,
    max_score: float = MAX_SCORE,
) -> float:
    """Map runtime to a linear score where lower runtime means higher score."""

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
    """Return a human-readable explanation of the scoring rule."""

    return (
        "score = clamp(max_score * (zero_score_seconds - elapsed_seconds) / "
        "(zero_score_seconds - ideal_seconds), 0, max_score)"
    )
