"""Helpers for proportional contiguous row-range partitioning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightedRange:
    """One contiguous slice assigned from a weighted partition."""

    index: int
    start: int
    end: int
    weight: float


def partition_contiguous_range(start: int, end: int, weights: list[float]) -> list[WeightedRange]:
    """Split ``[start, end)`` into contiguous slices proportional to ``weights``."""

    if end < start:
        raise ValueError("range end must be greater than or equal to range start")
    if not weights:
        return []

    length = end - start
    positive_weights = [max(0.0, float(weight)) for weight in weights]
    total_weight = sum(positive_weights)
    if total_weight <= 0.0:
        raise ValueError("at least one weight must be positive")

    raw_allocations = [length * weight / total_weight for weight in positive_weights]
    base_allocations = [int(value) for value in raw_allocations]
    fractional_parts = [value - int(value) for value in raw_allocations]
    assigned = sum(base_allocations)
    remainder = length - assigned

    if remainder > 0:
        for index in sorted(range(len(weights)), key=lambda item: fractional_parts[item], reverse=True)[:remainder]:
            base_allocations[index] += 1

    ranges: list[WeightedRange] = []
    current = start
    for index, allocation in enumerate(base_allocations):
        next_current = current + allocation
        ranges.append(
            WeightedRange(
                index=index,
                start=current,
                end=next_current,
                weight=positive_weights[index],
            )
        )
        current = next_current

    if ranges:
        ranges[-1] = WeightedRange(
            index=ranges[-1].index,
            start=ranges[-1].start,
            end=end,
            weight=ranges[-1].weight,
        )
    return ranges
