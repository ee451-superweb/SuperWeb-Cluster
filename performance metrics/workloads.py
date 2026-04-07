"""Preset workloads chosen to keep the full benchmark under five minutes."""

from __future__ import annotations

from models import WorkloadSpec

PRESET_WORKLOADS: dict[str, tuple[int, int, float, float]] = {
    # rows, cols, ideal_seconds, zero_score_seconds
    "smoke": (128, 384, 0.02, 0.75),
    "quick": (512, 1024, 0.05, 2.0),
    "standard": (1024, 4096, 0.10, 6.0),
    "extended": (2048, 4096, 0.20, 12.0),
}


def resolve_workload(
    preset: str = "standard",
    *,
    rows: int | None = None,
    cols: int | None = None,
) -> WorkloadSpec:
    """Return a benchmark workload from a preset or explicit shape."""

    if rows is not None or cols is not None:
        if rows is None or cols is None:
            raise ValueError("rows and cols must be provided together")
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        name = f"custom-{rows}x{cols}"
        # For custom workloads, keep the same linear score window shape but
        # scale it with operation size so large matrices are not unfairly
        # penalized.
        flops = 2 * rows * cols
        ideal_seconds = max(0.05, flops / 1.0e9)
        zero_score_seconds = max(3.0, flops / 5.0e7)
        return WorkloadSpec(
            name=name,
            preset="custom",
            rows=rows,
            cols=cols,
            ideal_seconds=ideal_seconds,
            zero_score_seconds=zero_score_seconds,
        )

    if preset not in PRESET_WORKLOADS:
        known = ", ".join(sorted(PRESET_WORKLOADS))
        raise ValueError(f"unknown preset {preset!r}; expected one of: {known}")

    preset_rows, preset_cols, ideal_seconds, zero_score_seconds = PRESET_WORKLOADS[preset]
    return WorkloadSpec(
        name=f"{preset}-{preset_rows}x{preset_cols}",
        preset=preset,
        rows=preset_rows,
        cols=preset_cols,
        ideal_seconds=ideal_seconds,
        zero_score_seconds=zero_score_seconds,
    )
