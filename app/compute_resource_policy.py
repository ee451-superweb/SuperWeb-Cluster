"""Shared compute-resource policy helpers for compute-intensive backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

from app.constants import (
    DEFAULT_COMPUTE_NODE_CPU_WORKER_FRACTION,
    DEFAULT_CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_SCALE,
)


@dataclass(frozen=True, slots=True)
class MetalHeadroomPolicy:
    """Describe one translated Metal runtime-throttling policy."""

    headroom_fraction: float
    work_chunk_size: int
    cooldown_ratio: float


def resolve_capped_cpu_worker_count(
    logical_cpu_count: int | None = None,
    *,
    fraction: float = DEFAULT_COMPUTE_NODE_CPU_WORKER_FRACTION,
) -> int:
    """Return the project-wide CPU worker cap for compute-heavy tasks.

    The default policy intentionally leaves some logical CPU headroom for the
    operating system and background services so compute-node execution does not
    completely starve the machine.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("CPU worker fraction must be within (0.0, 1.0].")

    resolved_cpu_count = os.cpu_count() if logical_cpu_count is None else logical_cpu_count
    resolved_cpu_count = max(1, int(resolved_cpu_count or 1))
    capped_workers = math.floor(resolved_cpu_count * fraction)
    return max(1, min(resolved_cpu_count, capped_workers))


def resolve_metal_headroom_policy(
    total_work_units: int,
    *,
    fraction: float = DEFAULT_COMPUTE_NODE_CPU_WORKER_FRACTION,
) -> MetalHeadroomPolicy:
    """Translate one shared headroom fraction into Metal-friendly throttling knobs.

    Metal does not expose a portable "use only N% of the GPU" API. We therefore
    translate the shared headroom target into:

    - a chunk size that inserts scheduling boundaries inside a long job
    - a cooldown ratio that expands each chunk's wall-clock duration into the
      requested average duty cycle
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("Metal headroom fraction must be within (0.0, 1.0].")

    resolved_total_units = max(1, int(total_work_units))
    if fraction >= 1.0 or resolved_total_units == 1:
        return MetalHeadroomPolicy(
            headroom_fraction=float(fraction),
            work_chunk_size=resolved_total_units,
            cooldown_ratio=0.0,
        )

    translated_chunk_size = math.floor(resolved_total_units * fraction)
    translated_chunk_size = max(1, min(resolved_total_units - 1, translated_chunk_size))
    cooldown_ratio = max(0.0, (1.0 / fraction) - 1.0)
    return MetalHeadroomPolicy(
        headroom_fraction=float(fraction),
        work_chunk_size=translated_chunk_size,
        cooldown_ratio=cooldown_ratio,
    )


def build_conv2d_cuda_output_channel_batch_candidates(output_channels: int) -> list[int]:
    """Return a short power-of-two sweep for Conv2D output-channel batching."""

    resolved_output_channels = max(1, int(output_channels))
    candidates: list[int] = []
    candidate = 1
    while candidate < resolved_output_channels:
        candidates.append(candidate)
        candidate *= 2
    if resolved_output_channels not in candidates:
        candidates.append(resolved_output_channels)
    return candidates


def resolve_scaled_conv2d_cuda_output_channel_batch(
    autotuned_batch: int | None,
    available_output_channels: int,
    *,
    scale: float = DEFAULT_CONV2D_CUDA_OUTPUT_CHANNEL_BATCH_SCALE,
) -> int:
    """Scale one autotuned CUDA batch size into a runtime-ready task batch size."""

    if scale <= 0.0:
        raise ValueError("Spatial CUDA output-channel batch scale must be positive.")

    resolved_output_channels = max(1, int(available_output_channels))
    resolved_autotuned_batch = int(autotuned_batch or 0)
    if resolved_autotuned_batch <= 0:
        resolved_autotuned_batch = resolved_output_channels

    scaled_batch = math.floor(resolved_autotuned_batch * scale)
    return max(1, min(resolved_output_channels, scaled_batch))
