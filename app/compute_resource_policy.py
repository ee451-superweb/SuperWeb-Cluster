"""Shared compute-resource policy helpers for compute-intensive backends."""

from __future__ import annotations


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
