"""Shared progress reporters for input-dataset generation CLIs."""

from __future__ import annotations

import sys
from collections.abc import Callable

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


def _format_binary_size(num_bytes: int) -> str:
    kib = 1024
    mib = kib * 1024
    gib = mib * 1024
    if num_bytes >= gib:
        return f"{num_bytes / gib:.3f} GiB"
    if num_bytes >= mib:
        return f"{num_bytes / mib:.3f} MiB"
    if num_bytes >= kib:
        return f"{num_bytes / kib:.3f} KiB"
    return f"{num_bytes} B"


def build_progress_reporter() -> tuple[Callable[[str, int, int], None], Callable[[], None]]:
    """Create one `(report, close)` pair for dataset-generation progress."""

    bars: dict[str, object] = {}

    def report(label: str, written_bytes: int, total_bytes: int) -> None:
        if tqdm is None:
            print(
                f"{label}: wrote {_format_binary_size(written_bytes)} / {_format_binary_size(total_bytes)}",
                flush=True,
            )
            return

        bar = bars.get(label)
        if bar is None:
            bar = tqdm(
                total=total_bytes,
                desc=label,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )
            bars[label] = bar

        delta = written_bytes - bar.n
        if delta > 0:
            bar.update(delta)
        if written_bytes >= total_bytes:
            bar.close()
            bars.pop(label, None)

    def close() -> None:
        for bar in list(bars.values()):
            bar.close()
        bars.clear()

    return report, close


__all__ = ["build_progress_reporter"]
