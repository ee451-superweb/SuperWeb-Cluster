"""Shared progress reporters for input-dataset generation CLIs."""

from __future__ import annotations

import sys
from collections.abc import Callable

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


def _format_binary_size(num_bytes: int) -> str:
    """Format a byte count with a human-readable binary unit suffix."""
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
    milestone_progress: dict[str, int] = {}
    supports_live_tqdm = tqdm is not None and bool(getattr(sys.stdout, "isatty", lambda: False)())

    def report(label: str, written_bytes: int, total_bytes: int) -> None:
        """Update or print progress for one labeled generation stream."""
        if not supports_live_tqdm:
            if total_bytes <= 0:
                return
            previous_milestone = milestone_progress.get(label, -10)
            current_percent = int((written_bytes / total_bytes) * 100)
            current_milestone = min(100, (current_percent // 10) * 10)
            if written_bytes >= total_bytes:
                current_milestone = 100
            if current_milestone > previous_milestone:
                print(
                    f"{label}: {current_milestone}% "
                    f"({_format_binary_size(written_bytes)} / {_format_binary_size(total_bytes)})",
                    flush=True,
                )
                if current_milestone >= 100:
                    milestone_progress.pop(label, None)
                else:
                    milestone_progress[label] = current_milestone
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
        """Close any open progress bars created by this reporter pair."""
        for bar in list(bars.values()):
            bar.close()
        bars.clear()
        milestone_progress.clear()

    return report, close


def emit_progress_message(message: str) -> None:
    """Write one dataset-generation status line without corrupting tqdm bars."""

    if tqdm is None:
        print(message, flush=True)
        return
    tqdm.write(message, file=sys.stdout)


__all__ = ["build_progress_reporter", "emit_progress_message"]
