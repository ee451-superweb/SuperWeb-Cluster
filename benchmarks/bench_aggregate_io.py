"""Micro-benchmark for the conv2d aggregate I/O hot path.

Reproduces the three candidate implementations under matching sizes so we can
see whether the aggregator's 95 MB/s ceiling comes from disk, from strided
access, or from the per-pixel Python loop.

Scenarios:
    1. pure_copy: shutil.copyfileobj with a 4MB buffer (upper bound for a
       single-slice 1:1 file copy).
    2. memmap_strided: numpy memmap assignment simulating N worker slices
       each covering an output-channel range.
    3. per_pixel: the current aggregator pattern (source.read + seek + write
       per pixel), also with N worker slices.

Usage:
    python benchmarks/bench_aggregate_io.py
    python benchmarks/bench_aggregate_io.py --h 2048 --w 2048 --cout 256 --slices 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


def _fill_source(path: Path, nbytes: int) -> None:
    chunk = np.random.default_rng(0).integers(0, 255, size=64 * 1024 * 1024, dtype=np.uint8).tobytes()
    written = 0
    with path.open("wb") as handle:
        while written < nbytes:
            remaining = nbytes - written
            if remaining >= len(chunk):
                handle.write(chunk)
                written += len(chunk)
            else:
                handle.write(chunk[:remaining])
                written += remaining


def _fresh_dest(path: Path, nbytes: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(nbytes)


def _print_result(name: str, seconds: float, nbytes: int) -> None:
    mb = nbytes / (1024 * 1024)
    print(f"{name:24s} {seconds:7.3f}s  {mb / seconds:7.1f} MB/s  ({mb:.0f} MB)")


def run_pure_copy(src: Path, dst: Path, nbytes: int) -> float:
    _fresh_dest(dst, 0)
    started = time.perf_counter()
    with src.open("rb") as s, dst.open("wb") as d:
        shutil.copyfileobj(s, d, 4 * 1024 * 1024)
    return time.perf_counter() - started


def run_memmap_strided(
    src_files: list[Path],
    dst: Path,
    spatial: int,
    total_cout: int,
    slice_ranges: list[tuple[int, int]],
) -> float:
    nbytes = spatial * total_cout * 4
    _fresh_dest(dst, nbytes)
    started = time.perf_counter()
    dst_mm = np.memmap(dst, dtype=np.float32, mode="r+", shape=(spatial, total_cout))
    for src_path, (start_oc, end_oc) in zip(src_files, slice_ranges):
        slice_channels = end_oc - start_oc
        src_mm = np.memmap(src_path, dtype=np.float32, mode="r", shape=(spatial, slice_channels))
        dst_mm[:, start_oc:end_oc] = src_mm
    dst_mm.flush()
    del dst_mm
    return time.perf_counter() - started


def run_per_pixel(
    src_files: list[Path],
    dst: Path,
    spatial: int,
    total_cout: int,
    slice_ranges: list[tuple[int, int]],
) -> float:
    nbytes = spatial * total_cout * 4
    _fresh_dest(dst, nbytes)
    started = time.perf_counter()
    with dst.open("r+b") as handle:
        handle.truncate(nbytes)
        for src_path, (start_oc, end_oc) in zip(src_files, slice_ranges):
            per_pixel_bytes = (end_oc - start_oc) * 4
            with src_path.open("rb") as source:
                for pixel in range(spatial):
                    payload = source.read(per_pixel_bytes)
                    dst_offset = ((pixel * total_cout) + start_oc) * 4
                    handle.seek(dst_offset)
                    handle.write(payload)
    return time.perf_counter() - started


def build_slice_ranges(total_cout: int, slices: int) -> list[tuple[int, int]]:
    if total_cout % slices != 0:
        raise SystemExit(f"total_cout={total_cout} must divide by slices={slices}")
    step = total_cout // slices
    return [(i * step, (i + 1) * step) for i in range(slices)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate I/O micro-benchmark")
    parser.add_argument("--h", type=int, default=2048, help="Output tensor height")
    parser.add_argument("--w", type=int, default=2048, help="Output tensor width")
    parser.add_argument("--cout", type=int, default=256, help="Total output channels")
    parser.add_argument("--slices", type=int, default=1, help="Number of worker slices")
    parser.add_argument("--tmpdir", type=str, default=None, help="Override working directory")
    args = parser.parse_args()

    spatial = args.h * args.w
    total_bytes = spatial * args.cout * 4
    slice_ranges = build_slice_ranges(args.cout, args.slices)

    tmp = Path(tempfile.mkdtemp(prefix="aggregate-bench-", dir=args.tmpdir))
    try:
        print(
            f"H={args.h} W={args.w} cout={args.cout} slices={args.slices} "
            f"total_bytes={total_bytes} ({total_bytes / (1024 ** 3):.2f} GiB)"
        )
        print(f"working dir: {tmp}")

        src_files: list[Path] = []
        for idx, (start_oc, end_oc) in enumerate(slice_ranges):
            slice_channels = end_oc - start_oc
            nbytes = spatial * slice_channels * 4
            src_path = tmp / f"slice-{idx}.bin"
            _fill_source(src_path, nbytes)
            src_files.append(src_path)
        print(f"generated {len(src_files)} source file(s) on disk")

        dst = tmp / "dst.bin"

        if args.slices == 1:
            t = run_pure_copy(src_files[0], dst, total_bytes)
            _print_result("pure_copy", t, total_bytes)

        t = run_memmap_strided(src_files, dst, spatial, args.cout, slice_ranges)
        _print_result("memmap_strided", t, total_bytes)

        t = run_per_pixel(src_files, dst, spatial, args.cout, slice_ranges)
        _print_result("per_pixel_loop", t, total_bytes)

    finally:
        for path in tmp.iterdir():
            try:
                path.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
