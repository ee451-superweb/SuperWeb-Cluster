"""Validate task slices and stitch worker outputs into one final result.

Use this module when the main node has collected per-worker ``TaskResult``
objects and needs to merge them back into one GEMV or conv2d
response buffer.
"""

from __future__ import annotations

import array
from pathlib import Path

from app.constants import CONV2D_STATS_MAX_SAMPLES
from wire.internal_protocol.runtime_transport import TaskResult


def summarize_conv2d_output_file(
    path: Path,
    *,
    max_samples: int = CONV2D_STATS_MAX_SAMPLES,
) -> tuple[int, float, float, tuple[float, ...]]:
    """Scan a merged float32 conv2d output file and return count, sum, sum of squares, and leading samples.

    File layout matches ``collect_conv2d_result_to_file`` (little-endian float32, row-major flattened).
    """
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"conv2d output file size is not a multiple of 4: {size}")
    element_count = size // 4
    sum_v = 0.0
    sum_sq = 0.0
    samples: list[float] = []
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(256 * 1024)
            if not chunk:
                break
            values = array.array("f")
            values.frombytes(chunk)
            for x in values:
                xf = float(x)
                sum_v += xf
                sum_sq += xf * xf
                if len(samples) < max_samples:
                    samples.append(xf)
    return element_count, sum_v, sum_sq, tuple(samples)


class ResultAggregator:
    """Merge validated worker slice results into one response payload."""

    def _validate_conv2d_result(
        self,
        *,
        result: TaskResult,
        out_h: int,
        out_w: int,
        spatial_size: int,
    ) -> int:
        """Validate one conv2d result header and return its channel count.

        Use this before merging a conv2d result so both inline-byte and
        file-backed result paths share the same dimensionality checks.

        Args:
            result: Worker task result covering one output-channel slice.
            out_h: Expected output tensor height.
            out_w: Expected output tensor width.
            spatial_size: Precomputed output pixel count.

        Returns:
            The number of output channels covered by this result slice.
        """
        if result.end_oc < result.start_oc:
            raise ValueError("task result end_oc is smaller than start_oc")
        if result.output_h != out_h or result.output_w != out_w:
            raise ValueError("task result output dimensions do not match the requested conv2d workload")

        expected_channels = result.end_oc - result.start_oc
        expected_length = spatial_size * expected_channels
        if result.output_length != expected_length:
            raise ValueError(
                f"task result output length mismatch for oc {result.start_oc}:{result.end_oc} "
                f"(declared {result.output_length}, expected {expected_length})"
            )
        if result.local_result_path:
            result_path = Path(result.local_result_path)
            if not result_path.exists():
                raise FileNotFoundError(f"task result local file is missing: {result_path}")
            if result_path.stat().st_size != expected_length * 4:
                raise ValueError(
                    f"task result file length mismatch for oc {result.start_oc}:{result.end_oc}"
                )
        elif len(result.output_vector) != expected_length * 4:
            raise ValueError(
                f"task result byte length mismatch for oc {result.start_oc}:{result.end_oc}"
            )
        return expected_channels

    def collect_gemv_result(self, *, rows: int, results: list[TaskResult]) -> bytes:
        """Merge row-partitioned GEMV task results into one vector buffer.

        Use this after GEMV worker slices finish so the final response preserves
        row order and validates that all rows were covered exactly once.

        Args:
            rows: Total number of rows expected in the final output vector.
            results: Per-worker task results covering row slices.

        Returns:
            The merged output vector bytes in row order.
        """
        ordered_results = sorted(results, key=lambda item: item.row_start)
        merged = bytearray()
        next_row = 0

        for result in ordered_results:
            if result.row_start != next_row:
                raise ValueError(
                    f"task results do not cover a contiguous row range: expected {next_row}, got {result.row_start}"
                )
            if result.row_end < result.row_start:
                raise ValueError("task result row_end is smaller than row_start")

            expected_length = result.row_end - result.row_start
            if result.output_length != expected_length:
                raise ValueError(
                    f"task result output length mismatch for rows {result.row_start}:{result.row_end} "
                    f"(declared {result.output_length}, expected {expected_length})"
                )
            if len(result.output_vector) != expected_length * 4:
                raise ValueError(
                    f"task result byte length mismatch for rows {result.row_start}:{result.row_end}"
                )

            merged.extend(result.output_vector)
            next_row = result.row_end

        if next_row != rows:
            raise ValueError(f"task results cover only {next_row} rows out of {rows}")
        return bytes(merged)

    def collect_conv2d_result(
        self,
        *,
        out_h: int,
        out_w: int,
        total_cout: int,
        results: list[TaskResult],
    ) -> bytes:
        """Merge channel-partitioned conv2d results into one output tensor.

        Use this when the caller wants the final conv2d output in
        memory and needs each worker slice copied into its channel range.

        Args:
            out_h: Output tensor height.
            out_w: Output tensor width.
            total_cout: Total output-channel count expected in the final tensor.
            results: Per-worker task results covering channel slices.

        Returns:
            The merged output tensor as raw float32 bytes.
        """
        ordered_results = sorted(results, key=lambda item: item.start_oc)
        spatial_size = out_h * out_w
        merged = array.array("f", [0.0] * (spatial_size * total_cout))
        next_oc = 0

        for result in ordered_results:
            if result.start_oc != next_oc:
                raise ValueError(
                    f"task results do not cover a contiguous output-channel range: expected {next_oc}, got {result.start_oc}"
                )
            expected_channels = self._validate_conv2d_result(
                result=result,
                out_h=out_h,
                out_w=out_w,
                spatial_size=spatial_size,
            )
            if result.local_result_path:
                with Path(result.local_result_path).open("rb") as handle:
                    per_pixel_bytes = expected_channels * 4
                    for pixel in range(spatial_size):
                        payload = handle.read(per_pixel_bytes)
                        if len(payload) != per_pixel_bytes:
                            raise ValueError(
                                f"task result file ended early for oc {result.start_oc}:{result.end_oc}"
                            )
                        slice_values = array.array("f")
                        slice_values.frombytes(payload)
                        dst_base = pixel * total_cout + result.start_oc
                        merged[dst_base:dst_base + expected_channels] = slice_values
            else:
                slice_values = array.array("f")
                slice_values.frombytes(result.output_vector)
                for pixel in range(spatial_size):
                    src_base = pixel * expected_channels
                    dst_base = pixel * total_cout + result.start_oc
                    merged[dst_base:dst_base + expected_channels] = slice_values[src_base:src_base + expected_channels]

            next_oc = result.end_oc

        if next_oc != total_cout:
            raise ValueError(f"task results cover only {next_oc} output channels out of {total_cout}")
        return merged.tobytes()

    def aggregate_conv2d_stats(
        self,
        *,
        results: list[TaskResult],
        total_cout: int,
        out_h: int,
        out_w: int,
        max_samples: int = CONV2D_STATS_MAX_SAMPLES,
    ) -> tuple[int, float, float, tuple[float, ...]]:
        """Combine worker-reported conv2d stats into one summary.

        Use this when worker slices ran in STATS_ONLY mode: each result already
        carries running count/sum/sum-of-squares plus its first-N float samples.
        Sums are additive; samples are concatenated in start_oc order so the
        leading-N invariant matches a full-tensor scan that visited channels in
        the same order.
        """
        spatial_size = out_h * out_w
        ordered_results = sorted(results, key=lambda item: item.start_oc)
        total_count = 0
        total_sum = 0.0
        total_sum_squares = 0.0
        merged_samples: list[float] = []
        next_oc = 0
        sample_cap = max(0, int(max_samples))
        for result in ordered_results:
            payload = result.conv2d_payload
            if payload is None:
                raise ValueError("conv2d stats aggregation requires Conv2dResultPayload entries")
            if result.start_oc != next_oc:
                raise ValueError(
                    f"task results do not cover a contiguous output-channel range: expected {next_oc}, got {result.start_oc}"
                )
            if result.end_oc < result.start_oc:
                raise ValueError("task result end_oc is smaller than start_oc")
            if result.output_h != out_h or result.output_w != out_w:
                raise ValueError("task result output dimensions do not match the requested conv2d workload")
            expected_channels = result.end_oc - result.start_oc
            expected_length = spatial_size * expected_channels
            if int(payload.stats_element_count) != expected_length:
                raise ValueError(
                    f"task result stats_element_count mismatch for oc {result.start_oc}:{result.end_oc} "
                    f"(declared {payload.stats_element_count}, expected {expected_length})"
                )
            total_count += int(payload.stats_element_count)
            total_sum += float(payload.stats_sum)
            total_sum_squares += float(payload.stats_sum_squares)
            if len(merged_samples) < sample_cap:
                remaining = sample_cap - len(merged_samples)
                merged_samples.extend(float(value) for value in payload.stats_samples[:remaining])
            next_oc = result.end_oc
        if next_oc != total_cout:
            raise ValueError(f"task results cover only {next_oc} output channels out of {total_cout}")
        return total_count, total_sum, total_sum_squares, tuple(merged_samples)

    def collect_conv2d_result_to_file(
        self,
        *,
        out_h: int,
        out_w: int,
        total_cout: int,
        results,
        output_path: Path,
    ) -> Path:
        """Stream channel-partitioned conv2d results into a destination file.

        Use this when the main node wants to avoid holding the merged conv2d
        output fully in memory and can instead write directly to disk.

        Args:
            out_h: Output tensor height.
            out_w: Output tensor width.
            total_cout: Total output-channel count expected in the final tensor.
            results: Per-worker task results covering channel slices.
            output_path: File path that should receive the merged tensor bytes.

        Returns:
            The same output path after the merged tensor has been written.
        """
        spatial_size = out_h * out_w
        bytes_per_float = 4
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seen_ranges: list[tuple[int, int]] = []

        with output_path.open("w+b") as handle:
            handle.truncate(spatial_size * total_cout * bytes_per_float)
            for result in results:
                expected_channels = self._validate_conv2d_result(
                    result=result,
                    out_h=out_h,
                    out_w=out_w,
                    spatial_size=spatial_size,
                )
                seen_ranges.append((result.start_oc, result.end_oc))
                per_pixel_bytes = expected_channels * bytes_per_float
                if result.local_result_path:
                    with Path(result.local_result_path).open("rb") as source:
                        for pixel in range(spatial_size):
                            payload = source.read(per_pixel_bytes)
                            if len(payload) != per_pixel_bytes:
                                raise ValueError(
                                    f"task result file ended early for oc {result.start_oc}:{result.end_oc}"
                                )
                            dst_offset = ((pixel * total_cout) + result.start_oc) * bytes_per_float
                            handle.seek(dst_offset)
                            handle.write(payload)
                else:
                    src = memoryview(result.output_vector)
                    for pixel in range(spatial_size):
                        src_start = pixel * per_pixel_bytes
                        src_end = src_start + per_pixel_bytes
                        dst_offset = ((pixel * total_cout) + result.start_oc) * bytes_per_float
                        handle.seek(dst_offset)
                        handle.write(src[src_start:src_end])

        next_oc = 0
        for start_oc, end_oc in sorted(seen_ranges):
            if start_oc != next_oc:
                raise ValueError(
                    f"task results do not cover a contiguous output-channel range: expected {next_oc}, got {start_oc}"
                )
            next_oc = end_oc
        if next_oc != total_cout:
            raise ValueError(f"task results cover only {next_oc} output channels out of {total_cout}")
        return output_path
