"""Validate task slices and stitch worker outputs into one final result.

Use this module when the main node has collected per-worker ``TaskResult``
objects and needs to merge them back into one FMVM or spatial-convolution
response buffer.
"""

from __future__ import annotations

import array
from pathlib import Path

from wire.internal_protocol.runtime_transport import TaskResult


class ResultAggregator:
    """Merge validated worker slice results into one response payload."""

    def _validate_spatial_result(
        self,
        *,
        result: TaskResult,
        out_h: int,
        out_w: int,
        spatial_size: int,
    ) -> int:
        """Validate one spatial result header and return its channel count.

        Use this before merging a spatial result so both inline-byte and
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

    def collect_fixed_matrix_vector_result(self, *, rows: int, results: list[TaskResult]) -> bytes:
        """Merge row-partitioned FMVM task results into one vector buffer.

        Use this after FMVM worker slices finish so the final response preserves
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

    def collect_spatial_convolution_result(
        self,
        *,
        out_h: int,
        out_w: int,
        total_cout: int,
        results: list[TaskResult],
    ) -> bytes:
        """Merge channel-partitioned conv2d results into one output tensor.

        Use this when the caller wants the final spatial-convolution output in
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
            expected_channels = self._validate_spatial_result(
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

    def collect_spatial_convolution_result_to_file(
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
                expected_channels = self._validate_spatial_result(
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
