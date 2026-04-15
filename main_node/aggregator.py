"""Task-result aggregation helpers for the main-node runtime."""

from __future__ import annotations

import array

from wire.runtime import TaskResult


class ResultAggregator:
    """Validate worker task slices and stitch them back into one result buffer."""

    def collect_fixed_matrix_vector_result(self, *, rows: int, results: list[TaskResult]) -> bytes:
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
        ordered_results = sorted(results, key=lambda item: item.start_oc)
        spatial_size = out_h * out_w
        merged = array.array("f", [0.0] * (spatial_size * total_cout))
        next_oc = 0

        for result in ordered_results:
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
            if result.output_length != expected_length:
                raise ValueError(
                    f"task result output length mismatch for oc {result.start_oc}:{result.end_oc} "
                    f"(declared {result.output_length}, expected {expected_length})"
                )
            if len(result.output_vector) != expected_length * 4:
                raise ValueError(
                    f"task result byte length mismatch for oc {result.start_oc}:{result.end_oc}"
                )

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
