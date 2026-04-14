"""Task-result aggregation helpers for the main-node runtime."""

from __future__ import annotations

from wire.runtime import TaskResult


class ResultAggregator:
    """Validate worker task slices and stitch them back into one output vector."""

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

