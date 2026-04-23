"""Aggregator tests focused on GEMM M-axis stitching."""

from __future__ import annotations

import array
import unittest

from core.constants import METHOD_GEMM, STATUS_OK
from main_node.aggregator import ResultAggregator
from wire.internal_protocol.control_plane import GemmResultPayload
from wire.internal_protocol.transport import TaskResult


def _float_bytes(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def _build_result(
    *,
    m_start: int,
    m_end: int,
    n: int,
    values: list[float],
    task_id: str = "req:w",
) -> TaskResult:
    return TaskResult(
        request_id="req",
        node_id="worker",
        task_id=task_id,
        timestamp_ms=0,
        status_code=STATUS_OK,
        iteration_count=1,
        result_payload=GemmResultPayload(
            m_start=m_start,
            m_end=m_end,
            output_length=(m_end - m_start) * n,
            output_vector=_float_bytes(values),
        ),
    )


class AggregatorGemmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregator = ResultAggregator()

    def test_collect_gemm_result_concatenates_rows_in_m_order(self) -> None:
        n = 3
        slice_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        slice_b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        result_a = _build_result(m_start=0, m_end=2, n=n, values=slice_a, task_id="req:a")
        result_b = _build_result(m_start=2, m_end=4, n=n, values=slice_b, task_id="req:b")

        # Feed in reverse order to confirm the aggregator reorders by m_start.
        merged = self.aggregator.collect_gemm_result(
            m=4,
            n=n,
            results=[result_b, result_a],
        )

        merged_values = list(array.array("f", merged))
        self.assertEqual(merged_values, slice_a + slice_b)
        self.assertEqual(len(merged), 4 * n * 4)

    def test_collect_gemm_result_rejects_non_contiguous_coverage(self) -> None:
        n = 2
        result_a = _build_result(m_start=0, m_end=1, n=n, values=[1.0, 2.0])
        result_c = _build_result(m_start=2, m_end=3, n=n, values=[5.0, 6.0])

        with self.assertRaises(ValueError):
            self.aggregator.collect_gemm_result(m=3, n=n, results=[result_a, result_c])

    def test_collect_gemm_result_rejects_incomplete_coverage(self) -> None:
        n = 2
        result_a = _build_result(m_start=0, m_end=2, n=n, values=[1.0, 2.0, 3.0, 4.0])

        with self.assertRaises(ValueError):
            self.aggregator.collect_gemm_result(m=4, n=n, results=[result_a])

    def test_collect_gemm_result_rejects_mismatched_output_length(self) -> None:
        n = 2
        # Declare output_length wrong even though the bytes are fine.
        result_a = TaskResult(
            request_id="req",
            node_id="worker",
            task_id="req:a",
            timestamp_ms=0,
            status_code=STATUS_OK,
            iteration_count=1,
            result_payload=GemmResultPayload(
                m_start=0,
                m_end=2,
                output_length=999,
                output_vector=_float_bytes([1.0, 2.0, 3.0, 4.0]),
            ),
        )

        with self.assertRaises(ValueError):
            self.aggregator.collect_gemm_result(m=2, n=n, results=[result_a])


if __name__ == "__main__":
    unittest.main()
