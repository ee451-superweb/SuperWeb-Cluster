"""Tests for conv2d merged-output statistics (STATS_ONLY path)."""

import array
import tempfile
import unittest
from pathlib import Path

from core.constants import CONV2D_STATS_MAX_SAMPLES, STATUS_OK
from main_node.aggregator import ResultAggregator, summarize_conv2d_output_file
from wire.internal_protocol.control_plane import Conv2dResultPayload
from wire.internal_protocol.transport import TaskResult


class Conv2dOutputStatsTests(unittest.TestCase):
    def test_summarize_matches_golden(self) -> None:
        values = [1.0, 2.0, -0.5, 3.25]
        raw = array.array("f", values).tobytes()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            path = Path(tmp.name)
            tmp.write(raw)
        try:
            count, sum_v, sum_sq, samples = summarize_conv2d_output_file(path, max_samples=16)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(count, 4)
        self.assertAlmostEqual(sum_v, 5.75)
        expected_sq = sum(x * x for x in values)
        self.assertAlmostEqual(sum_sq, expected_sq)
        self.assertEqual(samples, tuple(values))

    def test_summarize_caps_samples(self) -> None:
        values = list(range(100))
        raw = array.array("f", [float(x) for x in values]).tobytes()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            path = Path(tmp.name)
            tmp.write(raw)
        try:
            count, sum_v, _, samples = summarize_conv2d_output_file(
                path, max_samples=CONV2D_STATS_MAX_SAMPLES
            )
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(count, 100)
        self.assertAlmostEqual(sum_v, sum(values))
        self.assertEqual(len(samples), CONV2D_STATS_MAX_SAMPLES)
        self.assertEqual(samples, tuple(float(x) for x in range(CONV2D_STATS_MAX_SAMPLES)))


class Conv2dStatsAggregationTests(unittest.TestCase):
    def _make_result(self, *, start_oc: int, end_oc: int, out_h: int, out_w: int, values: list[float]) -> TaskResult:
        expected = (end_oc - start_oc) * out_h * out_w
        if len(values) < expected:
            raise AssertionError("test helper got fewer values than slice covers")
        sum_v = sum(values)
        sum_sq = sum(x * x for x in values)
        payload = Conv2dResultPayload(
            start_oc=start_oc,
            end_oc=end_oc,
            output_h=out_h,
            output_w=out_w,
            output_length=expected,
            output_vector=b"",
            result_artifact_id="",
            stats_element_count=expected,
            stats_sum=sum_v,
            stats_sum_squares=sum_sq,
            stats_samples=tuple(values[:4]),
        )
        return TaskResult(
            request_id="req-stats",
            node_id="worker-a",
            task_id="req-stats:worker-a:0",
            timestamp_ms=1,
            status_code=STATUS_OK,
            iteration_count=1,
            result_payload=payload,
        )

    def test_aggregate_sums_counts_sum_and_sum_squares(self) -> None:
        aggregator = ResultAggregator()
        slice_a = self._make_result(start_oc=0, end_oc=1, out_h=2, out_w=2, values=[1.0, 2.0, 3.0, 4.0])
        slice_b = self._make_result(start_oc=1, end_oc=2, out_h=2, out_w=2, values=[5.0, 6.0, 7.0, 8.0])
        count, sum_v, sum_sq, samples = aggregator.aggregate_conv2d_stats(
            results=[slice_b, slice_a],
            total_cout=2,
            out_h=2,
            out_w=2,
            max_samples=6,
        )
        self.assertEqual(count, 8)
        self.assertAlmostEqual(sum_v, 36.0)
        self.assertAlmostEqual(sum_sq, sum(x * x for x in range(1, 9)))
        self.assertEqual(samples, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_aggregate_rejects_gaps(self) -> None:
        aggregator = ResultAggregator()
        slice_a = self._make_result(start_oc=0, end_oc=1, out_h=2, out_w=2, values=[1.0, 2.0, 3.0, 4.0])
        slice_gap = self._make_result(start_oc=2, end_oc=3, out_h=2, out_w=2, values=[5.0, 6.0, 7.0, 8.0])
        with self.assertRaises(ValueError):
            aggregator.aggregate_conv2d_stats(
                results=[slice_a, slice_gap],
                total_cout=3,
                out_h=2,
                out_w=2,
                max_samples=4,
            )


if __name__ == "__main__":
    unittest.main()
