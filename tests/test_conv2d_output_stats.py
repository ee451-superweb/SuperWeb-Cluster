"""Tests for conv2d merged-output statistics (STATS_ONLY path)."""

import array
import tempfile
import unittest
from pathlib import Path

from app.constants import CONV2D_STATS_MAX_SAMPLES
from main_node.aggregator import summarize_conv2d_output_file


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


if __name__ == "__main__":
    unittest.main()
