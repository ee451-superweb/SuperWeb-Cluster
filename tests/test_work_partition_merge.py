"""Tests for common.work_partition.merge_ranges."""

import unittest

from core.work_partition import merge_ranges


class MergeRangesTests(unittest.TestCase):
    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(merge_ranges([]), [])

    def test_single_range_passes_through(self) -> None:
        self.assertEqual(merge_ranges([(10, 20)]), [(10, 20)])

    def test_adjacent_ranges_are_merged(self) -> None:
        self.assertEqual(merge_ranges([(0, 10), (10, 25)]), [(0, 25)])

    def test_overlapping_ranges_are_merged(self) -> None:
        self.assertEqual(merge_ranges([(0, 15), (10, 25)]), [(0, 25)])

    def test_nested_range_is_absorbed(self) -> None:
        self.assertEqual(merge_ranges([(0, 50), (10, 20)]), [(0, 50)])

    def test_non_adjacent_ranges_are_kept_separate(self) -> None:
        self.assertEqual(merge_ranges([(0, 10), (20, 30)]), [(0, 10), (20, 30)])

    def test_unsorted_input_is_sorted_before_merging(self) -> None:
        self.assertEqual(
            merge_ranges([(20, 30), (0, 10), (10, 15)]),
            [(0, 15), (20, 30)],
        )

    def test_mixed_merge_and_gap(self) -> None:
        self.assertEqual(
            merge_ranges([(0, 5), (5, 10), (20, 25), (25, 30), (40, 50)]),
            [(0, 10), (20, 30), (40, 50)],
        )

    def test_zero_length_range_is_dropped(self) -> None:
        self.assertEqual(merge_ranges([(5, 5), (10, 20)]), [(10, 20)])

    def test_reversed_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            merge_ranges([(10, 5)])


if __name__ == "__main__":
    unittest.main()
