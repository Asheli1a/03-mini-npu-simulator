import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import main as app
import compare_mac as comparison


class TestMacComparison(unittest.TestCase):
    def setUp(self):
        self.matrix = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]

    def test_result_structure(self):
        result = comparison.compare_mac(self.matrix, self.matrix, repeats=10, rounds=3)
        self.assertEqual(result["size"], 3)
        self.assertEqual(result["mac_count"], 9)
        self.assertEqual(result["score"], 5.0)
        self.assertEqual(set(result["stages"]),
                         {"2d", "1d_ready", "1d_with_conversion"})
        for stats in result["stages"].values():
            self.assertEqual(len(stats["samples_ms"]), 3)
            self.assertGreaterEqual(stats["average_ms"], 0.0)
            self.assertLessEqual(stats["min_ms"], stats["average_ms"])
            self.assertLessEqual(stats["average_ms"], stats["max_ms"])

    def test_invalid_repeats(self):
        for repeats in (True, 9, 10.0, "10"):
            with self.subTest(repeats=repeats):
                with self.assertRaises(ValueError):
                    comparison.compare_mac(self.matrix, self.matrix, repeats=repeats)

    def test_invalid_rounds(self):
        for rounds in (True, 0, 2, 4, 3.0):
            with self.subTest(rounds=rounds):
                with self.assertRaises(ValueError):
                    comparison.compare_mac(self.matrix, self.matrix, rounds=rounds)

    def test_invalid_matrix(self):
        with self.assertRaises(ValueError):
            comparison.compare_mac(self.matrix, [[1.0]])

    def test_overflow_rejected(self):
        with self.assertRaises(ValueError):
            comparison.compare_mac([[1e308]], [[1e308]])

    def test_inputs_unchanged(self):
        original = deepcopy(self.matrix)
        comparison.compare_mac(self.matrix, self.matrix, repeats=10, rounds=3)
        self.assertEqual(self.matrix, original)

    def test_unchecked_flatten_matches_public(self):
        self.assertEqual(comparison._flatten_unchecked(self.matrix),
                         app.flatten_matrix(self.matrix))

    def test_batch_average_unit_conversion(self):
        calls = []
        with patch.object(comparison, "perf_counter_ns", side_effect=[100, 10100]):
            average = comparison._batch_average_ms(lambda: calls.append(1), 10)
        self.assertEqual(len(calls), 10)
        self.assertEqual(average, 0.001)


if __name__ == "__main__":
    unittest.main()
