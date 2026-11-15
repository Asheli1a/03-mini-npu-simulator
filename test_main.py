from copy import deepcopy
from pathlib import Path
import unittest

from main import (
    EPSILON,
    calculate_score,
    decide_ab,
    decide_label,
    normalize_label,
    parse_row,
    analyze_case,
    analyze_data,
    parse_case_size,
    benchmark_mac,
    load_data,
)


class TestMac(unittest.TestCase):
    def setUp(self):
        self.cross = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]

        self.x = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]

    def test_cross_with_cross(self):
        self.assertEqual(calculate_score(self.cross, self.cross), 5.0)

    def test_cross_with_x(self):
        self.assertEqual(calculate_score(self.cross, self.x), 1.0)

    def test_x_with_cross(self):
        self.assertEqual(calculate_score(self.x, self.cross), 1.0)

    def test_x_with_x(self):
        self.assertEqual(calculate_score(self.x, self.x), 5.0)

    def test_zero_pattern(self):
        zero_pattern = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        self.assertEqual(calculate_score(zero_pattern, self.cross), 0.0)

    def test_decimal_values(self):
        pattern = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        filter_matrix = [
            [1, 1],
            [1, 1],
        ]

        self.assertAlmostEqual(
            calculate_score(pattern, filter_matrix),
            1.0,
        )

    def test_size_mismatch(self):
        small_filter = [
            [1, 0],
            [0, 1],
        ]

        with self.assertRaisesRegex(ValueError, "크기 불일치"):
            calculate_score(self.cross, small_filter)

    def test_irregular_rows(self):
        invalid_pattern = [
            [0, 1, 0],
            [1, 1],
            [0, 1, 0],
        ]

        with self.assertRaises(ValueError):
            calculate_score(invalid_pattern, self.cross)

    def test_non_numeric_value(self):
        invalid_pattern = [
            [0, 1, 0],
            [1, "숫자 아님", 1],
            [0, 1, 0],
        ]

        with self.assertRaises(ValueError):
            calculate_score(invalid_pattern, self.cross)

    def test_empty_matrix(self):
        with self.assertRaises(ValueError):
            calculate_score([], [])

class TestNormalizeLabel(unittest.TestCase):
    def test_cross_labels(self):
        for label in ("+", "cross", "Cross", "CROSS"):
            with self.subTest(label=label):
                self.assertEqual(normalize_label(label), "Cross")

    def test_x_labels(self):
        for label in ("x", "X"):
            with self.subTest(label=label):
                self.assertEqual(normalize_label(label), "X")

    def test_surrounding_whitespace(self):
        self.assertEqual(normalize_label(" + "), "Cross")
        self.assertEqual(normalize_label(" x "), "X")

    def test_unknown_labels(self):
        for label in ("", "circle", "UNDECIDED"):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    normalize_label(label)

    def test_non_string_labels(self):
        for label in (None, 1, True):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    normalize_label(label)


class TestDecideLabel(unittest.TestCase):
    def test_cross_wins(self):
        self.assertEqual(decide_label(5.0, 1.0), "Cross")

    def test_x_wins(self):
        self.assertEqual(decide_label(1.0, 5.0), "X")

    def test_exact_tie(self):
        self.assertEqual(decide_label(1.0, 1.0), "UNDECIDED")

    def test_float_rounding_tie(self):
        self.assertEqual(
            decide_label(0.9, 0.8999999999999999),
            "UNDECIDED",
        )

    def test_difference_below_epsilon(self):
        self.assertEqual(
            decide_label(EPSILON / 2, 0.0),
            "UNDECIDED",
        )
        self.assertEqual(
            decide_label(0.0, EPSILON / 2),
            "UNDECIDED",
        )

    def test_difference_equal_to_epsilon(self):
        self.assertEqual(decide_label(EPSILON, 0.0), "Cross")
        self.assertEqual(decide_label(0.0, EPSILON), "X")

    def test_difference_above_epsilon(self):
        self.assertEqual(decide_label(EPSILON * 2, 0.0), "Cross")
        self.assertEqual(decide_label(0.0, EPSILON * 2), "X")

    def test_calculate_then_decide(self):
        cross = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        x = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]

        score_cross = calculate_score(x, cross)
        score_x = calculate_score(x, x)

        self.assertEqual(
            decide_label(score_cross, score_x),
            normalize_label("x"),
        )

class TestParseRow(unittest.TestCase):
    def test_valid_integers(self):
        self.assertEqual(parse_row("0 1 0", 3), [0.0, 1.0, 0.0])

    def test_decimals_and_negative_values(self):
        self.assertEqual(
            parse_row("-0.5 0.3 1.2", 3),
            [-0.5, 0.3, 1.2],
        )

    def test_extra_whitespace(self):
        self.assertEqual(
            parse_row("  0   1\t0  ", 3),
            [0.0, 1.0, 0.0],
        )

    def test_wrong_column_count(self):
        for text in ("0 1", "0 1 0 1"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_row(text, 3)

    def test_non_numeric_value(self):
        with self.assertRaises(ValueError):
            parse_row("0 hello 1", 3)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            parse_row("", 3)

    def test_non_finite_values(self):
        for text in ("nan 0 1", "inf 0 1", "-inf 0 1", "1e999 0 1"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_row(text, 3)


class TestDecideAB(unittest.TestCase):
    def test_a_wins(self):
        self.assertEqual(decide_ab(5.0, 1.0), "A")

    def test_b_wins(self):
        self.assertEqual(decide_ab(1.0, 5.0), "B")

    def test_tie(self):
        self.assertEqual(
            decide_ab(0.9, 0.8999999999999999),
            "UNDECIDED",
        )

class TestJsonAnalysis(unittest.TestCase):
    def setUp(self):
        size = 5

        cross = [
            [
                1.0 if row == size // 2 or column == size // 2 else 0.0
                for column in range(size)
            ]
            for row in range(size)
        ]

        x = [
            [
                1.0 if row == column or row + column == size - 1 else 0.0
                for column in range(size)
            ]
            for row in range(size)
        ]

        self.filters = {
            "size_5": {
                "cross": cross,
                "x": x,
            }
        }

        self.case = {
            "input": deepcopy(cross),
            "expected": "+",
        }

    def test_parse_case_size(self):
        self.assertEqual(parse_case_size("size_5_1"), 5)
        self.assertEqual(parse_case_size("size_13_2"), 13)
        self.assertEqual(parse_case_size("size_25_1"), 25)

    def test_invalid_case_key(self):
        for case_id in ("size_5", "size_7_1", "size_5_bad", "other_5_1"):
            with self.subTest(case_id=case_id):
                with self.assertRaises(ValueError):
                    parse_case_size(case_id)

    def test_valid_case_passes(self):
        result = analyze_case("size_5_1", self.case, self.filters)

        self.assertEqual(result["prediction"], "Cross")
        self.assertEqual(result["expected"], "Cross")
        self.assertTrue(result["passed"])

    def test_tie_is_fail(self):
        self.case["input"] = [[0.0] * 5 for _ in range(5)]

        result = analyze_case("size_5_1", self.case, self.filters)

        self.assertEqual(result["prediction"], "UNDECIDED")
        self.assertFalse(result["passed"])

    def test_pattern_size_mismatch(self):
        self.case["input"] = [[0.0] * 3 for _ in range(3)]

        with self.assertRaisesRegex(ValueError, "패턴 크기 불일치"):
            analyze_case("size_5_1", self.case, self.filters)

    def test_filter_size_mismatch(self):
        self.filters["size_5"]["x"] = [[0.0] * 3 for _ in range(3)]

        with self.assertRaisesRegex(ValueError, "필터 크기 불일치"):
            analyze_case("size_5_1", self.case, self.filters)

    def test_invalid_case_does_not_stop_analysis(self):
        data = {
            "filters": self.filters,
            "patterns": {
                "bad_key": deepcopy(self.case),
                "size_5_2": deepcopy(self.case),
            },
        }

        results = analyze_data(data)

        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]["passed"])
        self.assertIn("데이터/스키마 오류", results[0]["reason"])
        self.assertTrue(results[1]["passed"])

class TestBenchmarkMac(unittest.TestCase):
    def setUp(self):
        self.matrix = [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ]

    def test_result_fields(self):
        result = benchmark_mac(self.matrix, self.matrix, repeats=10)

        self.assertEqual(result["size"], 3)
        self.assertEqual(result["repeats"], 10)
        self.assertEqual(result["mac_count"], 9)
        self.assertGreaterEqual(result["average_ms"], 0.0)

    def test_too_few_repeats(self):
        with self.assertRaises(ValueError):
            benchmark_mac(self.matrix, self.matrix, repeats=9)

    def test_non_integer_repeats(self):
        for repeats in (10.5, True):
            with self.subTest(repeats=repeats):
                with self.assertRaises(ValueError):
                    benchmark_mac(
                        self.matrix,
                        self.matrix,
                        repeats=repeats,
                    )

    def test_size_mismatch(self):
        small_matrix = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        with self.assertRaises(ValueError):
            benchmark_mac(self.matrix, small_matrix, repeats=10)

class TestProvidedData(unittest.TestCase):
    def test_original_data_results(self):
        path = Path(__file__).resolve().parent / "data.json"
        data = load_data(path)
        results = analyze_data(data)

        actual = {
            result["case_id"]: (
                result["prediction"],
                result["expected"],
                result["passed"],
            )
            for result in results
        }

        expected = {
            "size_5_1": ("UNDECIDED", "X", False),
            "size_5_2": ("Cross", "Cross", True),
            "size_13_1": ("X", "X", True),
            "size_13_2": ("UNDECIDED", "Cross", False),
            "size_25_1": ("UNDECIDED", "X", False),
            "size_25_2": ("Cross", "Cross", True),
        }

        self.assertEqual(len(results), 6)
        self.assertEqual(actual, expected)
        self.assertEqual(
            sum(result["passed"] for result in results),
            3,
        )

if __name__ == "__main__":
    unittest.main()