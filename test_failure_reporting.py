import io
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import main as app
from patterns import generate_cross, generate_x


class TestFailureReporting(unittest.TestCase):
    def setUp(self):
        self.cross = generate_cross(5)
        self.filters = {"size_5": {"cross": self.cross, "x": generate_x(5)}}
        self.case = {"input": deepcopy(self.cross), "expected": "+"}

    def data(self, cases):
        return {"filters": deepcopy(self.filters), "patterns": cases}

    def test_pass_has_no_failure_type(self):
        result = app.analyze_case("size_5_1", self.case, self.filters)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["failure_type"])

    def test_tie_tag(self):
        self.case["input"] = [[0.0] * 5 for _ in range(5)]
        result = app.analyze_case("size_5_1", self.case, self.filters)
        self.assertEqual(result["prediction"], "UNDECIDED")
        self.assertEqual(result["failure_type"], "tie")
        self.assertFalse(result["passed"])

    def test_prediction_mismatch_tag(self):
        self.case["expected"] = "x"
        result = app.analyze_case("size_5_1", self.case, self.filters)
        self.assertEqual(result["failure_type"], "prediction_mismatch")
        self.assertEqual(result["prediction"], "Cross")

    def test_schema_error_keeps_analysis_running(self):
        data = self.data({"bad_key": deepcopy(self.case), "size_5_2": self.case})
        results = app.analyze_data(data)
        self.assertEqual(results[0]["failure_type"], "data_schema")
        self.assertTrue(results[1]["passed"])
        self.assertEqual(len(results), 2)

    def test_numeric_errors_are_not_schema_errors(self):
        for error in (app.CalculationError("overflow"), OverflowError("overflow")):
            with self.subTest(error=type(error).__name__):
                with patch.object(app, "calculate_score", side_effect=error):
                    result = app.analyze_data(self.data({"size_5_1": self.case}))[0]
                self.assertEqual(result["failure_type"], "numeric")
                self.assertIn("수치 계산 오류", result["reason"])
        huge = {"input": [[1e308] * 5 for _ in range(5)], "expected": "+"}
        result = app.analyze_data(self.data({"size_5_1": huge}))[0]
        self.assertEqual(result["failure_type"], "numeric")

    def test_mixed_summary_invariants(self):
        tie = {"input": [[0.0] * 5 for _ in range(5)], "expected": "+"}
        mismatch = deepcopy(self.case)
        mismatch["expected"] = "x"
        huge = {"input": [[1e308] * 5 for _ in range(5)], "expected": "+"}
        data = self.data({"size_5_1": self.case, "size_5_2": tie,
                          "size_5_3": mismatch, "bad_key": self.case,
                          "size_5_4": huge})
        summary = app.summarize_results(app.analyze_data(data))
        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 4)
        self.assertEqual(summary["undecided"], 1)
        self.assertEqual(summary["failure_counts"],
                         {"tie": 1, "prediction_mismatch": 1, "data_schema": 1, "numeric": 1})
        self.assertEqual(summary["total"], summary["passed"] + summary["failed"])
        self.assertEqual(summary["failed"], sum(summary["failure_counts"].values()))

    def test_empty_all_pass_and_invalid_tag(self):
        for results in ([], app.analyze_data(self.data({"size_5_1": self.case}))):
            with self.subTest(total=len(results)):
                summary = app.summarize_results(results)
                self.assertEqual(summary["total"], len(results))
                self.assertEqual(summary["passed"], len(results))
                self.assertEqual(summary["failed"], 0)
                self.assertEqual(summary["undecided"], 0)
        for result in ({"passed": False, "failure_type": "unknown"},
                       {"passed": True, "failure_type": "tie"}):
            with self.assertRaises(ValueError):
                app.summarize_results([result])

    def test_summary_uses_tags_not_reason_strings(self):
        self.case["input"] = [[0.0] * 5 for _ in range(5)]
        results = app.analyze_data(self.data({"size_5_1": self.case}))
        expected = app.summarize_results(results)
        results[0]["reason"] = "데이터/스키마 오류라는 글자가 있어도 유형은 동점"
        before = deepcopy(results)
        self.assertEqual(app.summarize_results(results), expected)
        self.assertEqual(results, before)

    def test_original_json_summary_output(self):
        data = app.load_data(Path(app.__file__).resolve().parent / "data.json")
        with patch.object(app, "load_data", return_value=data):
            with patch.object(app, "run_json_performance"):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    app.run_json_mode()
        text = output.getvalue()
        for expected in ("총 테스트: 6개", "통과: 3개", "실패: 3개",
                         "동점은 실패에 포함", "동점 판정(UNDECIDED): 3개",
                         "예측 불일치: 0개", "데이터/스키마 오류: 0개", "수치 계산 오류: 0개"):
            self.assertIn(expected, text)
        summary = app.summarize_results(app.analyze_data(data))
        self.assertEqual(summary["failure_counts"]["tie"], 3)
        self.assertEqual(summary["failed"], 3)

    def test_manual_epsilon_hint_for_winner_and_tie(self):
        cross = ["0 1 0", "1 1 1", "0 1 0"]
        x = ["1 0 1", "0 1 0", "1 0 1"]
        zero = ["0 0 0"] * 3
        for pattern, expected in ((x, "판정: B"), (zero, "판정: 판정 불가")):
            with self.subTest(expected=expected):
                with patch("builtins.input", side_effect=cross + x + pattern):
                    with patch.object(app, "benchmark_mac", return_value={}):
                        with patch.object(app, "print_performance_table"):
                            with patch("sys.stdout", new_callable=io.StringIO) as output:
                                app.run_manual_mode()
                text = output.getvalue()
                self.assertIn(expected, text)
                self.assertEqual(text.count("판정 기준:"), 1)
                self.assertIn("두 점수 차이의 절댓값이 1e-09 미만이면", text)


if __name__ == "__main__":
    unittest.main()
