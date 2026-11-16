import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as app


class TestAdditional(unittest.TestCase):
    def test_provided_data_results(self):
        path = Path(app.__file__).resolve().parent / "data.json"
        results = app.analyze_data(app.load_data(path))

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

    def test_non_finite_scores_rejected(self):
        cases = [
            (float("inf"), 1.0),
            (1.0, float("-inf")),
            (float("nan"), 1.0),
            (float("inf"), float("inf")),
        ]

        for scores in cases:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    app.decide_label(*scores)

    def test_float_overflow_rejected(self):
        with self.assertRaises(ValueError):
            app.calculate_score([[1e308]], [[1e308]])

    def test_large_integer_overflow_rejected(self):
        with self.assertRaises(ValueError):
            app.calculate_score([[10 ** 400]], [[1]])

    def test_read_matrix_retries_invalid_row(self):
        inputs = [
            "0 1",
            "0 hello 1",
            "0 1 0",
            "1 1 1",
            "0 1 0",
        ]

        with patch("builtins.input", side_effect=inputs) as mocked:
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                matrix = app.read_matrix("테스트")

        self.assertEqual(matrix, [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        self.assertEqual(mocked.call_count, 5)
        self.assertEqual(output.getvalue().count("입력 형식 오류"), 2)

    def test_manual_overflow_returns_to_menu(self):
        huge_row = "1e308 1e308 1e308"
        inputs = ["1"] + [huge_row] * 9 + ["0"]

        with patch("builtins.input", side_effect=inputs):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                app.main()

        text = output.getvalue()
        self.assertIn("계산 오류", text)
        self.assertIn("프로그램을 종료합니다.", text)

    def test_json_mode_file_errors_return_to_menu(self):
        cases = [
            ("missing", None),
            ("invalid_json", "{"),
            ("invalid_schema", '{"filters": [], "patterns": {}}'),
        ]

        for label, content in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)

                    if content is not None:
                        (root / "data.json").write_text(
                            content,
                            encoding="utf-8",
                        )

                    with patch.object(app, "__file__", str(root / "main.py")):
                        with patch("builtins.input", side_effect=["2", "0"]):
                            with patch(
                                "sys.stdout",
                                new_callable=io.StringIO,
                            ) as output:
                                app.main()

                    text = output.getvalue()
                    self.assertIn("JSON 로드 실패", text)
                    self.assertIn("프로그램을 종료합니다.", text)

    def test_invalid_cases_do_not_stop_analysis(self):
        path = Path(app.__file__).resolve().parent / "data.json"
        original = app.load_data(path)

        for defect in (
            "missing_filter",
            "invalid_label",
            "invalid_number",
        ):
            with self.subTest(defect=defect):
                data = json.loads(json.dumps(original))

                if defect == "missing_filter":
                    del data["filters"]["size_5"]["x"]
                elif defect == "invalid_label":
                    data["patterns"]["size_5_1"]["expected"] = "circle"
                else:
                    data["patterns"]["size_5_1"]["input"][0][0] = "bad"

                results = app.analyze_data(data)
                by_id = {
                    result["case_id"]: result
                    for result in results
                }

                self.assertEqual(len(results), 6)
                self.assertIsNone(by_id["size_5_1"]["prediction"])
                self.assertFalse(by_id["size_5_1"]["passed"])
                self.assertTrue(by_id["size_25_2"]["passed"])



class TestReviewRegression(unittest.TestCase):
    def source_data(self):
        return app.load_data(Path(app.__file__).resolve().parent / "data.json")

    def test_numeric_error_is_not_schema_error(self):
        data = self.source_data()
        data["patterns"]["size_5_1"]["input"][0][0] = 1e308
        data["filters"]["size_5"]["x"][0][0] = 1e308
        results = app.analyze_data(data)
        self.assertIn("수치 계산 오류", results[0]["reason"])
        self.assertTrue(results[-1]["passed"])

    def test_benchmark_rejects_overflow_before_timing(self):
        with self.assertRaises(ValueError):
            app.benchmark_mac([[1e308]], [[1e308]], repeats=10)

    def test_non_finite_matrix_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    app.validate_matrix([[value]], "matrix")

    def test_normalization_collision_rejected(self):
        filters = self.source_data()["filters"]
        filters["size_5"]["+"] = filters["size_5"]["cross"]
        with self.assertRaises(ValueError):
            app.get_filters(filters, 5)

    def test_missing_case_fields(self):
        data = self.source_data()
        for field in ("input", "expected"):
            with self.subTest(field=field):
                case = dict(data["patterns"]["size_5_1"])
                del case[field]
                with self.assertRaises(ValueError):
                    app.analyze_case("size_5_1", case, data["filters"])

    def test_direct_analysis_invalid_top_level(self):
        for data in ([], {}, {"filters": [], "patterns": {}}):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    app.analyze_data(data)

    def test_empty_patterns_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text('{"filters": {}, "patterns": {}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                app.load_data(path)

    def test_bom_json_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(self.source_data()), encoding="utf-8-sig")
            self.assertEqual(len(app.load_data(path)["patterns"]), 6)

    def test_decimal_exact_ties(self):
        from decimal import Decimal, localcontext
        path = Path(app.__file__).resolve().parent / "data.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"),
                          parse_float=Decimal, parse_int=Decimal)
        with localcontext() as context:
            context.prec = 100
            for key in ("size_5_1", "size_13_2", "size_25_1"):
                with self.subTest(key=key):
                    n = int(key.split("_")[1])
                    pattern = data["patterns"][key]["input"]
                    filters = data["filters"][f"size_{n}"]
                    scores = [sum((pattern[i][j] * matrix[i][j]
                                   for i in range(n) for j in range(n)), Decimal(0))
                              for matrix in (filters["cross"], filters["x"])]
                    self.assertEqual(scores[0], scores[1])

    def test_manual_normal_and_menu_retry(self):
        cross = ["0 1 0", "1 1 1", "0 1 0"]
        x = ["1 0 1", "0 1 0", "1 0 1"]
        inputs = ["invalid", "1"] + cross + x + x + ["0"]
        with patch("builtins.input", side_effect=inputs):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                app.main()
        text = output.getvalue()
        self.assertIn("0, 1, 2, 3 중 하나", text)
        self.assertIn("판정: B", text)
        self.assertIn("성능 분석", text)

    def test_script_from_another_directory(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", str(Path(app.__file__).resolve())],
                cwd=directory, input="2\n0\n", text=True,
                encoding="utf-8", capture_output=True,
                env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8"),
                timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("총 테스트: 6개", result.stdout)
        self.assertIn("실패: 3개", result.stdout)

    def test_cli_eof_is_clean(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-B", str(Path(app.__file__).resolve())],
            input="", text=True, encoding="utf-8", capture_output=True,
            env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8"),
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("입력이 중단", result.stdout)


if __name__ == "__main__":
    unittest.main()