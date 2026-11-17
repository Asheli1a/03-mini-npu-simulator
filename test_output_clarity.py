import io
import re
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import main as app
from patterns import generate_cross, generate_x


class TestOutputClarity(unittest.TestCase):
    def setUp(self):
        self.filters = {"size_5": {"cross": generate_cross(5), "x": generate_x(5)}}

    def output(self, patterns, filters=None):
        data = {"filters": filters or self.filters, "patterns": patterns}
        with patch.object(app, "load_data", return_value=data):
            with patch.object(app, "run_json_performance"):
                with redirect_stdout(io.StringIO()) as output:
                    app.run_json_mode()
        return output.getvalue()

    def test_original_tie_evidence_and_summary(self):
        data = app.load_data(Path(app.__file__).resolve().parent / "data.json")
        text = self.output(data["patterns"], data["filters"])
        self.assertEqual(text.count("동점 근거:"), 3)
        self.assertEqual(text.count("판정 기준: 점수 차이 < 1e-09 (절대오차)"), 3)
        for result in app.analyze_data(data):
            if result["prediction"] == "UNDECIDED":
                diff = abs(result["score_cross"] - result["score_x"])
                self.assertIn(f"동점 근거: |Cross - X| = {diff!r}", text)
        for expected in ("총 테스트: 6개", "통과: 3개", "실패: 3개",
                         "동점 판정(UNDECIDED): 3개", "예측 불일치: 0개"):
            self.assertIn(expected, text)

    def test_exact_tie_displays_zero(self):
        text = self.output({"size_5_1": {"input": [[0.0] * 5 for _ in range(5)],
                                          "expected": "+"}})
        self.assertIn("동점 근거: |Cross - X| = 0.0", text)
        self.assertIn("판정: UNDECIDED", text)
        self.assertIn("실패: 1개", text)

    def test_near_tie_preserves_small_difference(self):
        pattern = [[0.0] * 5 for _ in range(5)]
        pattern[0][2] = app.EPSILON / 2
        text = self.output({"size_5_1": {"input": pattern, "expected": "+"}})
        self.assertIn("동점 근거: |Cross - X| = 5e-10", text)

    def test_winner_has_no_tie_evidence(self):
        text = self.output({"size_5_1": {"input": generate_cross(5), "expected": "+"}})
        self.assertNotIn("동점 근거:", text)
        self.assertNotIn("판정 기준: 점수 차이", text)
        self.assertIn("통과: 1개", text)

    def test_schema_error_keeps_next_case(self):
        text = self.output({"bad_key": {}, "size_5_2": {
            "input": generate_cross(5), "expected": "+"}})
        self.assertIn("허용 예: size_5_1, size_13_2, size_25_1", text)
        self.assertIn("판정: Cross", text)
        self.assertNotIn("동점 근거:", text)
        self.assertIn("데이터/스키마 오류: 1개", text)

    def test_invalid_key_has_examples(self):
        for key in ("size_5_bad", "size_05_1", "size_5_0", "other_5_1"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "허용 예: size_5_1"):
                    app.parse_case_size(key)

    def test_valid_sizes_and_unsupported_size_unchanged(self):
        for size in (5, 13, 25):
            self.assertEqual(app.parse_case_size(f"size_{size}_1"), size)
        with self.assertRaisesRegex(ValueError, "지원하지 않는 JSON 패턴 크기: 7"):
            app.parse_case_size("size_7_1")

    def test_performance_table_contract(self):
        with redirect_stdout(io.StringIO()) as output:
            app.print_performance_table([{"size": 3, "average_ms": 0.123456,
                                          "mac_count": 9, "repeats": 1000}])
        text = output.getvalue()
        self.assertIn("측정 대상: 필터 1개에 대한 MAC 함수 호출", text)
        self.assertIn("크기       평균 시간(ms)    MAC 횟수(N²)    반복 횟수", text)
        self.assertRegex(text, r"3×3\s+0\.123456\s+9\s+1000")

    def test_input_errors_match_documentation(self):
        inputs = ["0 1", "0 hello 1", "nan 0 1", "0 1 0", "1 1 1", "0 1 0"]
        with patch("builtins.input", side_effect=inputs) as reader:
            with redirect_stdout(io.StringIO()) as output:
                matrix = app.read_matrix("필터 A")
        text = output.getvalue()
        for message in ("각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.",
                        "숫자로 변환할 수 없는 값이 있습니다.",
                        "nan, inf 대신 유한한 숫자를 입력하세요."):
            self.assertIn("입력 형식 오류: " + message, text)
        self.assertEqual(reader.call_count, 6)
        self.assertEqual(matrix, generate_cross(3))

    def test_benchmark_keeps_call_boundary_and_units(self):
        events = []
        def timer():
            events.append("timer")
            return len(events) * 100
        def operation(*args):
            events.append("mac")
            return 1.0
        with patch.object(app, "mac", side_effect=operation):
            with patch.object(app, "perf_counter_ns", side_effect=timer):
                result = app.benchmark_mac([[1.0]], [[1.0]], repeats=10)
        self.assertEqual(events[:6], ["mac"] * 6)
        self.assertEqual(events[6:], ["timer", "mac", "timer"] * 10)
        self.assertEqual(result["mac_count"], 1)
        self.assertEqual(result["repeats"], 10)
        self.assertAlmostEqual(result["average_ms"], 0.0002)


if __name__ == "__main__":
    unittest.main()
