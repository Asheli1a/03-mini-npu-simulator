import io
from unittest.mock import patch

import main as app
import unittest

from main import calculate_score
from patterns import generate_cross, generate_x, validate_pattern_size


class TestPatternGenerator(unittest.TestCase):
    def test_invalid_types(self):
        for size in (True, False, 3.0, "3", None):
            for function in (validate_pattern_size, generate_cross, generate_x):
                with self.subTest(size=size, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(size)

    def test_invalid_sizes(self):
        for size in (-3, 0, 1, 2, 4):
            for function in (validate_pattern_size, generate_cross, generate_x):
                with self.subTest(size=size, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(size)

    def test_dimensions_and_values(self):
        for size in (3, 5, 13, 25):
            self.assertEqual(validate_pattern_size(size), size)
            for function in (generate_cross, generate_x):
                with self.subTest(size=size, function=function.__name__):
                    matrix = function(size)
                    self.assertEqual(len(matrix), size)
                    self.assertTrue(all(len(row) == size for row in matrix))
                    self.assertTrue(all(type(v) is float and v in (0.0, 1.0)
                                        for row in matrix for v in row))

    def test_cross_three(self):
        self.assertEqual(generate_cross(3), [
            [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])

    def test_x_three(self):
        self.assertEqual(generate_x(3), [
            [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    def test_active_cell_count(self):
        for size in (3, 5, 13, 25):
            for function in (generate_cross, generate_x):
                with self.subTest(size=size, function=function.__name__):
                    self.assertEqual(sum(sum(row) for row in function(size)),
                                     2 * size - 1)

    def test_pattern_rules_and_overlap(self):
        for size in (3, 5, 13, 25):
            cross, x = generate_cross(size), generate_x(size)
            overlap = []
            for row in range(size):
                for column in range(size):
                    self.assertEqual(bool(cross[row][column]),
                                     row == size // 2 or column == size // 2)
                    self.assertEqual(bool(x[row][column]),
                                     row == column or row + column == size - 1)
                    if cross[row][column] and x[row][column]:
                        overlap.append((row, column))
            self.assertEqual(overlap, [(size // 2, size // 2)])

    def test_rows_are_independent(self):
        for function in (generate_cross, generate_x):
            matrix = function(5)
            self.assertEqual(len({id(row) for row in matrix}), 5)
            other_rows = [row[:] for row in matrix[1:]]
            matrix[0][0] = 99.0
            self.assertEqual(matrix[1:], other_rows)

    def test_calls_are_independent(self):
        for function in (generate_cross, generate_x):
            first, second = function(5), function(5)
            original = [row[:] for row in second]
            first[0][0] = 99.0
            self.assertEqual(second, original)

    def test_mac_with_generated_patterns(self):
        for size in (3, 5, 13, 25):
            cross, x = generate_cross(size), generate_x(size)
            self.assertEqual(calculate_score(cross, cross), 2 * size - 1)
            self.assertEqual(calculate_score(x, x), 2 * size - 1)
            self.assertEqual(calculate_score(cross, x), 1.0)
            self.assertEqual(calculate_score(x, cross), 1.0)

class TestGeneratedMode(unittest.TestCase):
    def run_app(self, inputs):
        with patch("builtins.input", side_effect=inputs):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                app.main()

        return output.getvalue()

    def test_cross_mode(self):
        text = self.run_app(["3", "5", "1", "0"])

        self.assertIn("Cross 점수: 9.0", text)
        self.assertIn("X 점수: 1.0", text)
        self.assertIn("판정: Cross", text)
        self.assertIn("성능 분석", text)
        self.assertIn("프로그램을 종료합니다.", text)

    def test_x_mode(self):
        text = self.run_app(["3", "5", "2", "0"])

        self.assertIn("Cross 점수: 1.0", text)
        self.assertIn("X 점수: 9.0", text)
        self.assertIn("판정: X", text)

    def test_invalid_size_retry(self):
        with patch("builtins.input", side_effect=["hello", "4", "1", "5"]):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                size = app.read_generated_size()

        self.assertEqual(size, 5)
        self.assertEqual(output.getvalue().count("입력 오류"), 3)

    def test_invalid_pattern_choice_retry(self):
        text = self.run_app(["3", "3", "bad", "2", "0"])

        self.assertIn("입력 오류: 1 또는 2", text)
        self.assertIn("X 점수: 5.0", text)
        self.assertIn("판정: X", text)

    def test_large_pattern_display_omitted(self):
        text = self.run_app(["3", "27", "1", "0"])

        self.assertIn("큰 배열의 전체 출력은 생략합니다.", text)
        self.assertIn("Cross 점수: 53.0", text)
        self.assertIn("판정: Cross", text)


if __name__ == "__main__":
    unittest.main()
