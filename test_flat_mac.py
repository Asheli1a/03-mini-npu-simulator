import unittest
from pathlib import Path

import main as app
from patterns import generate_cross, generate_x


class TestFlatMac(unittest.TestCase):
    def test_flatten_order(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        self.assertEqual(
            app.flatten_matrix(matrix),
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        )

    def test_flatten_returns_independent_list(self):
        matrix = [
            [1, 2],
            [3, 4],
        ]

        flattened = app.flatten_matrix(matrix)
        flattened[0] = 99

        self.assertEqual(matrix[0][0], 1)

        matrix[1][1] = 88

        self.assertEqual(flattened[3], 4)

    def test_flatten_rejects_invalid_matrix(self):
        cases = [
            [],
            [[1, 2], [3]],
            [[True]],
            [["bad"]],
            [[float("nan")]],
        ]

        for matrix in cases:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    app.flatten_matrix(matrix)

    def test_known_score(self):
        self.assertEqual(
            app.calculate_flat_score(
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ),
            32.0,
        )

    def test_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "길이 불일치"):
            app.calculate_flat_score([1.0, 2.0], [1.0])

    def test_invalid_vectors(self):
        for vector in ([], None, (1, 2), [[1]], [True], ["1"]):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    app.calculate_flat_score(vector, [1.0])

                with self.assertRaises(ValueError):
                    app.calculate_flat_score([1.0], vector)

    def test_non_finite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    app.calculate_flat_score([value], [1.0])

                with self.assertRaises(ValueError):
                    app.calculate_flat_score([1.0], [value])

    def test_overflow_rejected(self):
        cases = [
            ([1e308], [1e308]),
            ([10 ** 400], [1]),
        ]

        for pattern, filter_vector in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaises(app.CalculationError):
                    app.calculate_flat_score(pattern, filter_vector)

    def test_generated_patterns_match_2d(self):
        for size in (3, 5, 13, 25):
            cross = generate_cross(size)
            x = generate_x(size)

            for pattern in (cross, x):
                for filter_matrix in (cross, x):
                    with self.subTest(
                        size=size,
                        same_pattern=pattern is filter_matrix,
                    ):
                        score_2d = app.calculate_score(
                            pattern,
                            filter_matrix,
                        )
                        score_1d = app.calculate_flat_score(
                            app.flatten_matrix(pattern),
                            app.flatten_matrix(filter_matrix),
                        )

                        self.assertEqual(score_1d, score_2d)

    def test_negative_decimal_values_match_2d(self):
        pattern = [
            [-0.1, 0.2],
            [0.3, -0.4],
        ]
        filter_matrix = [
            [0.5, -0.6],
            [0.7, 0.8],
        ]

        self.assertEqual(
            app.calculate_flat_score(
                app.flatten_matrix(pattern),
                app.flatten_matrix(filter_matrix),
            ),
            app.calculate_score(pattern, filter_matrix),
        )

    def test_original_data_scores_and_predictions_match(self):
        path = Path(app.__file__).resolve().parent / "data.json"
        data = app.load_data(path)

        for case_id, case in data["patterns"].items():
            with self.subTest(case_id=case_id):
                size = app.parse_case_size(case_id)
                filters = app.get_filters(data["filters"], size)

                pattern = case["input"]
                flat_pattern = app.flatten_matrix(pattern)

                scores_2d = []
                scores_1d = []

                for label in ("Cross", "X"):
                    filter_matrix = filters[label]

                    scores_2d.append(
                        app.calculate_score(pattern, filter_matrix)
                    )
                    scores_1d.append(
                        app.calculate_flat_score(
                            flat_pattern,
                            app.flatten_matrix(filter_matrix),
                        )
                    )

                self.assertEqual(scores_1d, scores_2d)
                self.assertEqual(
                    app.decide_label(*scores_1d),
                    app.decide_label(*scores_2d),
                )

    def test_raw_mac_matches_validated_calculation(self):
        pattern = [0.1, -0.2, 0.3]
        filter_vector = [0.4, 0.5, -0.6]

        self.assertEqual(
            app.mac_flat(pattern, filter_vector),
            app.calculate_flat_score(pattern, filter_vector),
        )


if __name__ == "__main__":
    unittest.main()