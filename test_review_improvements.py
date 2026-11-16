import csv
import io
import platform
import tempfile
import unittest
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import main as app
import compare_mac as comparison


class TestReviewImprovements(unittest.TestCase):
    def test_menu_lists_all_choices(self):
        with patch("builtins.input", side_effect=["bad", "0"]):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                app.main()
        self.assertIn("0, 1, 2, 3 중 하나를 입력하세요.", output.getvalue())
        self.assertNotIn("생성 패턴 분석은 3을 입력하세요.", output.getvalue())
        self.assertIn("프로그램을 종료합니다.", output.getvalue())

    def test_measurement_order_rotates_balanced(self):
        expected = ["two_d", "one_d", "converted",
                    "one_d", "converted", "two_d",
                    "converted", "two_d", "one_d"]
        for rounds in (3, 6):
            calls = []
            def measure(operation, repeats):
                self.assertEqual(repeats, 10)
                calls.append(operation.__name__)
                return 0.001
            with self.subTest(rounds=rounds):
                with patch.object(comparison, "_batch_average_ms", side_effect=measure):
                    result = comparison.compare_mac([[1.0]], [[1.0]],
                                                    repeats=10, rounds=rounds)
                self.assertEqual(calls, expected * (rounds // 3))
                for stats in result["stages"].values():
                    self.assertEqual(stats["samples_ms"], [0.001] * rounds)

    def test_mismatch_rejected_before_timing(self):
        with patch.object(app, "mac_flat", return_value=99.0):
            with patch.object(comparison, "_batch_average_ms") as timer:
                with self.assertRaisesRegex(ValueError, "일치"):
                    comparison.compare_mac([[1.0]], [[1.0]], repeats=10, rounds=3)
                timer.assert_not_called()

    def test_csv_values_and_script_relative_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(comparison, "__file__", str(root / "compare_mac.py")):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    results = comparison.run_comparison()
            path = root / "mac_comparison.csv"
            self.assertEqual({p.name for p in root.iterdir()}, {"mac_comparison.csv"})
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            with path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(len(reader.fieldnames), 12)
            self.assertEqual(len(rows), 12)
            self.assertIn("측정 결과 저장: mac_comparison.csv", output.getvalue())
            self.assertEqual({int(row["size"]) for row in rows}, {3, 5, 13, 25})
            lookup = {(r["size"], name): stats
                      for r in results for name, stats in r["stages"].items()}
            self.assertEqual(len({(row["size"], row["stage"]) for row in rows}), 12)
            for row in rows:
                size = int(row["size"])
                stats = lookup[(size, row["stage"])]
                samples = list(map(float, row["round_averages_ms"].split(";")))
                self.assertEqual(row["python"], platform.python_version())
                self.assertEqual(row["platform"], platform.platform())
                self.assertEqual(int(row["mac_count"]), size * size)
                self.assertEqual(float(row["score"]), 1.0)
                self.assertEqual(int(row["repeats_per_round"]), 1000)
                self.assertEqual(int(row["rounds"]), 6)
                self.assertEqual(samples, stats["samples_ms"])
                self.assertEqual(len(samples), 6)
                self.assertAlmostEqual(float(row["average_ms"]), mean(samples))
                self.assertEqual(float(row["min_ms"]), min(samples))
                self.assertEqual(float(row["max_ms"]), max(samples))

    def test_csv_overwrite_preserves_archive(self):
        fixture = comparison.compare_mac([[1.0]], [[1.0]], repeats=10, rounds=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mac_comparison.csv"
            path.write_text("old result", encoding="utf-8")
            archive = root / "mac_comparison_run1.csv"
            archive.write_bytes(b"preserve archive")
            with patch.object(comparison, "__file__", str(root / "compare_mac.py")):
                with patch.object(comparison, "compare_mac", return_value=fixture):
                    with patch("sys.stdout", new_callable=io.StringIO):
                        comparison.run_comparison()
            with path.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 12)
            self.assertNotIn("old result", path.read_text(encoding="utf-8-sig"))
            self.assertEqual(archive.read_bytes(), b"preserve archive")


if __name__ == "__main__":
    unittest.main()
