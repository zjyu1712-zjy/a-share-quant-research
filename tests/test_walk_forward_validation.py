import unittest

import pandas as pd

from src.walk_forward_validation import generate_folds, summarize_test_rows


class WalkForwardValidationTests(unittest.TestCase):
    def test_generates_non_overlapping_forward_windows(self):
        folds = generate_folds(
            "2020-01-01", "2022-12-31",
            train_months=24, test_months=6,
        )
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0].train_end, "2021-12-31")
        self.assertEqual(folds[0].test_start, "2022-01-01")
        self.assertEqual(folds[0].test_end, "2022-06-30")
        self.assertEqual(folds[1].train_end, "2022-06-30")
        self.assertEqual(folds[1].test_start, "2022-07-01")

    def test_pass_rule_requires_consistent_positive_oos_windows(self):
        rows = pd.DataFrame({
            "test_total_return": [3.0, 2.0, 1.0, -0.5, 2.0],
            "test_sharpe": [0.5, 0.4, 0.2, -0.1, 0.3],
            "test_max_drawdown": [-8, -9, -5, -7, -6],
            "test_avg_turnover_pct": [100, 90, 80, 70, 60],
        })
        summary = summarize_test_rows(rows)
        self.assertTrue(summary["passed"])
        rows.loc[0:3, "test_total_return"] = -3
        self.assertFalse(summarize_test_rows(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
