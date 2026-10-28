import unittest

import pandas as pd

from src.config import StrategyConfig
from src.low_turnover_validation import (
    candidate_builders,
    candidate_metadata,
    summarize_rows,
)


class LowTurnoverValidationTests(unittest.TestCase):
    def test_corrected_candidate_keeps_daily_40_90_windows(self):
        config = StrategyConfig()
        metadata = candidate_metadata(config, candidate_builders())
        corrected = metadata[
            metadata["candidate"] == "pullback90_monthly_top10_b20"
        ].iloc[0]
        current = metadata[
            metadata["candidate"] == "current_weekly_top5"
        ].iloc[0]
        self.assertEqual(corrected["effective_pullback_days"], 90)
        self.assertEqual(current["effective_momentum_days"], 8)
        self.assertEqual(current["effective_pullback_days"], 18)

    def test_pass_rule_requires_excess_and_lower_turnover(self):
        rows = pd.DataFrame({
            "test_total_return": [5, 4, 3, 2, 1],
            "test_benchmark_return": [1, 1, 1, 1, 1],
            "test_sharpe": [0.5, 0.4, 0.3, 0.2, 0.1],
            "test_max_drawdown": [-10, -11, -12, -9, -8],
            "test_annualized_selection_turnover_pct": [800] * 5,
            "test_total_fees": [100] * 5,
        })
        self.assertTrue(summarize_rows(rows)["passed"])
        rows["test_annualized_selection_turnover_pct"] = 2000
        self.assertFalse(summarize_rows(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
