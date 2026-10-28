import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.frozen_holdout_validation import (
    CANDIDATES,
    HOLDOUT_START,
    MIN_HOLDOUT_MONTHS,
    build_monthly_return_rows,
    configure_candidate,
    ensure_frozen_protocol,
    frozen_base_config,
    frozen_protocol,
    summarize_holdout,
)


class FrozenHoldoutValidationTests(unittest.TestCase):
    def test_protocol_freezes_exact_candidates_and_decision_boundary(self):
        protocol = frozen_protocol()
        self.assertEqual(
            set(protocol["candidates"]),
            {
                "current_weekly_top5",
                "pullback90_monthly_top10_b30",
                "low_volatility60_monthly_top10_b30",
            },
        )
        self.assertEqual(
            protocol["gate"]["minimum_observed_months"],
            MIN_HOLDOUT_MONTHS,
        )
        self.assertTrue(
            all(
                not details["prior_research_passed"]
                for details in CANDIDATES.values()
            )
        )
        self.assertIn("不自动修改正式策略", protocol["decision_boundary"])

    def test_existing_protocol_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path, digest = ensure_frozen_protocol(output)
            self.assertEqual(len(digest), 64)
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["holdout_end"] = "2099-01-01"
            path.write_text(
                json.dumps(saved, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ensure_frozen_protocol(output)

    def test_candidate_configuration_matches_frozen_metadata(self):
        base = frozen_base_config()
        weekly = configure_candidate(base, "current_weekly_top5")
        self.assertEqual(weekly.timeframe, "WEEKLY")
        self.assertEqual(weekly.portfolio.rebalance_frequency, "weekly")
        self.assertEqual(weekly.portfolio.top_n, 5)
        self.assertEqual(weekly.portfolio.rank_buffer, 0)

        low_turnover = configure_candidate(
            base,
            "pullback90_monthly_top10_b30",
        )
        self.assertEqual(low_turnover.timeframe, "DAILY")
        self.assertEqual(
            low_turnover.portfolio.rebalance_frequency,
            "monthly",
        )
        self.assertEqual(low_turnover.portfolio.top_n, 10)
        self.assertEqual(low_turnover.portfolio.rank_buffer, 30)
        self.assertEqual(low_turnover.scoring.momentum_weight, 0.0)
        self.assertEqual(low_turnover.scoring.pullback_weight, 1.0)

    def test_monthly_rows_use_initial_cash_for_first_month(self):
        equity_dates = pd.bdate_range(HOLDOUT_START, periods=45)
        equity = pd.DataFrame(
            {
                "equity": [
                    99_000 + position * 100
                    for position in range(len(equity_dates))
                ]
            },
            index=equity_dates,
        )
        index_data = pd.DataFrame(
            {
                "open": [100 + position for position in range(45)],
                "close": [101 + position for position in range(45)],
            },
            index=equity_dates,
        )
        rows = build_monthly_return_rows(
            candidate="current_weekly_top5",
            equity_curve=equity,
            index_data=index_data,
            first_execution=equity_dates[0],
            initial_cash=100_000,
            end=str(equity_dates[-1].date()),
        )
        first_month_end = equity.loc[: rows.iloc[0]["month_end"], "equity"].iloc[-1]
        expected = (first_month_end / 100_000 - 1) * 100
        self.assertAlmostEqual(
            rows.iloc[0]["strategy_return_pct"],
            expected,
        )

    def test_pass_rule_requires_twelve_months_and_every_gate(self):
        continuous = {
            "total_return": 15.0,
            "benchmark_return": 8.0,
            "excess_return": 7.0,
            "sharpe": 0.8,
            "max_drawdown": -12.0,
            "annualized_selection_turnover_pct": 500.0,
        }
        rows = pd.DataFrame({
            "strategy_return_pct": [1.0] * 12,
            "excess_return_pp": [0.5] * 12,
        })
        self.assertTrue(
            summarize_holdout(
                continuous=continuous,
                monthly_rows=rows,
            )["holdout_passed"]
        )
        self.assertFalse(
            summarize_holdout(
                continuous=continuous,
                monthly_rows=rows.iloc[:11],
            )["holdout_passed"]
        )
        rows.loc[:4, "excess_return_pp"] = -0.5
        self.assertFalse(
            summarize_holdout(
                continuous=continuous,
                monthly_rows=rows,
            )["holdout_passed"]
        )


if __name__ == "__main__":
    unittest.main()
