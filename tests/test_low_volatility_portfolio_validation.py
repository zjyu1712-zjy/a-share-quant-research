import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.low_volatility_portfolio_validation import (
    CANDIDATES,
    build_low_volatility_factor_frame,
    candidate_metadata,
    summarize_rows,
    validate_result_tables,
)


class LowVolatilityPortfolioValidationTests(unittest.TestCase):
    def test_factor_frame_uses_point_in_time_pool_and_no_future_prices(self):
        dates = pd.bdate_range("2022-01-03", periods=180)
        signal_date = dates[150]
        price_data = {
            "000001": pd.DataFrame(
                {
                    "close": 10
                    * np.cumprod(
                        1 + np.linspace(-0.008, 0.012, len(dates))
                    )
                },
                index=dates,
            ),
            "000002": pd.DataFrame(
                {
                    "close": 20
                    * np.cumprod(
                        1 + np.linspace(0.010, -0.006, len(dates))
                    )
                },
                index=dates,
            ),
            "999999": pd.DataFrame(
                {"close": np.linspace(10, 50, len(dates))},
                index=dates,
            ),
        }

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(
                    codes={"000001", "000002"},
                    code_to_sector={
                        "000001": "sector_a",
                        "000002": "sector_a",
                    },
                )

        original = build_low_volatility_factor_frame(
            price_data=price_data,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=FakeUniverse(),
            candidate="balanced_low_volatility_60",
        ).sort_values("code")
        self.assertEqual(original["code"].tolist(), ["000001", "000002"])

        future_changed = {
            code: frame.copy() for code, frame in price_data.items()
        }
        for frame in future_changed.values():
            frame.loc[frame.index > signal_date, "close"] *= 20
        recalculated = build_low_volatility_factor_frame(
            price_data=future_changed,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=FakeUniverse(),
            candidate="balanced_low_volatility_60",
        ).sort_values("code")
        np.testing.assert_allclose(
            original[["momentum", "pullback"]],
            recalculated[["momentum", "pullback"]],
        )

    def test_metadata_fixes_shared_portfolio_rules(self):
        metadata = candidate_metadata(100_000)
        self.assertTrue((metadata["top_n"] == 10).all())
        self.assertTrue((metadata["rank_buffer"] == 30).all())
        self.assertTrue((metadata["eligible_old_rank"] == 40).all())
        self.assertTrue((metadata["max_per_sector"] == 2).all())
        self.assertEqual(
            metadata["eligible_for_advancement"].sum(),
            3,
        )

    def test_pass_rule_requires_consistent_positive_excess(self):
        rows = pd.DataFrame({
            "test_total_return": [5, 4, 3, 2, 1],
            "test_benchmark_return": [1, 1, 1, 1, 1],
            "test_excess_return": [4, 3, 2, 1, 0.5],
            "test_sharpe": [0.8, 0.7, 0.6, 0.5, 0.4],
            "test_max_drawdown": [-8, -9, -10, -7, -6],
            "test_annualized_selection_turnover_pct": [900] * 5,
            "test_total_fees": [100] * 5,
            "test_slippage_cost": [50] * 5,
            "test_friction_drag_pct_of_initial_cash": [0.15] * 5,
            "test_avg_deployment_pct": [95] * 5,
        })
        self.assertTrue(summarize_rows(rows)["passed"])
        rows["test_excess_return"] = [-1, -1, -1, 1, 1]
        self.assertFalse(summarize_rows(rows)["passed"])

    def test_validation_recomputes_headline_metrics(self):
        rows = []
        for candidate in CANDIDATES:
            for fold in range(1, 6):
                rows.append({
                    "fold": fold,
                    "candidate": candidate,
                    "test_total_return": 2.0,
                    "test_benchmark_return": 1.0,
                    "test_excess_return": 1.0,
                    "test_sharpe": 0.5,
                    "test_max_drawdown": -5.0,
                    "test_annualized_selection_turnover_pct": 500.0,
                    "test_total_fees": 100.0,
                    "test_slippage_cost": 50.0,
                    "test_friction_drag_pct_of_initial_cash": 0.15,
                    "test_avg_deployment_pct": 95.0,
                })
        folds = pd.DataFrame(rows)
        summary = pd.DataFrame([
            {"strategy": candidate, **summarize_rows(group)}
            for candidate, group in folds.groupby("candidate")
        ])
        validation = validate_result_tables(
            folds_frame=folds,
            summary_table=summary,
            expected_folds=5,
        )
        self.assertEqual(validation["status"], "passed")


if __name__ == "__main__":
    unittest.main()
