import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.factor_screening import (
    _hac_t_stat,
    build_candidate_factors,
    calculate_signal_diagnostics,
    screen_factors,
)


class FactorScreeningTests(unittest.TestCase):
    def test_candidate_factors_use_point_in_time_pool_and_no_future_prices(self):
        dates = pd.bdate_range("2023-01-02", periods=150)
        signal_date = dates[130]
        price_data = {
            "000001": pd.DataFrame(
                {"close": 10 * np.cumprod(1 + np.linspace(-0.01, 0.02, 150))},
                index=dates,
            ),
            "000002": pd.DataFrame(
                {"close": 20 * np.cumprod(1 + np.linspace(0.015, -0.005, 150))},
                index=dates,
            ),
            "999999": pd.DataFrame(
                {"close": np.linspace(10, 100, 150)},
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

        original = build_candidate_factors(
            price_data=price_data,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=FakeUniverse(),
        ).sort_values("code")
        self.assertEqual(original["code"].tolist(), ["000001", "000002"])

        changed_future = {
            code: frame.copy() for code, frame in price_data.items()
        }
        for frame in changed_future.values():
            frame.loc[frame.index > signal_date, "close"] *= 10
        recalculated = build_candidate_factors(
            price_data=changed_future,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=FakeUniverse(),
        ).sort_values("code")
        factor_columns = [
            "low_volatility_60",
            "sector_neutral_low_volatility_60",
            "momentum_6_1",
            "sector_relative_strength_60",
            "pullback_90_reference",
        ]
        np.testing.assert_allclose(
            original[factor_columns],
            recalculated[factor_columns],
            equal_nan=True,
        )

    def test_hac_t_stat_has_expected_sign(self):
        positive = pd.Series(np.linspace(0.01, 0.20, 20))
        self.assertGreater(_hac_t_stat(positive, 3), 0)
        self.assertLess(_hac_t_stat(-positive, 3), 0)

    def test_signal_diagnostics_recognizes_monotonic_factor(self):
        rows = []
        for number in range(1, 101):
            rows.append({
                "date": pd.Timestamp("2024-01-05"),
                "code": f"{number:06d}",
                "test_factor": number,
                "forward_1w_pct": number / 10,
                "forward_2w_pct": number / 10,
                "forward_4w_pct": number / 10,
                "forward_8w_pct": number / 10,
            })
        result = calculate_signal_diagnostics(
            pd.DataFrame(rows),
            factor_names=["test_factor"],
        )
        self.assertTrue(np.allclose(result["rank_ic"], 1.0))
        self.assertTrue((result["top_bottom_spread_pct"] > 0).all())

    def test_screening_requires_fold_stability(self):
        aggregate = pd.DataFrame([
            {
                "factor": "low_volatility_60",
                "horizon_weeks": 4,
                "mean_rank_ic": 0.03,
                "positive_ic_ratio": 0.60,
                "hac_t_stat": 2.0,
                "mean_top_bottom_spread_pct": 1.0,
            },
            {
                "factor": "low_volatility_60",
                "horizon_weeks": 8,
                "mean_rank_ic": 0.02,
                "positive_ic_ratio": 0.60,
                "hac_t_stat": 2.0,
                "mean_top_bottom_spread_pct": 1.0,
            },
        ])
        by_fold = pd.DataFrame({
            "fold": range(1, 8),
            "factor": ["low_volatility_60"] * 7,
            "horizon_weeks": [4] * 7,
            "mean_rank_ic": [0.02] * 5 + [-0.01] * 2,
        })
        result = screen_factors(aggregate, by_fold)
        self.assertEqual(result.iloc[0]["status"], "promising")
        by_fold["mean_rank_ic"] = [0.02] * 4 + [-0.01] * 3
        result = screen_factors(aggregate, by_fold)
        self.assertEqual(result.iloc[0]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
