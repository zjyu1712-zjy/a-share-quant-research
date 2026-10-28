import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.fundamental_factor_screening import (
    FACTOR_NAMES,
    _asof_values,
    build_fundamental_factors,
    calculate_reference_correlations,
    profile_factor_quality,
    screen_fundamental_factors,
)


class FundamentalFactorScreeningTests(unittest.TestCase):
    @staticmethod
    def _frame(scale: float = 1.0) -> pd.DataFrame:
        dates = pd.bdate_range("2021-01-04", "2023-01-06")
        position = np.arange(len(dates), dtype=float)
        return pd.DataFrame(
            {
                "code": "sh.600000",
                "peTTM": 12.0 - 0.005 * position,
                "pbMRQ": (1.2 + 0.001 * position) * scale,
                "psTTM": 2.0 + 0.0005 * position,
                "pcfNcfTTM": 8.0,
            },
            index=dates,
        )

    def test_asof_matching_never_uses_future_record(self):
        frame = self._frame()
        targets = pd.DatetimeIndex(
            [pd.Timestamp("2022-06-11"), pd.Timestamp("2022-06-13")]
        )
        matched = _asof_values(frame, targets, tolerance_days=10)
        self.assertTrue(
            (matched["source_date"] <= matched["target_date"]).all()
        )
        self.assertEqual(
            matched.iloc[0]["source_date"],
            pd.Timestamp("2022-06-10"),
        )

    def test_factor_math_and_point_in_time_universe(self):
        signal_date = pd.Timestamp("2023-01-06")
        valuation_data = {
            f"{number:06d}": self._frame(scale=1 + number / 1000)
            for number in range(1, 61)
        }
        valuation_data["999999"] = self._frame(scale=10)

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(
                    codes={f"{number:06d}" for number in range(1, 61)},
                    code_to_sector={
                        f"{number:06d}": (
                            "sector_a" if number <= 30 else "sector_b"
                        )
                        for number in range(1, 61)
                    },
                )

        result = build_fundamental_factors(
            valuation_data=valuation_data,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=FakeUniverse(),
        )
        self.assertEqual(result["code"].nunique(), 60)
        self.assertNotIn("999999", set(result["code"]))
        self.assertTrue(set(FACTOR_NAMES).issubset(result.columns))
        self.assertTrue(
            (
                pd.to_datetime(result["source_date"])
                <= pd.to_datetime(result["date"])
            ).all()
        )
        self.assertTrue(
            (
                pd.to_datetime(result["previous_source_date"])
                <= pd.to_datetime(result["date"]) - pd.DateOffset(years=1)
            ).all()
        )
        sector_medians = result.groupby("sector")[
            "value_book_to_price_mrq"
        ].median()
        np.testing.assert_allclose(sector_medians, 0.0, atol=1e-12)

    def test_quality_profile_rejects_future_match(self):
        rows = []
        for number in range(60):
            row = {
                "date": pd.Timestamp("2023-01-06"),
                "code": f"{number:06d}",
                "source_date": pd.Timestamp("2023-01-06"),
                "previous_source_date": pd.Timestamp("2022-01-06"),
            }
            row.update({factor: 1.0 for factor in FACTOR_NAMES})
            rows.append(row)
        factors = pd.DataFrame(rows)
        _, summary = profile_factor_quality(factors)
        self.assertTrue(summary["passed"])
        factors.loc[0, "source_date"] = pd.Timestamp("2023-01-07")
        _, summary = profile_factor_quality(factors)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["future_current_matches"], 1)

    def test_screen_requires_coverage_and_fold_stability(self):
        factor = FACTOR_NAMES[0]
        aggregate = pd.DataFrame(
            [
                {
                    "factor": factor,
                    "horizon_weeks": 4,
                    "mean_rank_ic": 0.03,
                    "positive_ic_ratio": 0.60,
                    "hac_t_stat": 2.0,
                    "mean_top_bottom_spread_pct": 1.0,
                },
                {
                    "factor": factor,
                    "horizon_weeks": 8,
                    "mean_rank_ic": 0.02,
                    "positive_ic_ratio": 0.60,
                    "hac_t_stat": 2.0,
                    "mean_top_bottom_spread_pct": 1.0,
                },
            ]
        )
        by_fold = pd.DataFrame(
            {
                "fold": range(1, 8),
                "factor": [factor] * 7,
                "horizon_weeks": [4] * 7,
                "mean_rank_ic": [0.02] * 5 + [-0.01] * 2,
            }
        )
        quality = pd.DataFrame(
            {
                "factor": [factor] * 10,
                "coverage_ratio": [0.80] * 10,
            }
        )
        result = screen_fundamental_factors(
            aggregate,
            by_fold,
            quality,
        )
        self.assertEqual(result.iloc[0]["status"], "promising")
        quality["coverage_ratio"] = 0.50
        result = screen_fundamental_factors(
            aggregate,
            by_fold,
            quality,
        )
        self.assertEqual(result.iloc[0]["status"], "rejected")

    def test_reference_correlation_uses_matching_date_and_code(self):
        dates = pd.to_datetime(["2023-01-06"] * 60)
        codes = [f"{number:06d}" for number in range(60)]
        factors = pd.DataFrame(
            {
                "date": dates,
                "code": codes,
                **{
                    factor: np.arange(60, dtype=float)
                    for factor in FACTOR_NAMES
                },
            }
        )
        reference = pd.DataFrame(
            {
                "date": dates,
                "code": codes,
                "low_volatility_60": np.arange(60, dtype=float),
                "sector_neutral_low_volatility_60": -np.arange(
                    60, dtype=float
                ),
            }
        )
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.csv"
            reference.to_csv(path, index=False)
            _, summary = calculate_reference_correlations(factors, path)
        first = summary[
            (summary["factor"] == FACTOR_NAMES[0])
            & (summary["reference_factor"] == "low_volatility_60")
        ].iloc[0]
        self.assertAlmostEqual(first["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
