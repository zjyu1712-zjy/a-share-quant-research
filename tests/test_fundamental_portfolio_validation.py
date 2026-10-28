import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.fundamental_portfolio_validation import (
    CANDIDATES,
    build_candidate_factor_frames,
    candidate_metadata,
    profile_factor_frames,
    validate_result_tables,
)
from src.low_volatility_portfolio_validation import summarize_rows


class FundamentalPortfolioValidationTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        dates = pd.bdate_range("2021-01-04", "2023-01-06")
        signal_date = dates[-1]
        price_data = {}
        valuation_data = {}
        codes = {f"{number:06d}" for number in range(1, 61)}
        sectors = {
            code: ("sector_a" if index <= 30 else "sector_b")
            for index, code in enumerate(sorted(codes), start=1)
        }
        for number, code in enumerate(sorted(codes), start=1):
            wave = np.sin(np.arange(len(dates)) / (12 + number / 10))
            returns = 0.0002 + wave * (0.001 + number / 100_000)
            close = (10 + number / 10) * np.cumprod(1 + returns)
            price_data[code] = pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                },
                index=dates,
            )
            valuation_data[code] = pd.DataFrame(
                {
                    "code": code,
                    "peTTM": 12.0,
                    "pbMRQ": 0.8 + number / 100,
                    "psTTM": 2.0,
                    "pcfNcfTTM": 8.0,
                },
                index=dates,
            )

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(
                    codes=codes,
                    code_to_sector=sectors,
                )

        return dates, signal_date, price_data, valuation_data, FakeUniverse()

    def test_frames_use_only_signal_date_information_and_fixed_universe(self):
        dates, signal_date, prices, valuations, universe = self._fixture()
        frames = build_candidate_factor_frames(
            price_data=prices,
            valuation_data=valuations,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=universe,
        )
        self.assertEqual(set(frames), set(CANDIDATES))
        for frame in frames.values():
            self.assertEqual(frame["code"].nunique(), 60)
            self.assertEqual(frame["date"].nunique(), 1)

        future_prices = {code: frame.copy() for code, frame in prices.items()}
        future_date = signal_date + pd.offsets.BDay(1)
        for frame in future_prices.values():
            frame.loc[future_date] = frame.iloc[-1] * 50
        recalculated = build_candidate_factor_frames(
            price_data=future_prices,
            valuation_data=valuations,
            rebalance_dates=pd.DatetimeIndex([signal_date]),
            universe_store=universe,
        )
        np.testing.assert_allclose(
            frames["book50_lowvol50"].sort_values("code")[
                ["momentum", "pullback"]
            ],
            recalculated["book50_lowvol50"].sort_values("code")[
                ["momentum", "pullback"]
            ],
        )
        quality, summary = profile_factor_frames(frames, universe)
        self.assertFalse(quality.empty)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["future_fundamental_matches"], 0)

    def test_metadata_predeclares_weights_and_execution_rules(self):
        metadata = candidate_metadata(100_000)
        self.assertEqual(len(metadata), 5)
        self.assertTrue((metadata["top_n"] == 10).all())
        self.assertTrue((metadata["rank_buffer"] == 30).all())
        self.assertTrue((metadata["max_per_sector"] == 2).all())
        self.assertEqual(
            metadata["eligible_for_advancement"].sum(),
            4,
        )
        blended = metadata[
            metadata["candidate"] == "book50_lowvol50"
        ].iloc[0]
        self.assertEqual(blended["book_weight"], 0.5)
        self.assertEqual(blended["low_volatility_weight"], 0.5)

    def test_validation_recomputes_results_and_requires_factor_quality(self):
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
        factor_quality = {
            "duplicate_date_code_rows": 0,
            "future_fundamental_matches": 0,
            "minimum_coverage_ratio": 0.8,
            "median_coverage_ratio": 0.9,
            "passed": True,
        }
        validation = validate_result_tables(
            folds_frame=folds,
            summary_table=summary,
            factor_quality=factor_quality,
            expected_folds=5,
        )
        self.assertEqual(validation["status"], "passed")
        factor_quality["passed"] = False
        validation = validate_result_tables(
            folds_frame=folds,
            summary_table=summary,
            factor_quality=factor_quality,
            expected_folds=5,
        )
        self.assertEqual(validation["status"], "failed")


if __name__ == "__main__":
    unittest.main()
