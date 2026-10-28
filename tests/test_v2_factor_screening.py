import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2_factor_screening import (
    _render_report,
    add_forward_open_returns,
    build_factor_frame,
    calculate_monthly_diagnostics,
    calculate_price_factors,
    evaluate_factor_gates,
    latest_positive_pb_asof,
    run_v2_factor_screening,
    sector_neutralize_book_to_price,
    summarize_factor_diagnostics,
)
from src.tradability import TradabilityRules
from src.universe import PointInTimeUniverse
from src.v2_factor_data import FACTOR_NAMES, build_factor_protocol


class V2FactorScreeningTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _rules():
        return TradabilityRules(
            min_average_amount=50_000_000.0,
            amount_lookback=20,
            top_n=10,
            initial_cash=100_000.0,
            min_lot=100,
            commission=0.0005,
            min_commission=5.0,
            transfer_fee_rate=0.00001,
            slippage_bps=5.0,
        )

    @staticmethod
    def _passing_summaries():
        aggregate_rows = []
        year_rows = []
        fold_rows = []
        for factor in FACTOR_NAMES:
            for horizon, mean_ic in [(4, 0.025), (8, 0.010), (12, 0.005)]:
                aggregate_rows.append({
                    "factor": factor,
                    "horizon_weeks": horizon,
                    "mean_rank_ic": mean_ic,
                    "median_rank_ic": 0.010,
                    "positive_ic_ratio": 0.60,
                    "hac_t_stat": 3.0,
                    "annualized_icir": 0.60,
                    "annualized_gross_spread_pct": 5.0,
                    "aggregate_quintile_monotonic_corr": 1.0,
                    "correct_adjacent_quintile_steps": 4,
                    "median_coverage_ratio": 0.95,
                    "mean_top_quintile_turnover": 0.50,
                })
            for year in [2021, 2022, 2023, 2024]:
                year_rows.append({
                    "factor": factor,
                    "horizon_weeks": 4,
                    "year": year,
                    "mean_rank_ic": 0.02,
                    "median_coverage_ratio": 0.90,
                })
            for fold in range(1, 8):
                fold_rows.append({
                    "factor": factor,
                    "horizon_weeks": 4,
                    "fold": fold,
                    "mean_rank_ic": 0.02,
                })
        return {
            "aggregate": pd.DataFrame(aggregate_rows),
            "by_year": pd.DataFrame(year_rows),
            "by_fold": pd.DataFrame(fold_rows),
        }

    def test_factor_definitions_and_directions_match_protocol(self):
        dates = pd.bdate_range("2023-01-02", periods=253)
        close = pd.Series(np.linspace(10.0, 20.0, 253), index=dates)
        result = calculate_price_factors(close, dates[-1])
        expected_vol = close.pct_change().iloc[-60:].std() * np.sqrt(252)
        self.assertAlmostEqual(result["low_volatility_60"], -expected_vol)
        self.assertAlmostEqual(
            result["momentum_12_1"],
            close.iloc[-21] / close.iloc[-253] - 1,
        )

    def test_latest_pb_is_backward_only_and_must_be_positive(self):
        valuation = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-29", "2024-01-30", "2024-02-01",
            ]),
            "pbMRQ": [1.5, 2.0, 0.5],
        })
        self.assertEqual(
            latest_positive_pb_asof(valuation, pd.Timestamp("2024-01-31")),
            2.0,
        )
        valuation.loc[1, "pbMRQ"] = -2.0
        self.assertTrue(pd.isna(latest_positive_pb_asof(
            valuation, pd.Timestamp("2024-01-31")
        )))

    def test_sector_neutral_value_has_zero_sector_medians(self):
        cross_section = pd.DataFrame({
            "sector": ["银行", "银行", "制造", "制造"],
            "book_to_price": [0.5, 1.0, 0.25, 0.75],
        })
        neutral = sector_neutralize_book_to_price(cross_section)
        medians = neutral.groupby(cross_section["sector"]).median()
        self.assertAlmostEqual(float(medians.abs().max()), 0.0)

    def test_future_price_changes_do_not_change_historical_factors(self):
        signal = pd.Timestamp("2024-01-31")
        dates = pd.bdate_range(end=signal, periods=253)
        historical = pd.Series(np.linspace(10.0, 20.0, 253), index=dates)
        with_future = pd.concat([
            historical,
            pd.Series([999.0], index=[pd.Timestamp("2024-02-01")]),
        ])
        self.assertEqual(
            calculate_price_factors(historical, signal),
            calculate_price_factors(with_future, signal),
        )

    def test_forward_return_starts_at_next_session_open(self):
        signal = pd.Timestamp("2024-01-31")
        future_dates = pd.bdate_range("2024-02-01", periods=61)
        price = pd.DataFrame(
            {"open": np.arange(100.0, 161.0)}, index=future_dates
        )
        factors = pd.DataFrame({"date": [signal], "code": ["600001"]})
        result = add_forward_open_returns(factors, {"600001": price})
        self.assertAlmostEqual(
            result.iloc[0]["forward_4w"], 120.0 / 100.0 - 1
        )
        self.assertAlmostEqual(
            result.iloc[0]["forward_8w"], 140.0 / 100.0 - 1
        )
        self.assertAlmostEqual(
            result.iloc[0]["forward_12w"], 160.0 / 100.0 - 1
        )

    def test_rank_ic_quintiles_coverage_and_turnover_are_reported(self):
        rows = []
        for date in pd.to_datetime(["2024-01-31", "2024-02-29"]):
            for number in range(120):
                rows.append({
                    "date": date,
                    "code": f"600{number:03d}",
                    "tradable_pool_size": 120,
                    "low_volatility_60": float(number),
                    "sector_neutral_book_to_price_mrq": float(number),
                    "momentum_12_1": float(number),
                    "forward_4w": float(number) / 1000,
                    "forward_8w": float(number) / 900,
                    "forward_12w": float(number) / 800,
                })
        monthly, quintiles, turnover = calculate_monthly_diagnostics(
            pd.DataFrame(rows)
        )
        row = monthly.query(
            "factor == 'low_volatility_60' and horizon_weeks == 4"
        ).iloc[0]
        self.assertAlmostEqual(row["rank_ic"], 1.0)
        self.assertAlmostEqual(row["coverage_ratio"], 1.0)
        self.assertEqual(set(quintiles["quintile"]), {1, 2, 3, 4, 5})
        self.assertTrue(turnover["top_quintile_turnover"].between(0, 1).all())

    def test_factor_coverage_does_not_require_future_return_label(self):
        rows = pd.DataFrame({
            "date": [pd.Timestamp("2025-05-12")] * 120,
            "code": [f"600{number:03d}" for number in range(120)],
            "tradable_pool_size": [120] * 120,
            "low_volatility_60": np.arange(120, dtype=float),
            "sector_neutral_book_to_price_mrq": np.arange(120, dtype=float),
            "momentum_12_1": np.arange(120, dtype=float),
            "forward_4w": [np.nan] * 120,
            "forward_8w": [np.nan] * 120,
            "forward_12w": [np.nan] * 120,
        })
        monthly, _quintiles, _turnover = calculate_monthly_diagnostics(rows)
        row = monthly.query(
            "factor == 'low_volatility_60' and horizon_weeks == 4"
        ).iloc[0]
        self.assertEqual(row["factor_count"], 120)
        self.assertEqual(row["forward_pair_count"], 0)
        self.assertEqual(row["coverage_ratio"], 1.0)
        self.assertTrue(pd.isna(row["rank_ic"]))

    def test_factor_frame_uses_point_in_time_membership_and_tradability(self):
        signal = pd.Timestamp("2024-01-31")
        snapshot_dir = self.root / "snapshots"
        snapshot_dir.mkdir()
        pd.DataFrame({
            "code": ["600001", "600002"],
            "name": ["甲", "乙"],
            "sector": ["银行", "银行"],
        }).to_csv(snapshot_dir / "2024-01-31.csv", index=False)
        dates = pd.bdate_range(end=signal, periods=253)
        price_data = {}
        valuation_data = {}
        for number, code in enumerate(["600001", "600002", "600003"]):
            price_data[code] = pd.DataFrame({
                "open": np.linspace(10 + number, 20 + number, 253),
                "close": np.linspace(10 + number, 20 + number, 253),
                "amount": [100_000_000.0] * 253,
            }, index=dates)
            valuation_data[code] = pd.DataFrame({
                "date": [signal - pd.Timedelta(days=1)],
                "pbMRQ": [2.0 - number * 0.5],
            })
        result = build_factor_frame(
            price_data=price_data,
            valuation_data=valuation_data,
            signal_dates=pd.DatetimeIndex([signal]),
            universe_store=PointInTimeUniverse(snapshot_dir, min_members=1),
            tradability_rules=self._rules(),
        )
        self.assertEqual(set(result["code"]), {"600001", "600002"})
        self.assertEqual(set(result["tradable_pool_size"]), {2})
        values = result.set_index("code")[
            "sector_neutral_book_to_price_mrq"
        ]
        self.assertGreater(values["600002"], values["600001"])

    def test_candidate_advances_only_when_every_gate_passes(self):
        result = evaluate_factor_gates(
            self._passing_summaries(), build_factor_protocol()
        )
        self.assertEqual(set(result["status"]), {"advance"})
        self.assertTrue(result["failed_gates"].eq("").all())

    def test_two_bad_full_years_reject_without_reversing_direction(self):
        summaries = self._passing_summaries()
        mask = (
            (summaries["by_year"]["factor"] == "momentum_12_1")
            & summaries["by_year"]["year"].isin([2021, 2022])
        )
        summaries["by_year"].loc[mask, "mean_rank_ic"] = -0.01
        result = evaluate_factor_gates(
            summaries, build_factor_protocol()
        ).set_index("factor")
        row = result.loc["momentum_12_1"]
        self.assertEqual(row["status"], "reject")
        self.assertIn("positive_full_years", row["failed_gates"])

    def test_summary_assigns_the_seven_frozen_windows(self):
        dates = pd.to_datetime([
            "2022-01-31", "2022-07-29", "2023-01-31", "2023-07-31",
            "2024-01-31", "2024-07-31", "2025-01-27",
        ])
        monthly = pd.DataFrame({
            "date": dates,
            "factor": ["low_volatility_60"] * 7,
            "horizon_weeks": [4] * 7,
            "coverage_ratio": [0.95] * 7,
            "rank_ic": np.linspace(0.01, 0.07, 7),
            "top_bottom_spread": [0.01] * 7,
        })
        quintiles = pd.DataFrame([
            {
                "date": str(date.date()),
                "factor": "low_volatility_60",
                "horizon_weeks": 4,
                "quintile": quintile,
                "mean_forward_return": quintile / 100,
            }
            for date in dates for quintile in range(1, 6)
        ])
        turnover = pd.DataFrame({
            "date": dates[1:],
            "factor": ["low_volatility_60"] * 6,
            "top_quintile_turnover": [0.5] * 6,
        })
        summaries = summarize_factor_diagnostics(
            monthly, quintiles, turnover, protocol=build_factor_protocol()
        )
        self.assertEqual(set(summaries["by_fold"]["fold"]), set(range(1, 8)))

    def test_workflow_stops_at_data_audit_before_factor_calculation(self):
        snapshots = self.root / "snapshots"
        cache = self.root / "cache"
        output = self.root / "output"
        snapshots.mkdir()
        pd.DataFrame({
            "code": ["600001", "600002"],
            "name": ["甲", "乙"],
            "sector": ["银行", "制造"],
        }).to_csv(snapshots / "2024-01-31.csv", index=False)
        summary = run_v2_factor_screening(
            start="2024-01-01",
            end="2024-01-31",
            cache_root=cache,
            snapshot_dir=snapshots,
            output_dir=output,
        )
        self.assertEqual(summary["status"], "data_audit_failed")
        self.assertFalse((output / "factor_values.csv").exists())
        self.assertTrue((output / "data_audit_summary.json").exists())
        self.assertTrue((output / "report.html").exists())

    def test_workflow_refuses_v1_paper_trading_output(self):
        with self.assertRaisesRegex(ValueError, "v1"):
            run_v2_factor_screening(
                start="2024-01-01",
                end="2024-01-31",
                cache_root=self.root / "cache",
                snapshot_dir=self.root / "snapshots",
                output_dir=(
                    Path(__file__).resolve().parent.parent
                    / "data" / "paper_trading" / "v2"
                ),
            )

    def test_report_table_wraps_long_gate_names_without_collapsing_status(self):
        decisions = pd.DataFrame({
            "factor": ["momentum_12_1"],
            "status": ["reject"],
            "failed_gates": [
                "mean_ic,median_ic,positive_ic_ratio,hac_t,raw_p,fdr,"
                "gross_spread,quintile_monotonicity"
            ],
        })
        rendered = _render_report(
            summary={
                "status": "factor_passed",
                "universe_union_codes": 1648,
                "snapshot_count": 65,
            },
            decisions=decisions,
        )
        self.assertIn("table-layout:fixed", rendered)
        self.assertIn("overflow-wrap:anywhere", rendered)
        self.assertIn("th:nth-child(2){width:18%}", rendered)


if __name__ == "__main__":
    unittest.main()
