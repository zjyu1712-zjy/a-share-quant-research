import subprocess
import unittest
from pathlib import Path

import pandas as pd


class V2FactorPortfolioValidationTests(unittest.TestCase):
    def test_candidate_metadata_is_frozen_to_three_portfolios(self):
        from src.v2_factor_portfolio_validation import candidate_metadata

        frame = candidate_metadata(100_000)
        self.assertEqual(
            frame["candidate"].tolist(),
            [
                "low_volatility_60",
                "sector_neutral_book_to_price_mrq",
                "lowvol50_value50",
            ],
        )
        self.assertEqual(frame["top_n"].unique().tolist(), [10])
        self.assertEqual(frame["rank_buffer"].unique().tolist(), [30])
        self.assertEqual(frame["benchmark"].unique().tolist(), ["SSE Composite / 000001"])
        self.assertEqual(frame["max_drawdown_limit_pct"].unique().tolist(), [10.0])
        blend = frame.set_index("candidate").loc["lowvol50_value50"]
        self.assertEqual(float(blend["low_volatility_weight"]), 0.5)
        self.assertEqual(float(blend["book_to_price_weight"]), 0.5)

    def test_factor_frames_use_only_frozen_passed_columns(self):
        from src.v2_factor_portfolio_validation import build_candidate_factor_frames

        factors = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31"],
                "code": ["600000", "600001"],
                "low_volatility_60": [1.0, 2.0],
                "sector_neutral_book_to_price_mrq": [3.0, 4.0],
                "momentum_12_1": [99.0, 99.0],
            }
        )
        closes = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31"],
                "code": ["600000", "600001"],
                "close": [10.0, 20.0],
            }
        )
        frames = build_candidate_factor_frames(factors, closes)
        self.assertEqual(set(frames), {
            "low_volatility_60",
            "sector_neutral_book_to_price_mrq",
            "lowvol50_value50",
        })
        self.assertNotIn("momentum_12_1", frames["lowvol50_value50"].columns)
        self.assertEqual(frames["lowvol50_value50"]["momentum"].tolist(), [1.0, 2.0])
        self.assertEqual(frames["lowvol50_value50"]["pullback"].tolist(), [3.0, 4.0])

    def test_strict_gate_enforces_ten_percent_drawdown_and_five_of_seven(self):
        from src.v2_factor_portfolio_validation import evaluate_candidate_gate

        folds = pd.DataFrame(
            {
                "test_total_return": [1, 1, 1, 1, 1, -1, -1],
                "test_excess_return": [1, 1, 1, 1, 1, -1, -1],
                "test_max_drawdown": [-9.0] * 7,
            }
        )
        passing = evaluate_candidate_gate(
            folds,
            {"cagr": 10.0, "excess_return": 0.01, "max_drawdown": -10.0},
        )
        self.assertTrue(passing["passed"])
        folds.loc[6, "test_max_drawdown"] = -10.01
        failing = evaluate_candidate_gate(
            folds,
            {"cagr": 10.0, "excess_return": 0.01, "max_drawdown": -10.0},
        )
        self.assertFalse(failing["passed"])

    def test_run_mac_exposes_isolated_v2_portfolio_entry(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [str(root / "run_mac.sh"), "help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("v2portfolio", result.stdout)

    def test_universe_initialization_starts_at_first_available_snapshot(self):
        from src.v2_factor_portfolio_validation import effective_universe_start

        available = [pd.Timestamp("2020-01-23"), pd.Timestamp("2020-02-28")]
        self.assertEqual(
            effective_universe_start("2020-01-01", available),
            "2020-01-23",
        )
        self.assertEqual(
            effective_universe_start("2020-02-01", available),
            "2020-02-01",
        )


if __name__ == "__main__":
    unittest.main()
