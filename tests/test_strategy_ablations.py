import unittest

import pandas as pd

from src.execution_engine import simulate_portfolio
from src.indicators import compute_target_weights, select_with_rank_buffer


def _price_frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        },
        index=dates,
    )


class RankBufferTests(unittest.TestCase):
    def test_retains_old_holdings_inside_buffer_then_fills_by_rank(self):
        ranked = pd.DataFrame(
            {
                "code": ["A", "B", "C", "D", "E", "F"],
                "score": [6, 5, 4, 3, 2, 1],
            }
        )
        sectors = {code: code for code in ranked["code"]}
        picks = select_with_rank_buffer(
            ranked=ranked,
            top_n=3,
            sectors=sectors,
            previous_picks=["E", "F", "X"],
            rank_buffer=3,
            sector_enabled=True,
        )
        self.assertEqual(picks, ["E", "F", "A"])

    def test_zero_buffer_matches_current_rank(self):
        ranked = pd.DataFrame(
            {"code": ["A", "B", "C"], "score": [3, 2, 1]}
        )
        picks = select_with_rank_buffer(
            ranked=ranked,
            top_n=2,
            sectors={"A": "S1", "B": "S2", "C": "S3"},
            previous_picks=["C"],
            rank_buffer=0,
        )
        self.assertEqual(picks, ["A", "B"])


class RiskWeightTests(unittest.TestCase):
    def test_inverse_volatility_gives_lower_vol_stock_more_weight(self):
        low = [100.0]
        high = [100.0]
        for i in range(80):
            low.append(low[-1] * (1 + (0.01 if i % 2 == 0 else -0.01)))
            high.append(high[-1] * (1 + (0.05 if i % 2 == 0 else -0.05)))
        data = {"LOW": _price_frame(low), "HIGH": _price_frame(high)}
        weights = compute_target_weights(
            ["LOW", "HIGH"],
            data,
            as_of=data["LOW"].index[-1],
            method="inverse_volatility",
            volatility_window=60,
            max_weight=0.70,
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["LOW"], weights["HIGH"])
        self.assertLessEqual(weights["LOW"], 0.70 + 1e-12)

    def test_execution_uses_target_weights_for_share_counts(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        frame = pd.DataFrame(
            {
                "open": [10.0, 10.0],
                "high": [10.0, 10.0],
                "low": [10.0, 10.0],
                "close": [10.0, 10.0],
                "volume": [1_000_000, 1_000_000],
            },
            index=dates,
        )
        result = simulate_portfolio(
            selection_track=[{
                "date": "2026-01-02",
                "picks": ["A", "B"],
                "target_weights": {"A": 0.7, "B": 0.3},
            }],
            price_data={"A": frame, "B": frame.copy()},
            initial_capital=10_000,
            commission=0,
            min_lot=100,
            end_date="2026-01-05",
        )
        self.assertEqual(result["positions"]["A"]["shares"], 700)
        self.assertEqual(result["positions"]["B"]["shares"], 300)


if __name__ == "__main__":
    unittest.main()
