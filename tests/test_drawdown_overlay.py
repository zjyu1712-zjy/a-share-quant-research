import unittest

import pandas as pd

from src.drawdown_overlay import DrawdownRiskPolicy
from src.execution_engine import simulate_portfolio


class DrawdownOverlayTests(unittest.TestCase):
    def test_drawdown_thresholds_and_three_day_recovery(self):
        policy = DrawdownRiskPolicy(
            initial_peak=100_000,
            warning_threshold=0.07,
            defensive_threshold=0.09,
            recovery_threshold=0.05,
            warning_equity_cap=0.40,
            defensive_equity_cap=0.20,
            recovery_days=3,
        )
        dates = pd.bdate_range("2025-01-02", periods=5)
        warning = policy.observe(dates[0], 93_000)
        defensive = policy.observe(dates[1], 91_000)
        recovery1 = policy.observe(dates[2], 95_100)
        recovery2 = policy.observe(dates[3], 96_000)
        normal = policy.observe(dates[4], 97_000)
        self.assertEqual(warning.state, "warning")
        self.assertEqual(warning.equity_cap, 0.40)
        self.assertEqual(defensive.state, "defensive")
        self.assertEqual(defensive.equity_cap, 0.20)
        self.assertAlmostEqual(recovery1.equity_cap, 0.4666667, places=6)
        self.assertAlmostEqual(recovery2.equity_cap, 0.7333333, places=6)
        self.assertEqual(normal.state, "normal")
        self.assertEqual(normal.equity_cap, 1.0)

    def test_new_high_resets_peak_and_drawdown(self):
        policy = DrawdownRiskPolicy(100_000, 0.07, 0.09, 0.05, 0.40, 0.20, 3)
        decision = policy.observe(pd.Timestamp("2025-01-02"), 105_000)
        self.assertEqual(decision.drawdown, 0.0)
        self.assertEqual(policy.peak_equity, 105_000)

    def test_risk_action_uses_previous_close_and_executes_next_open(self):
        dates = pd.to_datetime([
            "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"
        ])
        prices = pd.DataFrame({
            "open": [10.0, 10.0, 10.0, 9.0],
            "high": [10.0, 10.0, 10.0, 9.0],
            "low": [10.0, 10.0, 9.0, 9.0],
            "close": [10.0, 10.0, 9.0, 9.0],
            "volume": [10_000_000] * 4,
        }, index=dates)
        result = simulate_portfolio(
            selection_track=[{"date": "2025-01-02", "picks": ["600001"]}],
            price_data={"600001": prices},
            initial_capital=100_000,
            drawdown_policy=DrawdownRiskPolicy(
                100_000, 0.07, 0.09, 0.05, 0.40, 0.20, 3
            ),
        )
        event = result["risk_events"][0]
        self.assertEqual(event["observed_at_close"], "2025-01-06")
        self.assertEqual(event["executed_at_open"], "2025-01-07")
        self.assertEqual(event["state"], "defensive")
        final = result["equity_curve"][-1]
        self.assertLessEqual(final["actual_equity_ratio"], 0.21)

    def test_overlay_disabled_keeps_default_full_exposure_behavior(self):
        dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
        prices = pd.DataFrame({
            "open": [10.0, 10.0], "high": [10.0, 10.0],
            "low": [10.0, 10.0], "close": [10.0, 10.0],
            "volume": [10_000_000, 10_000_000],
        }, index=dates)
        result = simulate_portfolio(
            selection_track=[{"date": "2025-01-02", "picks": ["600001"]}],
            price_data={"600001": prices},
            initial_capital=100_000,
        )
        self.assertEqual(result["risk_events"], [])
        self.assertEqual(result["equity_curve"][-1]["target_equity_cap"], 1.0)


if __name__ == "__main__":
    unittest.main()
