import unittest

import pandas as pd

from src.tradability import TradabilityRules, screen_tradable_codes


def _price_frame(close, amount, *, drop_last=False):
    dates = pd.bdate_range("2025-01-06", periods=20)
    frame = pd.DataFrame({
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000_000,
        "amount": amount,
    }, index=dates)
    return frame.iloc[:-1] if drop_last else frame


class TradabilityTests(unittest.TestCase):
    def setUp(self):
        self.rules = TradabilityRules(
            min_average_amount=50_000_000,
            amount_lookback=20,
            top_n=10,
            initial_cash=100_000,
            min_lot=100,
            commission=0.0005,
            min_commission=5.0,
            transfer_fee_rate=0.00001,
            slippage_bps=5.0,
        )

    def test_screen_rejects_low_liquidity_suspended_and_unaffordable(self):
        snapshot = pd.DataFrame({
            "code": ["600001", "600002", "600003", "600004"],
            "name": ["正常", "低流动", "停牌", "高价"],
            "sector": ["行业"] * 4,
        })
        prices = {
            "600001": _price_frame(10.0, 60_000_000),
            "600002": _price_frame(10.0, 10_000_000),
            "600003": _price_frame(10.0, 60_000_000, drop_last=True),
            "600004": _price_frame(150.0, 60_000_000),
        }
        eligible, rows, audit = screen_tradable_codes(
            snapshot=snapshot,
            price_data=prices,
            signal_date="2025-01-31",
            rules=self.rules,
        )
        self.assertEqual(eligible, {"600001"})
        self.assertEqual(audit["excluded_low_liquidity"], 1)
        self.assertEqual(audit["excluded_suspended_or_missing_price"], 1)
        self.assertEqual(audit["excluded_unaffordable_lot"], 1)
        self.assertEqual(audit["final_eligible"], 1)
        self.assertEqual(len(rows), 4)

    def test_liquidity_uses_only_signal_date_and_prior_rows(self):
        frame = _price_frame(10.0, 60_000_000)
        future = pd.DataFrame({
            "open": [10.0], "high": [10.0], "low": [10.0],
            "close": [10.0], "volume": [1], "amount": [1],
        }, index=pd.to_datetime(["2025-02-03"]))
        prices = {"600001": pd.concat([frame, future])}
        snapshot = pd.DataFrame({
            "code": ["600001"], "name": ["正常"], "sector": ["行业"]
        })
        eligible, rows, _ = screen_tradable_codes(
            snapshot, prices, "2025-01-31", self.rules
        )
        self.assertEqual(eligible, {"600001"})
        self.assertEqual(rows.iloc[0]["amount_observations"], 20)

    def test_missing_amount_fails_closed(self):
        snapshot = pd.DataFrame({
            "code": ["600001"], "name": ["缺成交额"], "sector": ["行业"]
        })
        prices = {"600001": _price_frame(10.0, 60_000_000).drop(columns="amount")}
        eligible, rows, audit = screen_tradable_codes(
            snapshot, prices, "2025-01-31", self.rules
        )
        self.assertEqual(eligible, set())
        self.assertEqual(rows.iloc[0]["reason"], "missing_amount")
        self.assertEqual(audit["excluded_missing_amount"], 1)


if __name__ == "__main__":
    unittest.main()
