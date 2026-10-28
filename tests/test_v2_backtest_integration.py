import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig
from src.v2_monthly_strategy import V2MonthlyBacktestEngine, build_v2_config


class FakeUniverse:
    def __init__(self, codes):
        self._codes = set(codes)
        self.available_dates = [pd.Timestamp("2025-01-31")]

    def snapshot_for(self, _date):
        ordered = sorted(self._codes)
        return SimpleNamespace(
            codes=set(ordered),
            members=pd.DataFrame({
                "code": ordered,
                "name": ["股票"] * len(ordered),
                "sector": ["行业"] * len(ordered),
            }),
            code_to_sector={code: "行业" for code in ordered},
        )


def factor_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31", "2025-01-31"]),
        "code": ["600001", "600002"],
        "score": [2.0, 1.0],
    })


def price_frame(amount):
    dates = pd.bdate_range("2025-01-06", periods=20)
    return pd.DataFrame({
        "open": 10.0,
        "high": 10.0,
        "low": 10.0,
        "close": 10.0,
        "volume": 1_000_000,
        "amount": amount,
    }, index=dates)


class V2BacktestIntegrationTests(unittest.TestCase):
    def test_base_engine_still_uses_snapshot_codes_only(self):
        engine = BacktestEngine(StrategyConfig())
        engine.universe_store = FakeUniverse({"600001"})
        result = engine.filter_factor_frame_for_date(
            "2025-01-31", factor_frame()
        )
        self.assertEqual(result["code"].tolist(), ["600001"])

    def test_v2_engine_applies_tradability_after_membership(self):
        engine = V2MonthlyBacktestEngine(build_v2_config())
        engine.universe_store = FakeUniverse({"600001", "600002"})
        engine.price_data = {
            "600001": price_frame(60_000_000),
            "600002": price_frame(10_000_000),
        }
        result = engine.filter_factor_frame_for_date(
            "2025-01-31", factor_frame()
        )
        self.assertEqual(result["code"].tolist(), ["600001"])
        self.assertEqual(engine.universe_audit_rows[-1]["final_eligible"], 1)
        self.assertEqual(
            engine.universe_exclusion_rows.iloc[1]["reason"],
            "low_liquidity",
        )

    def test_sse_pool_loader_requires_amount_without_changing_hs300(self):
        config = build_v2_config()
        engine = BacktestEngine(config)
        fake_members = pd.DataFrame({
            "code": ["600001"], "name": ["股票"], "sector": ["行业"]
        })
        fake_store = SimpleNamespace(union_for_range=lambda *_: fake_members)
        with patch("src.backtest_engine.PointInTimeUniverse", return_value=fake_store), patch(
            "src.backtest_engine.load_pool_data", return_value={}
        ) as load_pool:
            engine.load_price_data("2025-01-01", "2025-01-31")
        self.assertTrue(load_pool.call_args.kwargs["require_amount"])

        hs300 = StrategyConfig()
        hs300_engine = BacktestEngine(hs300)
        with patch("src.backtest_engine.PointInTimeUniverse", return_value=fake_store), patch(
            "src.backtest_engine.load_pool_data", return_value={}
        ) as load_pool:
            hs300_engine.load_price_data("2025-01-01", "2025-01-31")
        self.assertFalse(load_pool.call_args.kwargs["require_amount"])

    def test_only_v2_builds_drawdown_policy(self):
        self.assertIsNone(BacktestEngine(StrategyConfig()).build_drawdown_policy())
        policy = V2MonthlyBacktestEngine(build_v2_config()).build_drawdown_policy()
        self.assertEqual(policy.warning_threshold, 0.07)
        self.assertEqual(policy.defensive_threshold, 0.09)
        self.assertEqual(policy.recovery_threshold, 0.05)


if __name__ == "__main__":
    unittest.main()
