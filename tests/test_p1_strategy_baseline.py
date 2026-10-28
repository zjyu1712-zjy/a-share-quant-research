import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import StrategyConfig
from src.dual_factor import load_sentiment_for_pool
from src.universe import PointInTimeUniverse, UniverseSnapshotError


class BaselineConfigTests(unittest.TestCase):
    def test_baseline_overrides_legacy_hybrid_and_stop_values(self):
        env = {
            "BASELINE_MODE": "true",
            "BACKTEST_MODE": "hybrid",
            "SENTIMENT_WEIGHT": "0.4",
            "STOP_LOSS_PCT": "0.12",
            "TRAILING_STOP_PCT": "0.10",
            "TIME_STOP_MONTHS": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            config = StrategyConfig.from_env()
        self.assertEqual(config.backtest.mode, "price_only")
        self.assertEqual(config.scoring.sentiment_weight, 0.0)
        self.assertEqual(config.execution.stop_loss_pct, 0.0)
        self.assertEqual(config.execution.trailing_stop_pct, 0.0)
        self.assertEqual(config.execution.time_stop_months, 0)


class SentimentQualityTests(unittest.TestCase):
    def test_low_news_coverage_returns_empty_without_price_proxy(self):
        names = {"甲": "000001", "乙": "000002"}
        fake_prices = {
            code: pd.DataFrame({"close": range(100)})
            for code in names.values()
        }

        def fake_news(name, *_args, **_kwargs):
            if name == "甲":
                return pd.Series([0.2] * 5, index=pd.date_range("2024-01-01", periods=5))
            return pd.Series(dtype=float)

        with patch("src.dual_factor.load_sentiment_series", side_effect=fake_news):
            result = load_sentiment_for_pool(
                names,
                "2024-01-01",
                "2024-01-31",
                price_data=fake_prices,
                min_sqlite_days=3,
                min_pool_coverage=0.75,
                allow_price_proxy=False,
            )
        self.assertEqual(result, {})


class UniverseSnapshotTests(unittest.TestCase):
    def _write_snapshot(self, directory: Path, effective: str, codes: list[str]):
        pd.DataFrame({
            "code": codes,
            "name": [f"股票{i}" for i in range(len(codes))],
            "sector": ["行业"] * len(codes),
        }).to_csv(directory / f"{effective}.csv", index=False)

    def test_resolves_latest_snapshot_without_lookahead(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old_codes = [f"{i:06d}" for i in range(100)]
            new_codes = [f"{i:06d}" for i in range(50, 150)]
            self._write_snapshot(directory, "2020-01-01", old_codes)
            self._write_snapshot(directory, "2020-07-01", new_codes)
            store = PointInTimeUniverse(directory)

            self.assertEqual(
                store.snapshot_for("2020-06-30").effective_date,
                pd.Timestamp("2020-01-01"),
            )
            self.assertEqual(
                store.snapshot_for("2020-07-01").effective_date,
                pd.Timestamp("2020-07-01"),
            )

    def test_missing_prior_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_snapshot(
                directory, "2020-07-01", [f"{i:06d}" for i in range(100)]
            )
            with self.assertRaises(UniverseSnapshotError):
                PointInTimeUniverse(directory).snapshot_for("2020-01-01")


if __name__ == "__main__":
    unittest.main()
