import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.paper_trading import (
    initialize_tracker,
    load_events,
    record_signal,
    run_synthetic_dry_run,
    update_tracker,
)


def _plan(code: str = "000001") -> dict:
    return {
        "status": "pending",
        "signal_date": "2026-07-31",
        "market_data_as_of": "2026-07-31",
        "expected_execution_date": "2026-08-03",
        "positions": [{
            "code": code,
            "name": "测试股票",
            "sector": "测试行业",
            "signal_price": 10.0,
            "shares": 10_000,
        }],
        "target_weights": {code: 1.0},
        "backup_pool": [],
    }


class PaperTradingTests(unittest.TestCase):
    def test_initialization_is_idempotent_and_protocol_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            first = initialize_tracker(output, initial_cash=100_000)
            second = initialize_tracker(output, initial_cash=100_000)
            self.assertEqual(
                first["protocol_sha256"], second["protocol_sha256"]
            )
            self.assertEqual(
                len([
                    event for event in load_events(output)
                    if event["event_type"] == "initialize"
                ]),
                1,
            )
            with self.assertRaises(ValueError):
                initialize_tracker(output, initial_cash=200_000)

    def test_signal_rejects_stale_market_date_and_conflicting_rewrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            initialize_tracker(output)
            stale = _plan()
            stale["market_data_as_of"] = "2026-07-30"
            with self.assertRaises(ValueError):
                record_signal(stale, output)

            record_signal(_plan(), output)
            record_signal(_plan(), output)
            self.assertEqual(
                len([
                    event for event in load_events(output)
                    if event["event_type"] == "signal"
                ]),
                1,
            )
            with self.assertRaises(ValueError):
                record_signal(_plan("600000"), output)

    def test_synthetic_dry_run_is_unscored_and_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            result = run_synthetic_dry_run(output)
            status = result["status"]
            self.assertEqual(status["account_mode"], "synthetic_dry_run")
            self.assertEqual(status["scored_observation_count"], 0)
            self.assertEqual(status["rebalance_count"], 1)
            self.assertEqual(status["latest_mark_date"], "2026-08-04")
            self.assertTrue((output / "report.html").exists())
            self.assertTrue((output / "signals.csv").exists())
            self.assertTrue((output / "fills.csv").exists())
            self.assertTrue((output / "nav.csv").exists())
            self.assertGreater(status["execution_totals"]["total_fees"], 0)
            self.assertGreater(status["execution_totals"]["slippage_cost"], 0)

            # Replaying the same market data must not duplicate fills or NAV.
            before = len(load_events(output))
            dates = pd.to_datetime([
                "2026-07-31", "2026-08-03", "2026-08-04"
            ])
            stocks = {
                "000001": pd.DataFrame({
                    "open": [10.0, 10.1, 10.3],
                    "high": [10.2, 10.4, 10.6],
                    "low": [9.9, 10.0, 10.2],
                    "close": [10.0, 10.2, 10.5],
                    "volume": [5_000_000, 5_100_000, 5_200_000],
                }, index=dates),
                "600000": pd.DataFrame({
                    "open": [20.0, 19.8, 20.1],
                    "high": [20.2, 20.2, 20.4],
                    "low": [19.7, 19.6, 19.9],
                    "close": [20.0, 20.0, 20.2],
                    "volume": [4_000_000, 4_100_000, 4_200_000],
                }, index=dates),
            }
            benchmark = pd.DataFrame({
                "open": [4000.0, 4020.0, 4040.0],
                "high": [4010.0, 4050.0, 4070.0],
                "low": [3980.0, 4010.0, 4030.0],
                "close": [4005.0, 4040.0, 4060.0],
                "volume": [1, 1, 1],
            }, index=dates)
            replay = update_tracker(
                as_of="2026-08-04",
                output_dir=output,
                price_data=stocks,
                index_data=benchmark,
            )
            self.assertEqual(replay["rebalance_events_added"], 0)
            self.assertEqual(replay["mark_events_added"], 0)
            self.assertEqual(len(load_events(output)), before)

    def test_official_marks_are_scored_from_first_session_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            initialize_tracker(output)
            record_signal(_plan(), output)
            dates = pd.to_datetime([
                "2026-07-31", "2026-08-03", "2026-08-04"
            ])
            stocks = {
                "000001": pd.DataFrame({
                    "open": [10.0, 10.0, 10.0],
                    "high": [10.1, 10.1, 10.1],
                    "low": [9.9, 9.9, 9.9],
                    "close": [10.0, 10.0, 10.0],
                    "volume": [5_000_000, 5_000_000, 5_000_000],
                }, index=dates),
            }
            benchmark = pd.DataFrame({
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 101.0, 102.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 101.0, 102.0],
                "volume": [1, 1, 1],
            }, index=dates)
            status = update_tracker(
                as_of="2026-08-04",
                output_dir=output,
                price_data=stocks,
                index_data=benchmark,
            )["status"]
            self.assertEqual(status["scored_observation_count"], 2)
            self.assertAlmostEqual(status["benchmark_return_pct"], 2.0)
            nav = pd.read_csv(output / "nav.csv")
            self.assertEqual(nav["date"].tolist(), [
                "2026-08-03", "2026-08-04"
            ])
            self.assertTrue(nav["scored"].all())
            saved = json.loads((output / "status.json").read_text())
            self.assertEqual(saved["latest_mark_date"], "2026-08-04")


if __name__ == "__main__":
    unittest.main()
