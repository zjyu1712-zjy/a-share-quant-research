import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.concentration_filter import ConcentrationFilter
from src.database import init_db, save_news_batch
from src.dual_factor import load_sentiment_series
from src.execution_engine import generate_rebalance_dates, simulate_portfolio


def _prices(rows):
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")


class ExecutionEngineTests(unittest.TestCase):
    def test_period_end_is_real_trading_day(self):
        data = {
            "A": _prices([
                {"date": "2026-01-02", "open": 10, "high": 10, "low": 10, "close": 10},
                {"date": "2026-01-05", "open": 10, "high": 10, "low": 10, "close": 10},
                {"date": "2026-01-09", "open": 10, "high": 10, "low": 10, "close": 10},
            ])
        }
        dates = generate_rebalance_dates(
            data, "2026-01-01", "2026-01-09", frequency="weekly"
        )
        self.assertEqual(
            [d.strftime("%Y-%m-%d") for d in dates],
            ["2026-01-02", "2026-01-09"],
        )

    def test_signal_executes_at_next_session_open_with_real_lots(self):
        data = {
            "A": _prices([
                {"date": "2026-01-02", "open": 10, "high": 11, "low": 9, "close": 10},
                {"date": "2026-01-05", "open": 12, "high": 13, "low": 11, "close": 13},
                {"date": "2026-01-06", "open": 13, "high": 14, "low": 12, "close": 14},
            ])
        }
        result = simulate_portfolio(
            selection_track=[{"date": "2026-01-02", "picks": ["A"]}],
            price_data=data,
            initial_capital=10_000,
            commission=0,
            min_lot=100,
            end_date="2026-01-06",
        )

        trade = result["rebalance_log"][0]
        self.assertEqual(trade["date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(trade["buys"][0]["price"], 12)
        self.assertEqual(trade["buys"][0]["shares"], 800)
        self.assertEqual(result["equity_curve"][0]["equity"], 10_000)
        self.assertEqual(result["equity_curve"][1]["equity"], 10_800)

    def test_realistic_fees_and_slippage_are_charged(self):
        data = {
            "A": _prices([
                {"date": "2024-01-02", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100000},
                {"date": "2024-01-03", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100000},
                {"date": "2024-01-04", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100000},
                {"date": "2024-01-05", "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100000},
            ])
        }
        result = simulate_portfolio(
            selection_track=[
                {"date": "2024-01-02", "picks": ["A"]},
                {"date": "2024-01-03", "picks": []},
            ],
            price_data=data,
            initial_capital=10_000,
            commission=0.0003,
            min_commission=5,
            transfer_fee_rate=0.00001,
            stamp_duty_rate=0.0005,
            historical_stamp_duty_rate=0.001,
            slippage_bps=5,
            min_lot=100,
            end_date="2024-01-05",
        )
        stats = result["execution_stats"]
        self.assertEqual(stats["commission"], 10.0)
        self.assertAlmostEqual(stats["transfer_fee"], 0.18, places=2)
        self.assertAlmostEqual(stats["stamp_duty"], 4.5, places=2)
        self.assertAlmostEqual(stats["slippage_cost"], 9.0, places=2)
        self.assertLess(result["cash"], 10_000)

    def test_limit_up_lock_rejects_new_buy(self):
        rows = []
        for day in range(2, 8):
            rows.append({
                "date": f"2024-01-0{day}", "open": 10, "high": 10,
                "low": 10, "close": 10, "volume": 100000,
            })
        rows.append({
            "date": "2024-01-08", "open": 11, "high": 11,
            "low": 11, "close": 11, "volume": 100000,
        })
        result = simulate_portfolio(
            selection_track=[{"date": "2024-01-07", "picks": ["A"]}],
            price_data={"A": _prices(rows)},
            initial_capital=10_000,
            commission=0,
            min_lot=100,
            enforce_price_limits=True,
            end_date="2024-01-08",
        )
        self.assertEqual(result["positions"], {})
        reasons = [
            item["reason"]
            for item in result["rebalance_log"][0]["constraints"]
        ]
        self.assertIn("limit_up_locked", reasons)

    def test_volume_capacity_uses_prior_history(self):
        rows = []
        for day in range(2, 8):
            rows.append({
                "date": f"2024-01-0{day}", "open": 10, "high": 10,
                "low": 10, "close": 10, "volume": 1000,
            })
        rows.append({
            "date": "2024-01-08", "open": 10, "high": 10,
            "low": 10, "close": 10, "volume": 1_000_000,
        })
        result = simulate_portfolio(
            selection_track=[{"date": "2024-01-07", "picks": ["A"]}],
            price_data={"A": _prices(rows)},
            initial_capital=10_000,
            commission=0,
            min_lot=100,
            max_volume_participation=0.10,
            volume_lookback=5,
            end_date="2024-01-08",
        )
        self.assertEqual(result["positions"]["A"]["shares"], 100)
        self.assertGreater(result["execution_stats"]["constrained_orders"], 0)


class ConcentrationFilterTests(unittest.TestCase):
    def test_same_week_save_is_idempotent_and_apply_uses_signal_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "concentration.json"
            filt = ConcentrationFilter(
                lookback_weeks=4,
                max_selects=2,
                mode="exclude",
                state_path=state,
            )
            filt.save_history(["A"], "2026-07-10", persist=False)
            filt.save_history(["A"], "2026-07-10", persist=False)
            self.assertEqual(filt._history[("2026-07-06", "A")], 1)

            filt.save_history(["A"], "2026-07-17", persist=False)
            ranked = pd.DataFrame({"code": ["A", "B"], "score": [2.0, 1.0]})
            picks = filt.apply(
                current_picks=["A"],
                ranked_df=ranked,
                top_n=1,
                sectors={"A": "S1", "B": "S2"},
                rebalance_date="2026-07-17",
            )
            # 本周自己的旧记录不计入窗口，因此重跑结果保持稳定。
            self.assertEqual(picks, ["A"])


class NewsFailureIsolationTests(unittest.TestCase):
    def test_failed_analysis_is_not_inserted_and_legacy_failure_is_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "news.db"
            init_db(path)
            written = save_news_batch(
                [
                    {
                        "title": "测试公司正常新闻",
                        "source": "test",
                        "link": "https://example.com/ok",
                        "sentiment_score": 0.0,
                        "analysis_status": "success",
                        "reasoning": "常规中性信息",
                    },
                    {
                        "title": "测试公司失败新闻",
                        "source": "test",
                        "link": "https://example.com/fail",
                        "sentiment_score": None,
                        "analysis_status": "failed",
                        "reasoning": "API 调用失败，未完成分析",
                    },
                ],
                fetch_date="2026-07-25",
                db_path=path,
            )
            self.assertEqual(written, 1)

            conn = sqlite3.connect(path)
            conn.execute(
                """
                INSERT INTO news_archive
                    (fetch_date, title, source, sentiment_score,
                     sentiment_label, reasoning)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-07-24",
                    "测试公司旧失败记录",
                    "test",
                    0.0,
                    "中性",
                    "API 调用失败，未完成分析",
                ),
            )
            conn.commit()
            conn.close()

            series = load_sentiment_series(
                "测试公司", "2026-07-20", "2026-07-26", db_path=path
            )
            self.assertEqual(len(series), 1)
            self.assertEqual(float(series.iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
