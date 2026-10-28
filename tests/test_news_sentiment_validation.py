import unittest

import pandas as pd

from src.news_sentiment_validation import (
    _forward_open_returns,
    _rank_ic_analysis,
)


class NewsSentimentValidationTests(unittest.TestCase):
    def test_forward_return_uses_next_session_opens(self):
        dates = pd.to_datetime([
            "2026-01-02", "2026-01-05", "2026-01-09", "2026-01-12"
        ])
        frame = pd.DataFrame(
            {
                "open": [9.0, 10.0, 15.0, 12.0],
                "high": [9.0, 10.0, 15.0, 12.0],
                "low": [9.0, 10.0, 15.0, 12.0],
                "close": [9.0, 10.0, 15.0, 12.0],
            },
            index=dates,
        )
        result = _forward_open_returns(
            {"A": frame},
            pd.DatetimeIndex(["2026-01-02", "2026-01-09"]),
        )
        self.assertEqual(result.iloc[0]["entry_date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(result.iloc[0]["exit_date"], pd.Timestamp("2026-01-12"))
        self.assertAlmostEqual(result.iloc[0]["forward_return"], 0.2)

    def test_rank_ic_detects_monotonic_sentiment_signal(self):
        dates = [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-09")]
        combined_rows = []
        forward_rows = []
        sent_rows = []
        for date in dates:
            for index, code in enumerate(["A", "B", "C", "D", "E"]):
                combined_rows.append({
                    "date": date,
                    "code": code,
                    "score_sentiment": float(index),
                    "score_price": float(index % 2),
                })
                sent_rows.append({"date": date, "code": code})
                forward_rows.append({
                    "date": date,
                    "code": code,
                    "forward_return": float(index) / 100,
                })
        weekly, summary = _rank_ic_analysis(
            pd.DataFrame(combined_rows),
            pd.DataFrame(sent_rows),
            pd.DataFrame(forward_rows),
        )
        self.assertEqual(len(weekly), 2)
        self.assertAlmostEqual(
            summary["sentiment_rank_ic"]["mean"], 1.0
        )
        self.assertGreater(
            summary["top_bottom_weekly_spread"]["mean"], 0
        )


if __name__ == "__main__":
    unittest.main()
