import unittest

import numpy as np
import pandas as pd

from src.walk_forward_attribution import (
    _benchmark_return,
    _compound,
    _open_forward_return,
    build_factor_ic,
)


class WalkForwardAttributionTests(unittest.TestCase):
    def test_compounds_window_returns(self):
        self.assertAlmostEqual(_compound([10.0, -10.0]), -1.0)

    def test_forward_return_starts_after_signal(self):
        index = pd.bdate_range("2024-01-01", periods=8)
        frame = pd.DataFrame({"open": np.arange(10.0, 18.0)}, index=index)
        result = _open_forward_return(frame, index[0], periods=5)
        self.assertAlmostEqual(result, (16.0 / 11.0 - 1) * 100)

    def test_factor_ic_recognizes_monotonic_signal(self):
        signal = pd.Timestamp("2024-01-05")
        dates = pd.bdate_range("2024-01-01", periods=60)
        factors = []
        prices = {}
        for number in range(1, 31):
            code = f"{number:06d}"
            values = np.full(len(dates), 100.0)
            entry = int(dates.searchsorted(signal, side="right"))
            values[entry + 5] = 100.0 + number
            prices[code] = pd.DataFrame({"open": values}, index=dates)
            factors.append({
                "date": signal,
                "code": code,
                "momentum": number,
                "pullback": number,
                "score_price": number,
            })
        _, summary = build_factor_ic(pd.DataFrame(factors), prices)
        row = summary[
            (summary["factor"] == "score_price")
            & (summary["horizon_weeks"] == 1)
        ].iloc[0]
        self.assertAlmostEqual(row["mean_rank_ic"], 1.0)

    def test_benchmark_uses_first_execution_open_and_end_close(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        frame = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 110],
        }, index=dates)
        result = _benchmark_return(frame, dates[1], str(dates[-1].date()))
        self.assertAlmostEqual(result, (110 / 101 - 1) * 100)


if __name__ == "__main__":
    unittest.main()
