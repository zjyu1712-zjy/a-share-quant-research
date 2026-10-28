import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.factor_reset_screening import (
    FACTOR_NAMES,
    REFERENCE_FACTOR,
    add_forward_returns,
    audit_price_provenance,
    benjamini_hochberg,
    build_protocol,
    build_reset_factors,
    calculate_top_quintile_turnover,
)


class _Snapshot:
    def __init__(self, codes):
        self.codes = set(codes)
        self.code_to_sector = {
            code: ("A" if index % 2 == 0 else "B")
            for index, code in enumerate(codes)
        }


class _Universe:
    def __init__(self, codes):
        self.snapshot = _Snapshot(codes)

    def snapshot_for(self, _date):
        return self.snapshot


def _prices(multiplier=1.0, periods=330):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = (100 + np.arange(periods) * 0.05) * multiplier
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 1_000_000),
        },
        index=dates,
    )


class FactorResetScreeningTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_and_index_preserving(self):
        values = pd.Series([0.01, 0.04, 0.03, np.nan], index=list("abcd"))
        adjusted = benjamini_hochberg(values)
        self.assertEqual(list(adjusted.index), list(values.index))
        self.assertTrue(np.isnan(adjusted["d"]))
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertLessEqual(adjusted["c"], adjusted["b"])
        self.assertTrue((adjusted.dropna() >= values.dropna()).all())

    def test_forward_return_starts_after_signal_open(self):
        frame = _prices(periods=100)
        signal = frame.index[20]
        factors = pd.DataFrame({"date": [signal], "code": ["000001"]})
        result = add_forward_returns(factors, {"000001": frame})
        entry = frame.index.get_loc(signal) + 1
        expected = (
            frame.iloc[entry + 20]["open"] / frame.iloc[entry]["open"] - 1
        ) * 100
        self.assertTrue(np.isclose(result.loc[0, "forward_4w_pct"], expected))

    def test_factor_values_do_not_change_when_future_prices_change(self):
        codes = [f"{index:06d}" for index in range(100)]
        base = {
            code: _prices(1 + index / 1000)
            for index, code in enumerate(codes)
        }
        index = _prices()
        signal = base[codes[0]].index[300]
        universe = _Universe(codes)
        before = build_reset_factors(
            price_data=base,
            index_data=index,
            rebalance_dates=pd.DatetimeIndex([signal]),
            universe_store=universe,
        ).sort_values("code")
        changed = {code: frame.copy() for code, frame in base.items()}
        for frame in changed.values():
            frame.loc[
                frame.index > signal, ["open", "high", "low", "close"]
            ] *= 10
        after = build_reset_factors(
            price_data=changed,
            index_data=index,
            rebalance_dates=pd.DatetimeIndex([signal]),
            universe_store=universe,
        ).sort_values("code")
        pd.testing.assert_frame_equal(
            before[FACTOR_NAMES + [REFERENCE_FACTOR]].reset_index(drop=True),
            after[FACTOR_NAMES + [REFERENCE_FACTOR]].reset_index(drop=True),
        )

    def test_sector_neutralized_factor_medians_are_zero(self):
        codes = [f"{index:06d}" for index in range(100)]
        data = {
            code: _prices(1 + index / 500, periods=330)
            for index, code in enumerate(codes)
        }
        signal = next(iter(data.values())).index[300]
        factors = build_reset_factors(
            price_data=data,
            index_data=_prices(periods=330),
            rebalance_dates=pd.DatetimeIndex([signal]),
            universe_store=_Universe(codes),
        )
        for factor in FACTOR_NAMES + [REFERENCE_FACTOR]:
            medians = factors.groupby("sector")[factor].median().dropna()
            self.assertTrue(
                np.allclose(medians.to_numpy(), 0.0, atol=1e-12)
            )

    def test_turnover_is_zero_for_unchanged_top_group(self):
        rows = []
        for date in pd.to_datetime(["2024-01-31", "2024-02-29"]):
            for index in range(100):
                rows.append({
                    "date": date,
                    "code": f"{index:06d}",
                    **{factor: float(index) for factor in FACTOR_NAMES},
                })
        result = calculate_top_quintile_turnover(pd.DataFrame(rows))
        self.assertTrue((result["top_quintile_turnover"] == 0).all())

    def test_protocol_hash_is_stable_and_marks_future_boundary(self):
        first = build_protocol(
            history_start="2020-01-01",
            research_start="2021-01-01",
            research_end="2025-05-12",
        )
        second = build_protocol(
            history_start="2020-01-01",
            research_start="2021-01-01",
            research_end="2025-05-12",
        )
        self.assertEqual(first["protocol_hash"], second["protocol_hash"])
        self.assertEqual(first["future_holdout_start"], "2026-08-03")
        self.assertEqual(len(first["candidate_directions"]), 6)

    def test_price_provenance_audit_rejects_unknown_adjustment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "stock_000001.coverage.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "source": "test",
                    "adjust": "none",
                }),
                encoding="utf-8",
            )
            audit = audit_price_provenance(
                codes=["000001"], cache_dir=root
            )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["failures"][0]["reason"], "unknown_source,not_qfq"
        )


if __name__ == "__main__":
    unittest.main()
