import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.v2_factor_data import (
    audit_factor_cache,
    bao_code,
    build_factor_protocol,
    cache_pair_valid,
    discover_universe_union,
    ensure_factor_protocol,
    normalize_price_rows,
    normalize_valuation_rows,
    update_factor_cache,
    write_cache_pair,
)


START = "2019-01-01"
END = "2025-05-12"


class FakeBaoStockAdapter:
    def __init__(self, owner, failing=None):
        self.owner = owner
        self.failing = set(failing or [])
        self.price_calls = 0
        self.valuation_calls = 0

    def query_price(self, code, start, end):
        self.price_calls += 1
        if code in self.failing:
            raise RuntimeError("synthetic failure")
        return self.owner._sample_price_frame(code)

    def query_valuation(self, code, start, end):
        self.valuation_calls += 1
        if code in self.failing:
            raise RuntimeError("synthetic failure")
        return pd.DataFrame({
            "date": pd.to_datetime(["2019-01-02", "2025-05-12"]),
            "code": [code, code],
            "pbMRQ": [1.0, 1.2],
        })


class V2FactorDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _sample_price_frame(code="600001"):
        return pd.DataFrame({
            "date": pd.to_datetime(["2019-01-02", "2025-05-12"]),
            "code": [code, code],
            "open": [10.0, 12.0],
            "high": [11.0, 13.0],
            "low": [9.0, 11.0],
            "close": [10.5, 12.5],
            "volume": [1_000.0, 2_000.0],
            "amount": [10_500.0, 25_000.0],
            "tradestatus": [1, 1],
        })

    @staticmethod
    def _price_metadata(code="600001"):
        return {
            "schema_version": 1,
            "kind": "price",
            "code": code,
            "source": "baostock_query_history_k_data_plus",
            "adjustflag": "2",
            "requested_start": "2019-01-01",
            "requested_end": "2025-05-12",
            "amount_source": "baostock_reported",
        }

    @staticmethod
    def _sample_valuation_frame(code="600001"):
        return pd.DataFrame({
            "date": pd.to_datetime(["2019-01-02", "2025-05-12"]),
            "code": [code, code],
            "pbMRQ": [1.0, 1.2],
        })

    @staticmethod
    def _valuation_metadata(code="600001"):
        return {
            "schema_version": 1,
            "kind": "valuation",
            "code": code,
            "source": "baostock_query_history_k_data_plus",
            "adjustflag": "3",
            "requested_start": START,
            "requested_end": END,
            "amount_source": None,
        }

    def _cache_dirs(self):
        return {
            "price_dir": self.root / "prices",
            "valuation_dir": self.root / "valuation",
            "audit_dir": self.root / "audit",
        }

    def test_union_discovery_uses_all_snapshots_and_normalizes_codes(self):
        pd.DataFrame({"code": ["600001", "600002"]}).to_csv(
            self.root / "2020-01-31.csv", index=False
        )
        pd.DataFrame({"code": ["600002", "600003"]}).to_csv(
            self.root / "2020-02-28.csv", index=False
        )
        self.assertEqual(
            discover_universe_union(self.root),
            ["600001", "600002", "600003"],
        )

    def test_protocol_freezes_sources_dates_factors_and_gates(self):
        protocol = build_factor_protocol()
        self.assertEqual(protocol["universe"]["expected_union_codes"], 1648)
        self.assertEqual(
            protocol["cache_window"],
            {"start": "2019-01-01", "end": "2025-05-12"},
        )
        self.assertEqual(
            protocol["data_sources"]["price"]["adjustflag"], "2"
        )
        self.assertEqual(
            set(protocol["factors"]),
            {
                "low_volatility_60",
                "sector_neutral_book_to_price_mrq",
                "momentum_12_1",
            },
        )
        self.assertEqual(len(protocol["fixed_windows"]), 7)
        path, digest = ensure_factor_protocol(self.root)
        self.assertTrue(path.exists())
        self.assertEqual(len(digest), 64)

    def test_existing_protocol_cannot_be_silently_rewritten(self):
        ensure_factor_protocol(self.root)
        path = self.root / "research_protocol.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cache_window"]["end"] = "2025-06-01"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "协议"):
            ensure_factor_protocol(self.root)

    def test_price_normalizer_keeps_reported_amount_and_trade_status(self):
        fields = [
            "date", "code", "open", "high", "low", "close",
            "volume", "amount", "tradestatus",
        ]
        rows = [[
            "2020-01-02", "sh.600001", "10", "11", "9", "10.5",
            "1000", "10500", "1",
        ]]
        frame = normalize_price_rows(rows, fields, "600001")
        self.assertEqual(frame.columns.tolist(), fields)
        self.assertEqual(frame.loc[0, "code"], "600001")
        self.assertEqual(frame.loc[0, "amount"], 10500.0)
        self.assertEqual(frame.loc[0, "tradestatus"], 1)

    def test_price_normalizer_never_estimates_missing_amount(self):
        fields = [
            "date", "code", "open", "high", "low", "close",
            "volume", "amount", "tradestatus",
        ]
        rows = [[
            "2020-01-02", "sh.600001", "10", "11", "9", "10.5",
            "1000", "", "1",
        ]]
        frame = normalize_price_rows(rows, fields, "600001")
        self.assertTrue(pd.isna(frame.loc[0, "amount"]))

    def test_valuation_normalizer_preserves_historical_pb_and_invalids(self):
        frame = normalize_valuation_rows(
            [
                ["2020-01-02", "sh.600001", "1.25"],
                ["2020-01-03", "sh.600001", ""],
            ],
            ["date", "code", "pbMRQ"],
            "600001",
        )
        self.assertEqual(frame.loc[0, "pbMRQ"], 1.25)
        self.assertTrue(pd.isna(frame.loc[1, "pbMRQ"]))
        self.assertEqual(bao_code("600001"), "sh.600001")

    def test_complete_cache_pair_is_valid_and_corruption_is_rejected(self):
        path = self.root / "price_600001.csv"
        write_cache_pair(
            self._sample_price_frame(), path, self._price_metadata()
        )
        self.assertTrue(cache_pair_valid(
            path,
            code="600001",
            start="2019-01-01",
            end="2025-05-12",
            kind="price",
        ))
        with path.open("a", encoding="utf-8") as handle:
            handle.write("corruption")
        self.assertFalse(cache_pair_valid(
            path,
            code="600001",
            start="2019-01-01",
            end="2025-05-12",
            kind="price",
        ))

    def test_price_cache_without_reported_amount_source_is_rejected(self):
        path = self.root / "price_600001.csv"
        metadata = self._price_metadata()
        metadata["amount_source"] = "estimated"
        write_cache_pair(self._sample_price_frame(), path, metadata)
        self.assertFalse(cache_pair_valid(
            path,
            code="600001",
            start="2019-01-01",
            end="2025-05-12",
            kind="price",
        ))

    def test_update_skips_only_two_valid_cache_pairs(self):
        adapter = FakeBaoStockAdapter(self)
        first = update_factor_cache(
            codes=["600001", "600002"],
            start=START,
            end=END,
            adapter=adapter,
            **self._cache_dirs(),
        )
        self.assertEqual(first["complete"], 2)
        resumed = FakeBaoStockAdapter(self)
        second = update_factor_cache(
            codes=["600001", "600002"],
            start=START,
            end=END,
            adapter=resumed,
            **self._cache_dirs(),
        )
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(resumed.price_calls, 0)
        self.assertEqual(resumed.valuation_calls, 0)

    def test_failure_is_recorded_without_removing_completed_stocks(self):
        adapter = FakeBaoStockAdapter(self, failing={"600002"})
        summary = update_factor_cache(
            codes=["600001", "600002"],
            start=START,
            end=END,
            adapter=adapter,
            **self._cache_dirs(),
        )
        self.assertEqual(summary["complete"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertTrue(
            (self.root / "audit" / "download_failures.csv").exists()
        )
        self.assertTrue(
            (self.root / "prices" / "price_600001.csv").exists()
        )

    def test_readiness_rejects_valid_257_code_subset(self):
        expected = [f"6{number:05d}" for number in range(300)]
        directories = self._cache_dirs()
        for code in expected[:257]:
            write_cache_pair(
                self._sample_price_frame(code),
                directories["price_dir"] / f"price_{code}.csv",
                self._price_metadata(code),
            )
            write_cache_pair(
                self._sample_valuation_frame(code),
                directories["valuation_dir"] / f"valuation_{code}.csv",
                self._valuation_metadata(code),
            )
        rows, summary = audit_factor_cache(
            codes=expected,
            start=START,
            end=END,
            **directories,
        )
        self.assertEqual(len(rows), 300)
        self.assertEqual(summary["complete_codes"], 257)
        self.assertEqual(summary["unattempted_codes"], 43)
        self.assertFalse(summary["ready_for_screening"])


if __name__ == "__main__":
    unittest.main()
