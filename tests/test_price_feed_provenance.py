import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.price_feed import _check_cache_coverage, _standardize_columns


class PriceFeedProvenanceTests(unittest.TestCase):
    @staticmethod
    def _write_cache(root: Path, metadata: dict) -> Path:
        cache = root / "stock_000001.csv"
        pd.DataFrame(
            {"close": [10.0]},
            index=pd.to_datetime(["2025-05-12"]),
        ).to_csv(cache)
        cache.with_suffix(".coverage.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return cache

    def test_legacy_cache_without_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = self._write_cache(
                Path(temp_dir),
                {
                    "requested_start": "2020-01-01",
                    "requested_end": "2025-05-12",
                    "actual_end": "2025-05-12",
                },
            )
            self.assertFalse(
                _check_cache_coverage(
                    cache,
                    "2020-01-01",
                    "2025-05-12",
                    required_adjust="qfq",
                )
            )

    def test_qfq_cache_with_source_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = self._write_cache(
                Path(temp_dir),
                {
                    "schema_version": 2,
                    "source": "akshare_stock_zh_a_hist_tx",
                    "adjust": "qfq",
                    "requested_start": "2020-01-01",
                    "requested_end": "2025-05-12",
                    "actual_end": "2025-05-12",
                },
            )
            self.assertTrue(
                _check_cache_coverage(
                    cache,
                    "2020-01-01",
                    "2025-05-12",
                    required_adjust="qfq",
                )
            )

    def test_wrong_adjustment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = self._write_cache(
                Path(temp_dir),
                {
                    "schema_version": 2,
                    "source": "akshare_stock_zh_a_hist_tx",
                    "adjust": "none",
                    "requested_start": "2020-01-01",
                    "requested_end": "2025-05-12",
                    "actual_end": "2025-05-12",
                },
            )
            self.assertFalse(
                _check_cache_coverage(
                    cache,
                    "2020-01-01",
                    "2025-05-12",
                    required_adjust="qfq",
                )
            )

    def test_completed_request_accepts_terminal_stock_before_requested_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = self._write_cache(
                Path(temp_dir),
                {
                    "schema_version": 2,
                    "source": "akshare_stock_zh_a_hist_tx",
                    "adjust": "qfq",
                    "download_completed": True,
                    "requested_start": "2020-01-01",
                    "requested_end": "2025-05-12",
                    "actual_end": "2023-06-30",
                },
            )
            self.assertTrue(
                _check_cache_coverage(
                    cache,
                    "2020-01-01",
                    "2025-05-12",
                    required_adjust="qfq",
                )
            )

    def test_standardized_price_data_preserves_amount(self):
        raw = pd.DataFrame({
            "日期": ["2025-01-02"],
            "开盘": [10.0],
            "最高": [11.0],
            "最低": [9.0],
            "收盘": [10.5],
            "成交量": [1_000_000],
            "成交额": [10_500_000],
        })
        result = _standardize_columns(raw)
        self.assertEqual(
            result.loc[pd.Timestamp("2025-01-02"), "amount"],
            10_500_000,
        )
        self.assertEqual(result.attrs["amount_source"], "reported")

    def test_v1_accepts_schema2_but_v2_amount_request_rejects_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = self._write_cache(
                Path(temp_dir),
                {
                    "schema_version": 2,
                    "source": "akshare_stock_zh_a_hist_tx",
                    "adjust": "qfq",
                    "requested_start": "2020-01-01",
                    "requested_end": "2025-05-12",
                    "actual_end": "2025-05-12",
                },
            )
            self.assertTrue(_check_cache_coverage(
                cache, "2020-01-01", "2025-05-12", required_adjust="qfq"
            ))
            self.assertFalse(_check_cache_coverage(
                cache,
                "2020-01-01",
                "2025-05-12",
                required_adjust="qfq",
                require_amount=True,
            ))


if __name__ == "__main__":
    unittest.main()
