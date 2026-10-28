import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sse_composite_universe import (
    build_snapshot,
    update_sse_snapshots,
)
from src.universe import PointInTimeUniverse, UniverseSnapshotError


class FakeAdapter:
    def __init__(self, basics, active_by_date, active_error=None):
        self._basics = pd.DataFrame(
            basics,
            columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
        )
        self._active_by_date = {
            date: pd.DataFrame(
                rows, columns=["code", "tradeStatus", "code_name"]
            )
            for date, rows in active_by_date.items()
        }
        self._active_error = active_error
        self.active_calls = 0
        self.industry_calls = 0

    def stock_basics(self):
        return self._basics.copy()

    def active_securities(self, as_of):
        self.active_calls += 1
        if self._active_error:
            raise self._active_error
        return self._active_by_date.get(
            str(as_of),
            pd.DataFrame(columns=["code", "tradeStatus", "code_name"]),
        ).copy()

    def industries(self, _as_of):
        self.industry_calls += 1
        return {"600001": "银行", "600003": "制造"}


class SSECompositeUniverseTests(unittest.TestCase):
    def _adapter(self):
        basics = [
            ("sh.600001", "甲", "2018-01-01", "", "1", "1"),
            ("sh.688001", "科创甲", "2018-01-01", "", "1", "1"),
            ("sh.900901", "B股甲", "2018-01-01", "", "1", "1"),
            ("sh.600002", "新股甲", "2024-08-01", "", "1", "1"),
            ("sh.600003", "ST风险", "2018-01-01", "", "1", "1"),
            ("sh.600004", "停牌甲", "2018-01-01", "", "1", "1"),
            ("sz.000001", "深市甲", "1991-01-01", "", "1", "1"),
            ("sh.000001", "上证指数", "1991-01-01", "", "2", "1"),
        ]
        active = [
            ("sh.600001", "1", "甲"),
            ("sh.688001", "1", "科创甲"),
            ("sh.900901", "1", "B股甲"),
            ("sh.600002", "1", "新股甲"),
            ("sh.600003", "1", "ST风险"),
            ("sh.600004", "0", "停牌甲"),
            ("sz.000001", "1", "深市甲"),
            ("sh.000001", "1", "上证指数"),
        ]
        return FakeAdapter(basics, {"2025-01-31": active})

    def test_snapshot_keeps_only_mature_tradable_main_board_a_shares(self):
        members, audit = build_snapshot("2025-01-31", self._adapter())
        self.assertEqual(members["code"].tolist(), ["600001"])
        self.assertEqual(members.iloc[0]["sector"], "银行")
        self.assertEqual(audit["excluded_star"], 1)
        self.assertEqual(audit["excluded_b_share"], 1)
        self.assertEqual(audit["excluded_listing_age"], 1)
        self.assertEqual(audit["excluded_st"], 1)
        self.assertEqual(audit["excluded_suspended"], 1)
        self.assertEqual(audit["final_members"], 1)

    def test_snapshot_fails_closed_without_as_of_membership(self):
        adapter = FakeAdapter(
            basics=[],
            active_by_date={},
            active_error=RuntimeError("no as-of data"),
        )
        with self.assertRaisesRegex(UniverseSnapshotError, "历史时点"):
            build_snapshot("2021-01-29", adapter)

    def test_update_writes_snapshot_and_audit_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            written = update_sse_snapshots(
                "2025-01-01",
                "2025-01-31",
                snapshot_dir=output,
                adapter=self._adapter(),
                signal_dates=["2025-01-31"],
            )
            self.assertEqual(written, [output / "2025-01-31.csv"])
            audit_path = output / "2025-01-31.audit.json"
            self.assertTrue(audit_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["effective_date"], "2025-01-31")
            store = PointInTimeUniverse(output, min_members=1)
            self.assertEqual(store.snapshot_for("2025-02-03").codes, {"600001"})

    def test_update_resumes_by_skipping_a_complete_existing_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first_adapter = self._adapter()
            update_sse_snapshots(
                "2025-01-01",
                "2025-01-31",
                snapshot_dir=output,
                adapter=first_adapter,
                signal_dates=["2025-01-31"],
            )

            resumed_adapter = self._adapter()
            progress = []
            written = update_sse_snapshots(
                "2025-01-01",
                "2025-01-31",
                snapshot_dir=output,
                adapter=resumed_adapter,
                progress=progress.append,
            )

            self.assertEqual(written, [])
            self.assertEqual(resumed_adapter.active_calls, 0)
            self.assertEqual(resumed_adapter.industry_calls, 0)
            self.assertTrue(any("跳过" in message for message in progress))

    def test_point_in_time_store_does_not_use_later_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame({
                "code": [f"600{i:03d}" for i in range(100)],
                "name": ["股票"] * 100,
                "sector": ["行业"] * 100,
            }).to_csv(directory / "2025-01-31.csv", index=False)
            with self.assertRaises(UniverseSnapshotError):
                PointInTimeUniverse(directory).snapshot_for("2025-01-30")


if __name__ == "__main__":
    unittest.main()
