import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.statement_factor_screening import (
    FACTOR_NAMES,
    _normalize_report_rows,
    build_statement_factors,
    profile_quality,
    profile_source_dates,
    screen_factors,
)


class StatementFactorScreeningTests(unittest.TestCase):
    def test_raw_report_mapping_keeps_update_and_notice_dates(self):
        rows = [{
            "SECURITY_CODE": "600000",
            "SECURITY_NAME_ABBR": "浦发银行",
            "SECURITY_TYPE": "A股",
            "REPORTDATE": "2023-12-31",
            "UPDATE_DATE": "2025-03-29",
            "NOTICE_DATE": "2024-04-30",
            "BASIC_EPS": 1.07,
            "TOTAL_OPERATE_INCOME": 100.0,
            "PARENT_NETPROFIT": 20.0,
            "WEIGHTAVG_ROE": 5.21,
            "YSTZ": -8.0,
            "SJLTZ": -28.0,
            "MGJYXJJE": 13.23,
            "XSMLL": None,
        }]
        frame = _normalize_report_rows(rows)
        self.assertEqual(frame.iloc[0]["code"], "600000")
        self.assertEqual(
            frame.iloc[0]["update_date"],
            pd.Timestamp("2025-03-29"),
        )
        self.assertEqual(
            frame.iloc[0]["notice_date"],
            pd.Timestamp("2024-04-30"),
        )

    def test_factor_builder_excludes_future_announcement(self):
        codes = {f"{number:06d}" for number in range(1, 61)}
        rows = []
        for number, code in enumerate(sorted(codes), start=1):
            for year in [2021, 2022]:
                rows.append({
                    "code": code,
                    "name": code,
                    "report_date": pd.Timestamp(f"{year}-12-31"),
                    "update_date": pd.Timestamp(f"{year + 1}-04-20"),
                    "notice_date": pd.Timestamp(f"{year + 1}-04-15"),
                    "eps": 0.5 + number / 100,
                    "revenue": 100 + number,
                    "net_profit": 10 + number / 10,
                    "roe_pct": 5 + number / 10 + year - 2021,
                    "revenue_yoy_pct": 2 + number / 10,
                    "net_profit_yoy_pct": 3 + number / 10,
                    "operating_cashflow_per_share": 0.6,
                    "gross_margin_pct": 20 + number / 10,
                })
        # 未来公告值很极端，但在信号日不可见。
        rows.append({
            **rows[-1],
            "code": "000001",
            "report_date": pd.Timestamp("2023-12-31"),
            "update_date": pd.Timestamp("2025-01-01"),
            "notice_date": pd.Timestamp("2024-04-15"),
            "roe_pct": 999,
        })
        reports = pd.DataFrame(rows)

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(
                    codes=codes,
                    code_to_sector={
                        code: (
                            "sector_a"
                            if int(code) <= 30 else "sector_b"
                        )
                        for code in codes
                    },
                )

        signal = pd.Timestamp("2023-06-30")
        factors = build_statement_factors(
            reports=reports,
            rebalance_dates=pd.DatetimeIndex([signal]),
            universe_store=FakeUniverse(),
        )
        self.assertEqual(factors["code"].nunique(), 60)
        self.assertTrue(set(FACTOR_NAMES).issubset(factors.columns))
        self.assertTrue(
            (factors["source_availability_date"] <= signal).all()
        )
        self.assertLess(
            factors.loc[
                factors["code"] == "000001",
                "quality_roe_annual_raw",
            ].iloc[0],
            999,
        )
        sector_medians = factors.groupby("sector")[
            "quality_roe_annual"
        ].median()
        np.testing.assert_allclose(sector_medians, 0, atol=1e-12)

    def test_source_date_audit_rejects_update_as_availability_proxy(self):
        reports = pd.DataFrame({
            "code": ["600000", "000001"],
            "report_date": pd.to_datetime(["2023-12-31", "2023-12-31"]),
            "notice_date": pd.to_datetime(["2024-04-30", "2024-03-15"]),
            "update_date": pd.to_datetime(["2025-03-29", "2025-03-15"]),
        })
        audit = profile_source_dates(reports)
        self.assertEqual(audit["report_code_duplicates"], 0)
        self.assertEqual(audit["notice_before_report"], 0)
        self.assertEqual(audit["availability_field"], "NOTICE_DATE")
        self.assertGreater(
            audit["median_update_after_notice_days"],
            300,
        )

    def test_quality_uses_full_universe_denominator(self):
        codes = {f"{number:06d}" for number in range(1, 101)}
        rows = []
        for number in range(60):
            row = {
                "date": pd.Timestamp("2023-06-30"),
                "code": f"{number:06d}",
                "source_update_date": pd.Timestamp("2023-04-30"),
                "source_availability_date": pd.Timestamp("2023-04-15"),
            }
            row.update({factor: 1.0 for factor in FACTOR_NAMES})
            rows.append(row)

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(codes=codes)

        quality, summary = profile_quality(
            pd.DataFrame(rows),
            FakeUniverse(),
        )
        self.assertTrue(summary["passed"])
        self.assertTrue((quality["coverage_ratio"] == 0.60).all())

    def test_screen_rule_rejects_low_coverage(self):
        factor = FACTOR_NAMES[0]
        aggregate = pd.DataFrame([
            {
                "factor": factor,
                "horizon_weeks": 4,
                "mean_rank_ic": 0.03,
                "positive_ic_ratio": 0.60,
                "hac_t_stat": 2.0,
                "mean_top_bottom_spread_pct": 1.0,
            },
            {
                "factor": factor,
                "horizon_weeks": 8,
                "mean_rank_ic": 0.02,
                "positive_ic_ratio": 0.60,
                "hac_t_stat": 2.0,
                "mean_top_bottom_spread_pct": 1.0,
            },
        ])
        folds = pd.DataFrame({
            "fold": range(1, 8),
            "factor": factor,
            "horizon_weeks": 4,
            "mean_rank_ic": [0.02] * 5 + [-0.01] * 2,
        })
        quality = pd.DataFrame({
            "factor": factor,
            "coverage_ratio": [0.8] * 10,
        })
        self.assertEqual(
            screen_factors(aggregate, folds, quality).iloc[0]["status"],
            "promising",
        )
        quality["coverage_ratio"] = 0.5
        self.assertEqual(
            screen_factors(aggregate, folds, quality).iloc[0]["status"],
            "rejected",
        )


if __name__ == "__main__":
    unittest.main()
