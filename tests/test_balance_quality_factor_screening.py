import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.balance_quality_factor_screening import (
    FACTOR_NAMES,
    _normalize_balance_rows,
    _normalize_cashflow_rows,
    build_balance_quality_factors,
    join_statement_sources,
    profile_factor_quality,
)


class BalanceQualityFactorScreeningTests(unittest.TestCase):
    def test_normalizers_keep_notice_dates_and_fields(self):
        balance = _normalize_balance_rows([{
            "SECURITY_CODE": "000001",
            "SECURITY_NAME_ABBR": "样本",
            "REPORT_DATE": "2023-12-31",
            "NOTICE_DATE": "2024-03-15",
            "TOTAL_ASSETS": 100,
            "TOTAL_LIABILITIES": 40,
            "MONETARYFUNDS": 10,
            "ACCOUNTS_RECE": 12,
            "INVENTORY": 8,
            "ACCOUNTS_PAYABLE": 7,
            "ADVANCE_RECEIVABLES": 3,
        }])
        cashflow = _normalize_cashflow_rows([{
            "SECURITY_CODE": "000001",
            "SECURITY_NAME_ABBR": "样本",
            "REPORT_DATE": "2023-12-31",
            "NOTICE_DATE": "2024-03-20",
            "NETCASH_OPERATE": 15,
        }])
        self.assertEqual(balance.iloc[0]["total_assets"], 100)
        self.assertEqual(
            balance.iloc[0]["balance_notice_date"],
            pd.Timestamp("2024-03-15"),
        )
        self.assertEqual(cashflow.iloc[0]["operating_cashflow"], 15)

    def test_join_uses_latest_component_notice_date(self):
        income = pd.DataFrame({
            "code": ["000001"],
            "report_date": pd.to_datetime(["2023-12-31"]),
            "notice_date": pd.to_datetime(["2024-03-10"]),
            "net_profit": [10.0],
        })
        balance = pd.DataFrame({
            "code": ["000001"],
            "name": ["样本"],
            "report_date": pd.to_datetime(["2023-12-31"]),
            "balance_notice_date": pd.to_datetime(["2024-03-15"]),
            "total_assets": [100.0],
            "total_liabilities": [40.0],
            "cash": [10.0],
            "accounts_receivable": [12.0],
            "inventory": [8.0],
            "accounts_payable": [7.0],
            "advance_receipts": [3.0],
        })
        cashflow = pd.DataFrame({
            "code": ["000001"],
            "name": ["样本"],
            "report_date": pd.to_datetime(["2023-12-31"]),
            "cashflow_notice_date": pd.to_datetime(["2024-03-20"]),
            "operating_cashflow": [15.0],
        })
        joined, audit = join_statement_sources(
            income, balance, cashflow
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(
            joined.iloc[0]["source_availability_date"],
            pd.Timestamp("2024-03-20"),
        )
        self.assertEqual(
            joined.iloc[0]["non_cash_working_capital"], 10
        )

    def test_builder_excludes_financials_and_future_reports(self):
        codes = {f"{number:06d}" for number in range(1, 62)}
        rows = []
        for number, code in enumerate(sorted(codes), start=1):
            for year in [2021, 2022]:
                rows.append({
                    "code": code,
                    "report_date": pd.Timestamp(f"{year}-12-31"),
                    "income_notice_date": pd.Timestamp(
                        f"{year + 1}-03-10"
                    ),
                    "balance_notice_date": pd.Timestamp(
                        f"{year + 1}-03-15"
                    ),
                    "cashflow_notice_date": pd.Timestamp(
                        f"{year + 1}-03-20"
                    ),
                    "source_availability_date": pd.Timestamp(
                        f"{year + 1}-03-20"
                    ),
                    "net_profit": 10 + number,
                    "total_assets": 100 + number,
                    "total_liabilities": 40 + number / 2,
                    "operating_cashflow": 12 + number,
                    "non_cash_working_capital": 10 + number + year,
                })
        rows.append({
            **rows[1],
            "code": "000001",
            "report_date": pd.Timestamp("2023-12-31"),
            "source_availability_date": pd.Timestamp("2024-03-20"),
            "income_notice_date": pd.Timestamp("2024-03-10"),
            "balance_notice_date": pd.Timestamp("2024-03-15"),
            "cashflow_notice_date": pd.Timestamp("2024-03-20"),
            "total_assets": 999999,
        })
        statements = pd.DataFrame(rows)

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(
                    codes=codes,
                    code_to_sector={
                        code: (
                            "J66_financial"
                            if code == "000061"
                            else (
                                "sector_a"
                                if int(code) <= 30 else "sector_b"
                            )
                        )
                        for code in codes
                    },
                )

        signal = pd.Timestamp("2023-06-30")
        factors = build_balance_quality_factors(
            statements=statements,
            rebalance_dates=pd.DatetimeIndex([signal]),
            universe_store=FakeUniverse(),
        )
        self.assertEqual(factors["code"].nunique(), 60)
        self.assertNotIn("000061", set(factors["code"]))
        self.assertTrue(set(FACTOR_NAMES).issubset(factors.columns))
        self.assertTrue(
            (factors["source_availability_date"] <= signal).all()
        )
        self.assertLess(
            factors.loc[
                factors["code"] == "000001", "source_report_date"
            ].iloc[0],
            pd.Timestamp("2023-12-31"),
        )
        medians = factors.groupby("sector")[FACTOR_NAMES].median()
        np.testing.assert_allclose(
            medians.to_numpy(), 0, atol=1e-12, equal_nan=True
        )

    def test_quality_denominator_excludes_financials(self):
        factors = pd.DataFrame({
            "date": [pd.Timestamp("2023-06-30")] * 60,
            "code": [f"{number:06d}" for number in range(60)],
            "source_availability_date": [
                pd.Timestamp("2023-03-20")
            ] * 60,
            **{factor: [1.0] * 60 for factor in FACTOR_NAMES},
        })

        class FakeUniverse:
            @staticmethod
            def snapshot_for(_date):
                return SimpleNamespace(
                    codes={f"{number:06d}" for number in range(100)},
                    code_to_sector={
                        f"{number:06d}": (
                            "J66_financial" if number >= 90 else "industrial"
                        )
                        for number in range(100)
                    },
                )

        by_date, summary = profile_factor_quality(
            factors, FakeUniverse()
        )
        self.assertTrue(summary["passed"])
        self.assertTrue(
            (by_date["non_financial_universe_count"] == 90).all()
        )
        self.assertTrue(
            np.allclose(by_date["coverage_ratio"], 2 / 3)
        )


if __name__ == "__main__":
    unittest.main()
