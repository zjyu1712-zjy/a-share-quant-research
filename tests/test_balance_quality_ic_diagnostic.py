import unittest

import numpy as np
import pandas as pd

from src.balance_quality_ic_diagnostic import (
    FILTER_NAME,
    assign_quintiles,
    estimate_market_cap,
    evaluate_filter_gate,
    summarize_quintile_shape,
)
from src.balance_quality_portfolio_validation import MATCHED_REFERENCE


class BalanceQualityICDiagnosticTests(unittest.TestCase):
    def test_quintiles_preserve_high_score_direction(self):
        values = pd.Series(np.arange(100), dtype=float)
        quintiles = assign_quintiles(values)
        self.assertEqual(int(quintiles.iloc[0]), 1)
        self.assertEqual(int(quintiles.iloc[-1]), 5)
        self.assertEqual(
            quintiles.value_counts().sort_index().tolist(),
            [20, 20, 20, 20, 20],
        )

    def test_market_cap_proxy_uses_matching_report(self):
        frame = pd.DataFrame({
            "date": [pd.Timestamp("2024-05-01")],
            "code": ["000001"],
            "source_report_date": [pd.Timestamp("2023-12-31")],
            "close": [10.0],
        })
        income = pd.DataFrame({
            "code": ["000001"],
            "report_date": [pd.Timestamp("2023-12-31")],
            "eps": [2.0],
            "net_profit": [2_000_000_000.0],
        })
        result = estimate_market_cap(frame, income)
        self.assertEqual(result.loc[0, "estimated_shares"], 1_000_000_000)
        self.assertEqual(
            result.loc[0, "estimated_market_cap"], 10_000_000_000
        )

    def test_shape_requires_bottom_worst_and_five_folds(self):
        summary_rows = []
        date_rows = []
        for quintile, value in enumerate([0, 1, 2, 3, 4], start=1):
            summary_rows.append({
                "candidate": "low_accruals",
                "horizon_weeks": 4,
                "quintile": quintile,
                "mean_return_pct": value,
            })
        for fold in range(1, 8):
            for quintile, value in enumerate(
                [0, 1, 2, 3, 4], start=1
            ):
                date_rows.append({
                    "candidate": "low_accruals",
                    "horizon_weeks": 4,
                    "fold": fold,
                    "date": f"2024-01-{fold:02d}",
                    "quintile": quintile,
                    "equal_weight_return_pct": value,
                })
        # 补齐函数所需的另外两个候选和8周口径，断言只看目标行。
        for candidate in [
            "working_capital_discipline",
            "accruals50_working_capital50",
        ]:
            for horizon in [4, 8]:
                for quintile, value in enumerate(
                    [0, 1, 2, 3, 4], start=1
                ):
                    summary_rows.append({
                        "candidate": candidate,
                        "horizon_weeks": horizon,
                        "quintile": quintile,
                        "mean_return_pct": value,
                    })
        for quintile, value in enumerate([0, 1, 2, 3, 4], start=1):
            summary_rows.append({
                "candidate": "low_accruals",
                "horizon_weeks": 8,
                "quintile": quintile,
                "mean_return_pct": value,
            })
        result = summarize_quintile_shape(
            pd.DataFrame(date_rows),
            pd.DataFrame(summary_rows),
        )
        row = result[
            (result["candidate"] == "low_accruals")
            & (result["horizon_weeks"] == 4)
        ].iloc[0]
        self.assertTrue(row["stable_monotonic_signal"])
        self.assertEqual(row["positive_spread_folds"], 7)

    def test_filter_gate_is_strict(self):
        folds = []
        for fold in range(1, 8):
            folds.extend([
                {
                    "fold": fold,
                    "strategy": MATCHED_REFERENCE,
                    "test_total_return": 1.0,
                },
                {
                    "fold": fold,
                    "strategy": FILTER_NAME,
                    "test_total_return": 2.0 if fold <= 5 else 0.0,
                },
            ])
        summary = pd.DataFrame([
            {
                "strategy": MATCHED_REFERENCE,
                "compounded_oos_return": 7.0,
                "mean_annualized_selection_turnover_pct": 100.0,
                "worst_fold_drawdown": -10.0,
            },
            {
                "strategy": FILTER_NAME,
                "compounded_oos_return": 8.0,
                "mean_annualized_selection_turnover_pct": 110.0,
                "worst_fold_drawdown": -13.0,
            },
        ])
        gate = evaluate_filter_gate(pd.DataFrame(folds), summary)
        self.assertTrue(gate["passed"])
        summary.loc[
            summary["strategy"] == FILTER_NAME,
            "mean_annualized_selection_turnover_pct",
        ] = 110.01
        self.assertFalse(
            evaluate_filter_gate(pd.DataFrame(folds), summary)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
