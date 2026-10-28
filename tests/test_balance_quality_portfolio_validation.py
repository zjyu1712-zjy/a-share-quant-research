import unittest

import pandas as pd

from src.balance_quality_portfolio_validation import (
    CANDIDATES,
    MATCHED_REFERENCE,
    calculate_incremental_gates,
    candidate_metadata,
    validate_result_tables,
)


def _summary_row(
    strategy,
    *,
    compounded=10.0,
    turnover=100.0,
    drawdown=-10.0,
    passed=True,
):
    return {
        "strategy": strategy,
        "compounded_oos_return": compounded,
        "compounded_benchmark_return": -5.0,
        "compounded_excess_return_pp": compounded + 5.0,
        "median_fold_return": 1.0,
        "median_fold_excess_return": 1.0,
        "positive_return_fold_ratio": 6 / 7,
        "positive_excess_fold_ratio": 6 / 7,
        "mean_fold_sharpe": 0.5,
        "worst_fold_drawdown": drawdown,
        "mean_annualized_selection_turnover_pct": turnover,
        "passed": passed,
    }


class BalanceQualityPortfolioValidationTests(unittest.TestCase):
    def test_metadata_freezes_three_candidates_and_execution(self):
        metadata = candidate_metadata(100_000)
        eligible = metadata[metadata["eligible_for_advancement"]]
        self.assertEqual(len(eligible), 3)
        self.assertTrue((metadata["top_n"] == 10).all())
        self.assertTrue((metadata["rank_buffer"] == 30).all())
        self.assertTrue((metadata["max_per_sector"] == 2).all())
        composite = metadata[
            metadata["candidate"]
            == "accruals50_working_capital50"
        ].iloc[0]
        self.assertEqual(composite["factor_one_weight"], 0.5)
        self.assertEqual(composite["factor_two_weight"], 0.5)

    def test_incremental_gate_requires_five_of_seven_wins(self):
        candidates = list(CANDIDATES)
        summaries = []
        folds = []
        for candidate in candidates:
            summaries.append(_summary_row(
                candidate,
                compounded=10.0 if candidate == MATCHED_REFERENCE else 12.0,
            ))
            returns = (
                [1.0] * 7
                if candidate == MATCHED_REFERENCE
                else [2.0] * 5 + [0.0] * 2
            )
            for fold, value in enumerate(returns, start=1):
                folds.append({
                    "fold": fold,
                    "candidate": candidate,
                    "test_total_return": value,
                })
        gates = calculate_incremental_gates(
            pd.DataFrame(folds), pd.DataFrame(summaries)
        )
        self.assertEqual(len(gates), 3)
        self.assertTrue(gates["passed"].all())
        self.assertTrue(
            (gates["positive_incremental_folds"] == 5).all()
        )

    def test_incremental_gate_rejects_turnover_or_drawdown_damage(self):
        summaries = []
        folds = []
        for candidate in CANDIDATES:
            turnover = 130.0 if candidate == "low_accruals" else 100.0
            drawdown = (
                -14.0
                if candidate == "working_capital_discipline" else -10.0
            )
            summaries.append(_summary_row(
                candidate,
                compounded=10.0 if candidate == MATCHED_REFERENCE else 12.0,
                turnover=turnover,
                drawdown=drawdown,
            ))
            returns = (
                [1.0] * 7
                if candidate == MATCHED_REFERENCE else [2.0] * 7
            )
            for fold, value in enumerate(returns, start=1):
                folds.append({
                    "fold": fold,
                    "candidate": candidate,
                    "test_total_return": value,
                })
        gates = calculate_incremental_gates(
            pd.DataFrame(folds), pd.DataFrame(summaries)
        ).set_index("candidate")
        self.assertFalse(gates.loc["low_accruals", "passed"])
        self.assertFalse(
            gates.loc["working_capital_discipline", "passed"]
        )
        self.assertTrue(
            gates.loc["accruals50_working_capital50", "passed"]
        )

    def test_validation_recomputes_headline_metrics(self):
        rows = []
        summaries = []
        for candidate in CANDIDATES:
            candidate_rows = []
            for fold in range(1, 8):
                row = {
                    "fold": fold,
                    "candidate": candidate,
                    "test_total_return": 1.0,
                    "test_benchmark_return": 0.0,
                    "test_excess_return": 1.0,
                    "test_sharpe": 0.5,
                    "test_max_drawdown": -5.0,
                    "test_annualized_selection_turnover_pct": 100.0,
                    "test_total_fees": 1.0,
                    "test_slippage_cost": 1.0,
                    "test_friction_drag_pct_of_initial_cash": 0.01,
                    "test_avg_deployment_pct": 95.0,
                }
                rows.append(row)
                candidate_rows.append(row)
            summary = {
                "strategy": candidate,
                **__import__(
                    "src.low_volatility_portfolio_validation",
                    fromlist=["summarize_rows"],
                ).summarize_rows(pd.DataFrame(candidate_rows)),
            }
            summaries.append(summary)
        folds = pd.DataFrame(rows)
        summary_table = pd.DataFrame(summaries)
        incremental = calculate_incremental_gates(
            folds, summary_table
        )
        result = validate_result_tables(
            folds_frame=folds,
            summary_table=summary_table,
            incremental=incremental,
            factor_quality={
                "passed": True,
                "duplicate_date_code_rows": 0,
                "future_fundamental_matches": 0,
                "median_coverage_ratio": 0.99,
            },
            expected_folds=7,
        )
        self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
