import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.v2_monthly_strategy import v2_research_protocol
from src.v2_monthly_validation import (
    evaluate_v2_gate,
    run_infrastructure_audit,
)


class V2MonthlyValidationTests(unittest.TestCase):
    def _passing_folds(self):
        return pd.DataFrame({
            "test_total_return": [5, 4, 3, 2, 1, 2, -1],
            "test_excess_return": [2, 2, 1, 1, 1, 1, -1],
            "test_max_drawdown": [-5, -6, -7, -8, -9, -9.5, -9.9],
        })

    def test_gate_requires_return_excess_and_every_window_dd10(self):
        folds = self._passing_folds()
        continuous = {
            "cagr": 10.1,
            "excess_return": 1.0,
            "max_drawdown": -9.9,
        }
        self.assertTrue(evaluate_v2_gate(folds, continuous)["passed"])
        continuous["max_drawdown"] = -10.01
        self.assertFalse(evaluate_v2_gate(folds, continuous)["passed"])
        continuous = {
            "cagr": 9.99,
            "excess_return": 1.0,
            "max_drawdown": -9.0,
        }
        self.assertFalse(evaluate_v2_gate(folds, continuous)["passed"])

    def test_exactly_ten_percent_drawdown_is_allowed_by_research_gate(self):
        folds = self._passing_folds()
        folds.loc[0, "test_max_drawdown"] = -10.0
        result = evaluate_v2_gate(
            folds,
            {"cagr": 10.0, "excess_return": 0.01, "max_drawdown": -10.0},
        )
        self.assertTrue(result["passed"])

    def test_protocol_uses_sse_benchmark_but_cannot_deploy(self):
        protocol = v2_research_protocol()
        self.assertEqual(
            protocol["primary_benchmark"], "SSE Composite / 000001"
        )
        self.assertEqual(protocol["research_status"], "infrastructure_only")
        self.assertFalse(protocol["deployment_authorized"])

    def test_infrastructure_audit_writes_fixed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            output = root / "output"
            snapshots.mkdir()
            members = pd.DataFrame({
                "code": [f"600{i:03d}" for i in range(120)],
                "name": ["股票"] * 120,
                "sector": ["行业"] * 120,
            })
            members.to_csv(snapshots / "2025-01-31.csv", index=False)
            (snapshots / "2025-01-31.audit.json").write_text(
                json.dumps({
                    "effective_date": "2025-01-31",
                    "raw_securities": 180,
                    "final_members": 120,
                    "excluded_star": 40,
                }),
                encoding="utf-8",
            )
            summary = run_infrastructure_audit(
                start="2025-01-01",
                end="2025-01-31",
                output_dir=output,
                snapshot_dir=snapshots,
            )
            self.assertEqual(summary["snapshot_count"], 1)
            self.assertEqual(summary["research_status"], "infrastructure_only")
            self.assertFalse(summary["deployment_authorized"])
            for name in (
                "research_protocol.json",
                "universe_audit_by_date.csv",
                "universe_exclusions.csv",
                "validation_checks.json",
                "summary.json",
                "report.html",
            ):
                self.assertTrue((output / name).exists(), name)

    def test_infrastructure_audit_rejects_missing_requested_months(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            output = root / "output"
            snapshots.mkdir()
            members = pd.DataFrame({
                "code": [f"600{i:03d}" for i in range(120)],
                "name": ["股票"] * 120,
                "sector": ["行业"] * 120,
            })
            members.to_csv(snapshots / "2025-01-31.csv", index=False)
            (snapshots / "2025-01-31.audit.json").write_text(
                json.dumps({
                    "effective_date": "2025-01-31",
                    "final_members": 120,
                }),
                encoding="utf-8",
            )

            summary = run_infrastructure_audit(
                start="2025-01-01",
                end="2025-03-31",
                output_dir=output,
                snapshot_dir=snapshots,
            )

            self.assertFalse(summary["infrastructure_checks_passed"])
            self.assertEqual(summary["expected_snapshot_months"], 3)
            self.assertEqual(
                summary["missing_snapshot_months"], ["2025-02", "2025-03"]
            )


if __name__ == "__main__":
    unittest.main()
