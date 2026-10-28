import subprocess
import unittest
from pathlib import Path

import pandas as pd


class V2LowVolFailureAttributionTests(unittest.TestCase):
    def test_ablation_metadata_is_one_fixed_additive_chain(self):
        from src.v2_lowvol_failure_attribution import ablation_metadata

        frame = ablation_metadata(100_000).set_index("variant")
        self.assertEqual(
            frame.index.tolist(),
            [
                "raw_lowvol",
                "add_sector_constraint",
                "add_rank_buffer",
                "add_drawdown_overlay",
                "full_framework",
            ],
        )
        self.assertFalse(bool(frame.loc["raw_lowvol", "sector_enabled"]))
        self.assertEqual(int(frame.loc["raw_lowvol", "rank_buffer"]), 0)
        self.assertFalse(bool(frame.loc["raw_lowvol", "drawdown_overlay"]))
        self.assertTrue(bool(frame.loc["raw_lowvol", "frictionless"]))
        self.assertTrue(bool(frame.loc["full_framework", "sector_enabled"]))
        self.assertEqual(int(frame.loc["full_framework", "rank_buffer"]), 30)
        self.assertTrue(bool(frame.loc["full_framework", "drawdown_overlay"]))
        self.assertFalse(bool(frame.loc["full_framework", "frictionless"]))

    def test_waterfall_deltas_reconcile_raw_to_full(self):
        from src.v2_lowvol_failure_attribution import build_waterfall

        metrics = pd.DataFrame(
            {
                "variant": [
                    "raw_lowvol",
                    "add_sector_constraint",
                    "add_rank_buffer",
                    "add_drawdown_overlay",
                    "full_framework",
                ],
                "total_return": [12.0, 10.0, 11.0, 4.0, 3.0],
                "cagr": [4.0, 3.0, 3.5, 1.0, 0.5],
                "max_drawdown": [-20.0, -18.0, -18.5, -9.0, -9.2],
            }
        )
        result = build_waterfall(metrics)
        self.assertAlmostEqual(result["delta_total_return"].sum(), -9.0)
        self.assertAlmostEqual(result["delta_cagr"].sum(), -3.5)
        self.assertEqual(result.iloc[0]["component"], "raw_factor_portfolio")
        self.assertEqual(result.iloc[-1]["component"], "trading_frictions")

    def test_failure_classifier_distinguishes_factor_from_execution(self):
        from src.v2_lowvol_failure_attribution import classify_failure

        raw_bad = pd.DataFrame(
            {
                "variant": ["raw_lowvol", "full_framework"],
                "cagr": [-1.0, -2.0],
                "excess_return": [-2.0, -3.0],
            }
        )
        raw_result = classify_failure(raw_bad)
        self.assertEqual(raw_result["primary_cause"], "factor_translation")
        self.assertEqual(raw_result["primary_cause_label"], "因子到集中组合的转化")

        overlay_bad = pd.DataFrame(
            {
                "variant": [
                    "raw_lowvol",
                    "add_sector_constraint",
                    "add_rank_buffer",
                    "add_drawdown_overlay",
                    "full_framework",
                ],
                "cagr": [12.0, 12.0, 11.0, 2.0, 1.5],
                "excess_return": [8.0, 8.0, 7.0, -2.0, -2.5],
            }
        )
        overlay_result = classify_failure(overlay_bad)
        self.assertEqual(overlay_result["primary_cause"], "drawdown_overlay")
        self.assertEqual(overlay_result["primary_cause_label"], "回撤仓位控制")

    def test_run_mac_lists_failure_attribution_entry(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [str(root / "run_mac.sh"), "help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("v2attribution", result.stdout)

    def test_report_surfaces_concentration_and_path_dependence(self):
        from src.v2_lowvol_failure_attribution import _render_report

        summary = {
            "failure_classification": {
                "primary_cause": "drawdown_overlay",
                "explanation": "test",
            },
            "selection_summary": {
                "average_raw_unique_sectors": 4.1,
                "average_constrained_unique_sectors": 7.7,
                "average_raw_max_sector_share": 0.634,
                "average_constrained_max_sector_share": 0.2,
                "average_buffer_retained_outside_top10": 5.4,
                "average_holding_overlap_ratio": 0.789,
            },
        }
        metrics = pd.DataFrame(
            {"variant": ["full_framework"], "total_return": [-3.59]}
        )
        waterfall = pd.DataFrame(
            {"component": ["trading_frictions"], "delta_cagr": [0.11]}
        )
        report = _render_report(summary, metrics, waterfall)
        self.assertIn("63.4%", report)
        self.assertIn("路径依赖", report)

    def test_report_uses_compact_reader_table_not_full_audit_schema(self):
        from src.v2_lowvol_failure_attribution import _render_report

        summary = {
            "failure_classification": {
                "primary_cause": "drawdown_overlay",
                "explanation": "test",
            },
            "selection_summary": {},
        }
        metrics = pd.DataFrame({
            "variant": ["raw_lowvol"], "label": ["原始组合"],
            "total_return": [39.6], "cagr": [10.69], "sharpe": [0.724],
            "max_drawdown": [-11.84], "benchmark_return": [-1.13],
            "excess_return": [40.73], "avg_turnover_pct": [69.0],
            "avg_deployment_pct": [97.5], "total_fees": [0.0],
            "slippage_cost": [0.0], "friction_drag_pct_of_initial_cash": [0.0],
            "risk_event_count": [0], "warning_or_defensive_events": [0],
        })
        waterfall = pd.DataFrame({
            "component": ["raw_factor_portfolio"], "delta_cagr": [0.0]
        })
        report = _render_report(summary, metrics, waterfall)
        self.assertIn("年化收益", report)
        self.assertNotIn("warning_or_defensive_events", report)
        self.assertNotIn("friction_drag_pct_of_initial_cash", report)


if __name__ == "__main__":
    unittest.main()
