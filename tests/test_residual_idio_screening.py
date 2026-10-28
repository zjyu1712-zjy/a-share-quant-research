import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.residual_idio_screening import (
    FACTOR,
    build_protocol,
    build_report,
    build_residual_factor,
)


class ResidualIdioScreeningTests(unittest.TestCase):
    def test_residual_is_orthogonal_to_lowvol_rank_and_unique(self):
        rows = []
        for date in pd.to_datetime(["2023-01-31", "2023-02-28"]):
            for index in range(120):
                lowvol = index / 120
                rows.append({
                    "date": date,
                    "code": f"{index:06d}",
                    "sector": f"S{index % 6}",
                    "low_volatility_60_reference": lowvol,
                    "low_idiosyncratic_volatility_120": (
                        0.8 * lowvol
                        + (index % 7) / 20
                        + (index % 6) / 50
                    ),
                })
        factors, audit = build_residual_factor(pd.DataFrame(rows))
        self.assertFalse(factors.duplicated(["date", "code"]).any())
        self.assertTrue(factors[FACTOR].notna().all())
        self.assertLess(
            audit["pearson_corr_with_lowvol_rank"].abs().max(), 1e-10
        )

    def test_protocol_freezes_one_direction_and_future_boundary(self):
        protocol = build_protocol()
        self.assertTrue(protocol["no_parameter_search"])
        self.assertEqual(protocol["direction"], "higher_is_better")
        self.assertEqual(protocol["future_holdout_start"], "2026-08-03")
        self.assertEqual(len(protocol["protocol_hash"]), 64)

    def test_report_headline_follows_decision(self):
        result = {
            "status": "advance",
            "failed_gates": [],
            "four_week_mean_rank_ic": 0.03,
            "four_week_hac_t_stat": 3.0,
            "four_week_one_sided_p_value": 0.001,
            "annualized_gross_spread_pct": 5.0,
            "aggregate_quintile_monotonic_corr": 0.9,
            "mean_abs_rank_correlation_to_lowvol": 0.1,
            "mean_monthly_top_quintile_turnover": 0.4,
            "positive_full_year_ratio": 1.0,
            "mean_residualization_r_squared": 0.7,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.html"
            build_report(build_protocol(), result, report)
            html = report.read_text(encoding="utf-8")
        self.assertIn("通过历史准入门槛</h1>", html)
        self.assertIn("未通过：无", html)


if __name__ == "__main__":
    unittest.main()
