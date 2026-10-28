import unittest

import numpy as np
import pandas as pd

from src.quality_value_residual_screening import (
    FACTOR_NAMES,
    build_robust_decision,
    is_financial_sector,
    residualize_cross_section,
)


class QualityValueResidualScreeningTests(unittest.TestCase):
    def test_financial_sector_detection_is_explicit(self):
        for sector in [
            "J66货币金融服务",
            "J67资本市场服务",
            "J68保险业",
            "J69其他金融业",
        ]:
            self.assertTrue(is_financial_sector(sector))
        self.assertFalse(is_financial_sector("K70房地产业"))
        self.assertFalse(is_financial_sector("C39电子设备制造业"))

    def test_residual_removes_value_relation_and_sector_medians(self):
        count = 120
        value = np.linspace(-2, 2, count)
        frame = pd.DataFrame({
            "quality": 0.8 * value + np.sin(np.arange(count)) * 0.05,
            "value_book_to_price_mrq": value,
            "sector": ["sector_a"] * 60 + ["sector_b"] * 60,
        })
        residual, diagnostics = residualize_cross_section(
            frame,
            quality_column="quality",
        )
        self.assertEqual(residual.notna().sum(), count)
        self.assertGreater(diagnostics["pre_rank_correlation"], 0.95)
        self.assertLess(
            abs(diagnostics["post_rank_correlation"]),
            0.10,
        )
        np.testing.assert_allclose(
            residual.groupby(frame["sector"]).median(),
            0,
            atol=1e-12,
        )

    def test_robust_decision_requires_both_universes(self):
        rows = []
        for factor in FACTOR_NAMES:
            rows.extend([
                {
                    "universe_variant": "all_hs300",
                    "factor": factor,
                    "status": "promising",
                },
                {
                    "universe_variant": "ex_financials",
                    "factor": factor,
                    "status": (
                        "rejected"
                        if factor == FACTOR_NAMES[0]
                        else "promising"
                    ),
                },
            ])
        result = build_robust_decision(pd.DataFrame(rows))
        self.assertEqual(result.iloc[0]["status"], "rejected")
        self.assertTrue(
            (result.iloc[1:]["status"] == "promising").all()
        )


if __name__ == "__main__":
    unittest.main()
