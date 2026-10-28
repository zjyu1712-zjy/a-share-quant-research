import json
import tempfile
import unittest
from pathlib import Path

from src.v2_monthly_strategy import (
    build_v2_config,
    ensure_v2_protocol,
    v2_research_protocol,
)


class V2MonthlyStrategyTests(unittest.TestCase):
    def test_v2_config_is_monthly_top10_dd10_and_sse_only(self):
        config = build_v2_config()
        self.assertEqual(config.portfolio.rebalance_frequency, "monthly")
        self.assertEqual(config.portfolio.top_n, 10)
        self.assertEqual(config.portfolio.rank_buffer, 30)
        self.assertEqual(config.sector_neutral.max_per_sector, 2)
        self.assertEqual(config.universe.source, "sse_composite_a_pit")
        self.assertEqual(config.universe.snapshot_dir, "data/universe/sse_composite_a")
        self.assertEqual(config.risk_control.max_drawdown_hard_limit, 0.10)
        self.assertEqual(config.risk_control.warning_dd_threshold, 0.07)
        self.assertEqual(config.risk_control.circuit_breaker_dd_threshold, 0.09)
        self.assertEqual(config.risk_control.recovery_dd_threshold, 0.05)
        self.assertFalse(config.market_timing.enabled)
        self.assertEqual(config.scoring.sentiment_weight, 0.0)

    def test_v2_protocol_has_separate_output_and_no_deployment_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = ensure_v2_protocol(Path(tmp))
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["strategy_id"], "v2_monthly_top10_dd10")
        self.assertEqual(payload["research_status"], "infrastructure_only")
        self.assertFalse(payload["deployment_authorized"])
        self.assertNotIn("data/paper_trading", str(path))
        self.assertEqual(len(digest), 64)

    def test_existing_protocol_cannot_be_silently_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            path = output / "research_protocol.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "研究协议"):
                ensure_v2_protocol(output)

    def test_protocol_identifies_primary_benchmark(self):
        protocol = v2_research_protocol()
        self.assertEqual(protocol["primary_benchmark"], "SSE Composite / 000001")


if __name__ == "__main__":
    unittest.main()
