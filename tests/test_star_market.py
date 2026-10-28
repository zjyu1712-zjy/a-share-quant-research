import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.star_market import (
    DEFAULT_CONFIG_PATH,
    HistoricalStarUniverse,
    PROJECT_ROOT,
    IsolationError,
    StarMarketConfig,
    StarUniverseSnapshot,
    UniverseEvidenceError,
    assert_isolated_output_path,
    enforce_star_market_limits,
    load_star_market_config,
    main,
    screen_star_candidates,
    update_star_market_config,
    write_star_market_report,
)


class StarMarketConfigTests(unittest.TestCase):
    def test_default_config_keeps_entry_closed(self):
        config = load_star_market_config(DEFAULT_CONFIG_PATH)

        self.assertFalse(config.enabled)
        self.assertFalse(config.investor_permission_confirmed)
        self.assertFalse(config.entry_open)
        self.assertEqual(config.max_holdings, 2)
        self.assertEqual(config.max_portfolio_weight, 0.20)
        self.assertTrue(config.exclude_unprofitable_u)

    def test_enabled_without_permission_stays_closed(self):
        config = StarMarketConfig.from_mapping({
            "enabled": True,
            "investor_permission_confirmed": False,
            "max_portfolio_weight": 0.20,
            "max_holdings": 2,
            "exclude_unprofitable_u": True,
        })

        self.assertFalse(config.entry_open)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            StarMarketConfig.from_mapping({
                "enabled": False,
                "investor_permission_confirmed": False,
                "max_portfolio_weight": 1.20,
                "max_holdings": 2,
                "exclude_unprofitable_u": True,
            })

        with self.assertRaises(ValueError):
            StarMarketConfig(max_holdings=2.5)

    def test_enabling_requires_manual_permission_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "star_market.json"
            path.write_text(json.dumps({
                "enabled": False,
                "investor_permission_confirmed": False,
                "max_portfolio_weight": 0.20,
                "max_holdings": 2,
                "exclude_unprofitable_u": True,
            }), encoding="utf-8")

            with self.assertRaises(PermissionError):
                update_star_market_config(path, enabled=True)

            unchanged = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(unchanged["enabled"])
            self.assertFalse(unchanged["investor_permission_confirmed"])

    def test_v1_output_path_and_descendants_are_rejected(self):
        v1_path = PROJECT_ROOT / "data" / "paper_trading"

        with self.assertRaises(IsolationError):
            assert_isolated_output_path(v1_path)
        with self.assertRaises(IsolationError):
            assert_isolated_output_path(v1_path / "star_market")

    def test_v2_output_path_is_accepted(self):
        path = PROJECT_ROOT / "data" / "v2_monthly_top10_dd10" / "star_market"

        self.assertEqual(assert_isolated_output_path(path), path.resolve())


class HistoricalStarUniverseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshot_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_snapshot(
        self,
        effective_date: str,
        *,
        known_at: str | None = None,
        evidence_type: str = "official_historical_snapshot",
        row_effective_date: str | None = None,
    ) -> Path:
        frame = pd.DataFrame([
            {
                "code": "688001",
                "name": "华兴源创",
                "sector": "C39",
                "effective_date": row_effective_date or effective_date,
                "known_at": known_at or effective_date,
                "source": "上交所历史公告 SSE-TEST-001",
                "evidence_type": evidence_type,
                "inclusion_reason": "公告确认在样本中",
                "exclusion_reason": "",
                "is_member": True,
                "is_unprofitable_u": False,
            },
            {
                "code": "688002",
                "name": "示例-U",
                "sector": "C27",
                "effective_date": row_effective_date or effective_date,
                "known_at": known_at or effective_date,
                "source": "上交所历史公告 SSE-TEST-001",
                "evidence_type": evidence_type,
                "inclusion_reason": "",
                "exclusion_reason": "不在当期上证综指样本",
                "is_member": False,
                "is_unprofitable_u": True,
            },
        ])
        path = self.snapshot_dir / f"{effective_date}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        return path

    def test_missing_historical_snapshot_stops_research(self):
        store = HistoricalStarUniverse(self.snapshot_dir)

        with self.assertRaises(UniverseEvidenceError):
            store.snapshot_for("2020-01-31")

    def test_snapshot_never_uses_a_future_file(self):
        self.write_snapshot("2021-01-01", known_at="2020-12-31")

        with self.assertRaises(UniverseEvidenceError):
            HistoricalStarUniverse(self.snapshot_dir).snapshot_for("2020-12-31")

    def test_future_known_at_is_rejected(self):
        self.write_snapshot("2021-01-01", known_at="2021-01-02")

        with self.assertRaises(UniverseEvidenceError):
            HistoricalStarUniverse(self.snapshot_dir).snapshot_for("2021-01-01")

    def test_effective_date_must_match_filename(self):
        self.write_snapshot(
            "2021-01-01",
            row_effective_date="2020-12-31",
        )

        with self.assertRaises(UniverseEvidenceError):
            HistoricalStarUniverse(self.snapshot_dir).snapshot_for("2021-01-01")

    def test_current_constituent_backfill_evidence_is_rejected(self):
        self.write_snapshot(
            "2021-01-01",
            evidence_type="current_constituents_backfill",
        )

        with self.assertRaises(UniverseEvidenceError):
            HistoricalStarUniverse(self.snapshot_dir).snapshot_for("2021-01-01")

    def test_audit_records_source_hash_counts_and_reasons(self):
        path = self.write_snapshot("2021-01-01", known_at="2020-12-31")
        store = HistoricalStarUniverse(self.snapshot_dir)

        snapshot = store.snapshot_for("2021-01-31")
        audit = store.audit("2021-01-31")

        self.assertEqual(snapshot.codes, {"688001"})
        self.assertEqual(audit["requested_date"], "2021-01-31")
        self.assertEqual(audit["effective_date"], "2021-01-01")
        self.assertEqual(audit["known_at_max"], "2020-12-31")
        self.assertEqual(audit["raw_row_count"], 2)
        self.assertEqual(audit["included_count"], 1)
        self.assertEqual(audit["excluded_count"], 1)
        self.assertEqual(
            audit["exclusion_reason_counts"],
            {"不在当期上证综指样本": 1},
        )
        self.assertEqual(audit["unprofitable_u_count"], 1)
        self.assertTrue(audit["historical_evidence_complete"])
        self.assertEqual(len(audit["sha256"]), 64)
        self.assertEqual(snapshot.path, path)


class StarMarketSelectionTests(unittest.TestCase):
    def setUp(self):
        records = pd.DataFrame([
            {
                "code": "688001",
                "name": "华兴源创",
                "is_member": True,
                "is_unprofitable_u": False,
            },
            {
                "code": "688002",
                "name": "睿创微纳-U",
                "is_member": True,
                "is_unprofitable_u": True,
            },
            {
                "code": "688003",
                "name": "天准科技",
                "is_member": True,
                "is_unprofitable_u": False,
            },
            {
                "code": "688004",
                "name": "博汇科技",
                "is_member": False,
                "is_unprofitable_u": False,
            },
        ])
        self.snapshot = StarUniverseSnapshot(
            pd.Timestamp("2026-07-31"),
            records,
            Path("2026-07-31.csv"),
        )
        self.closed = StarMarketConfig()
        self.open = StarMarketConfig(
            enabled=True,
            investor_permission_confirmed=True,
        )

    @staticmethod
    def candidates(*codes: str) -> pd.DataFrame:
        names = {
            "688001": "华兴源创",
            "688002": "睿创微纳-U",
            "688003": "天准科技",
            "688004": "博汇科技",
        }
        return pd.DataFrame([
            {
                "code": code,
                "name": names[code],
                "rank": rank,
                "target_weight": 0.15,
                "selection_reason": f"综合排名第{rank}",
            }
            for rank, code in enumerate(codes, start=1)
        ])

    def test_rank_one_cannot_enter_until_both_confirmations_are_true(self):
        cases = [
            StarMarketConfig(enabled=False, investor_permission_confirmed=True),
            StarMarketConfig(enabled=True, investor_permission_confirmed=False),
        ]

        for config in cases:
            with self.subTest(config=config):
                eligible, audit = screen_star_candidates(
                    self.candidates("688001"),
                    self.snapshot,
                    config,
                )
                self.assertTrue(eligible.empty)
                self.assertEqual(audit.iloc[0]["reason"], "star_market_entry_closed")

    def test_unprofitable_u_is_excluded_by_default(self):
        eligible, audit = screen_star_candidates(
            self.candidates("688002"),
            self.snapshot,
            self.open,
        )

        self.assertTrue(eligible.empty)
        self.assertEqual(audit.iloc[0]["reason"], "unprofitable_u_excluded")

    def test_nonmember_is_excluded_with_point_in_time_reason(self):
        eligible, audit = screen_star_candidates(
            self.candidates("688004"),
            self.snapshot,
            self.open,
        )

        self.assertTrue(eligible.empty)
        self.assertEqual(audit.iloc[0]["reason"], "not_in_historical_snapshot")
        self.assertEqual(audit.iloc[0]["universe_effective_date"], "2026-07-31")

    def test_limits_keep_at_most_two_and_cap_total_weight(self):
        targets = self.candidates("688001", "688003", "688004")

        constrained, audit = enforce_star_market_limits(targets, self.open)

        self.assertEqual(list(constrained["code"]), ["688001", "688003"])
        self.assertAlmostEqual(constrained["target_weight"].sum(), 0.20)
        self.assertAlmostEqual(constrained.iloc[0]["target_weight"], 0.10)
        self.assertAlmostEqual(constrained.iloc[1]["target_weight"], 0.10)
        rejected = audit[audit["code"] == "688004"].iloc[0]
        self.assertEqual(rejected["reason"], "max_star_holdings_exceeded")

    def test_closed_gate_cannot_be_bypassed_by_calling_limit_function(self):
        constrained, audit = enforce_star_market_limits(
            self.candidates("688001"),
            self.closed,
        )

        self.assertTrue(constrained.empty)
        self.assertEqual(audit.iloc[0]["reason"], "star_market_entry_closed")


class StarMarketReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name) / "star_market_report"
        self.nav = pd.DataFrame({
            "date": ["2026-06-30", "2026-07-31", "2026-08-31"],
            "portfolio_equity": [1000.0, 1100.0, 900.0],
            "star_equity": [100.0, 120.0, 90.0],
            "star_daily_net_pnl": [0.0, 20.0, -30.0],
            "star_fees": [2.0, 3.0, 3.0],
            "star_slippage_cost": [1.0, 1.0, 1.0],
        })
        self.fills = pd.DataFrame([
            {
                "date": "2026-07-01",
                "code": "688001",
                "side": "BUY",
                "gross_value": 100.0,
                "fees": 5.0,
                "slippage_cost": 2.0,
            }
        ])
        self.audit = pd.DataFrame([
            {
                "code": "688001",
                "name": "华兴源创",
                "rank": 1,
                "decision": "selected",
                "reason": "selected_within_limits",
                "selection_reason": "综合排名第1",
                "universe_effective_date": "2026-06-30",
                "original_target_weight": 0.10,
                "final_target_weight": 0.10,
            },
            {
                "code": "688002",
                "name": "示例-U",
                "rank": 2,
                "decision": "excluded",
                "reason": "unprofitable_u_excluded",
                "selection_reason": "综合排名第2",
                "universe_effective_date": "2026-06-30",
                "original_target_weight": 0.10,
                "final_target_weight": 0.0,
            },
        ])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_report_separates_return_drawdown_costs_and_reasons(self):
        summary = write_star_market_report(
            self.nav,
            self.fills,
            self.audit,
            self.output,
        )

        self.assertAlmostEqual(summary["star_market_return_pct"], -10.0)
        self.assertAlmostEqual(summary["star_market_max_drawdown_pct"], -25.0)
        self.assertEqual(summary["fees"], 8.0)
        self.assertEqual(summary["slippage_cost"], 3.0)
        self.assertAlmostEqual(summary["latest_portfolio_weight_pct"], 10.0)
        self.assertEqual(
            summary["selection_reason_counts"]["selected_within_limits"],
            1,
        )
        self.assertEqual(
            summary["selection_reason_counts"]["unprofitable_u_excluded"],
            1,
        )
        self.assertEqual(len(summary["protocol_sha256"]), 64)

    def test_report_writes_only_independent_views(self):
        write_star_market_report(self.nav, self.fills, self.audit, self.output)

        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {
                "star_market_nav.csv",
                "star_market_fills.csv",
                "star_market_selection_audit.csv",
                "star_market_summary.json",
                "star_market_protocol.json",
                "star_market_report.html",
            },
        )

    def test_zero_star_equity_records_full_loss_and_drawdown(self):
        nav = self.nav.iloc[:2].copy()
        nav["star_equity"] = [100.0, 0.0]

        summary = write_star_market_report(
            nav,
            self.fills,
            self.audit,
            self.output,
        )

        self.assertEqual(summary["star_market_return_pct"], -100.0)
        self.assertEqual(summary["star_market_max_drawdown_pct"], -100.0)

    def test_report_refuses_v1_account_directory(self):
        with self.assertRaises(IsolationError):
            write_star_market_report(
                self.nav,
                self.fills,
                self.audit,
                PROJECT_ROOT / "data" / "paper_trading" / "star",
            )


class StarMarketCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "star_market.json"
        self.config_path.write_text(json.dumps({
            "enabled": False,
            "investor_permission_confirmed": False,
            "max_portfolio_weight": 0.20,
            "max_holdings": 2,
            "exclude_unprofitable_u": True,
        }), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_enable_without_permission_flag_fails_and_keeps_config_closed(self):
        exit_code = main(["--config", str(self.config_path), "enable"])

        config = load_star_market_config(self.config_path)
        self.assertEqual(exit_code, 2)
        self.assertFalse(config.enabled)
        self.assertFalse(config.investor_permission_confirmed)

    def test_enable_with_manual_confirmation_sets_both_flags(self):
        exit_code = main([
            "--config",
            str(self.config_path),
            "enable",
            "--permission-confirmed",
        ])

        config = load_star_market_config(self.config_path)
        self.assertEqual(exit_code, 0)
        self.assertTrue(config.enabled)
        self.assertTrue(config.investor_permission_confirmed)
        self.assertTrue(config.entry_open)
        activation = json.loads(
            self.config_path.with_name(
                "star_market_activation_protocol.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(activation["strategy_id"], "v2_monthly_top10_dd10")
        self.assertEqual(activation["subsystem"], "star_market")
        self.assertTrue(activation["config"]["enabled"])
        self.assertTrue(
            activation["config"]["investor_permission_confirmed"]
        )
        self.assertIn("effective_at", activation)
        self.assertEqual(len(activation["protocol_sha256"]), 64)

    def test_disable_closes_entry_but_preserves_permission_record(self):
        main([
            "--config",
            str(self.config_path),
            "enable",
            "--permission-confirmed",
        ])

        exit_code = main(["--config", str(self.config_path), "disable"])

        config = load_star_market_config(self.config_path)
        self.assertEqual(exit_code, 0)
        self.assertFalse(config.enabled)
        self.assertTrue(config.investor_permission_confirmed)
        self.assertFalse(config.entry_open)

    def test_revoke_permission_closes_entry_and_clears_both_flags(self):
        main([
            "--config",
            str(self.config_path),
            "enable",
            "--permission-confirmed",
        ])

        exit_code = main([
            "--config",
            str(self.config_path),
            "revoke-permission",
        ])

        config = load_star_market_config(self.config_path)
        self.assertEqual(exit_code, 0)
        self.assertFalse(config.enabled)
        self.assertFalse(config.investor_permission_confirmed)


if __name__ == "__main__":
    unittest.main()
