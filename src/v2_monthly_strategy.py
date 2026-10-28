"""Frozen infrastructure configuration for the second-generation strategy.

This module is deliberately separate from the official v1 paper account.  It
defines research rules only and never authorizes deployment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig
from src.drawdown_overlay import DrawdownRiskPolicy
from src.tradability import TradabilityRules, screen_tradable_codes


STRATEGY_ID = "v2_monthly_top10_dd10"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "backtest_results"
    / STRATEGY_ID
)


def build_v2_config() -> StrategyConfig:
    """Return a deterministic v2 research configuration without ``.env``."""
    config = StrategyConfig()
    config.strategy_name = "上证综指可交易A股月频Top10研究策略"
    config.version = "2.0-research"
    config.timeframe = "DAILY"
    config.profile = "v2_monthly_top10_dd10"
    config.baseline_mode = True

    config.portfolio.rebalance_frequency = "monthly"
    config.portfolio.top_n = 10
    config.portfolio.rank_buffer = 30
    config.portfolio.weighting = "equal_weight"
    config.portfolio.max_position_weight = 0.10

    config.sector_neutral.enabled = True
    config.sector_neutral.max_per_sector = 2

    config.universe.source = "sse_composite_a_pit"
    config.universe.snapshot_dir = "data/universe/sse_composite_a"
    config.universe.strict_point_in_time = True

    config.execution.initial_cash = 100_000.0
    config.execution.commission = 0.0005
    config.execution.min_commission = 5.0
    config.execution.transfer_fee_rate = 0.00001
    config.execution.stamp_duty_rate = 0.0005
    config.execution.historical_stamp_duty_rate = 0.001
    config.execution.stamp_duty_cutover_date = "2023-08-28"
    config.execution.slippage_bps = 5.0
    config.execution.max_volume_participation = 0.05
    config.execution.volume_lookback = 20
    config.execution.min_lot = 100
    config.execution.enforce_price_limits = True
    config.execution.stop_loss_pct = 0.0
    config.execution.trailing_stop_pct = 0.0
    config.execution.time_stop_months = 0

    config.risk_control.max_drawdown_hard_limit = 0.10
    config.risk_control.warning_dd_threshold = 0.07
    config.risk_control.circuit_breaker_dd_threshold = 0.09
    config.risk_control.recovery_dd_threshold = 0.05
    config.risk_control.warning_equity_cap = 0.40
    config.risk_control.circuit_breaker_equity_cap = 0.20
    config.risk_control.recovery_days = 3
    config.risk_control.liquidity_min_daily_volume = 50_000_000.0

    config.market_timing.enabled = False
    config.market_timing.index = "000001"
    config.scoring.sentiment_weight = 0.0
    config.sentiment_data.allow_price_proxy = False
    config.backtest.mode = "price_only"
    config.backtest.export_dir = str(DEFAULT_OUTPUT)
    return config


def v2_research_protocol() -> dict:
    """Return the frozen phase-one protocol for the v2 infrastructure."""
    config = build_v2_config()
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "strategy_label": "上证综指可交易A股月频Top10（研究基础设施）",
        "research_status": "infrastructure_only",
        "deployment_authorized": False,
        "primary_benchmark": "SSE Composite / 000001",
        "secondary_benchmark": "CSI 300 / 000300",
        "universe": {
            "source": config.universe.source,
            "snapshot_dir": config.universe.snapshot_dir,
            "strict_point_in_time": True,
            "include_star_market": False,
            "minimum_listing_days": 365,
            "minimum_20d_average_amount": 50_000_000.0,
            "minimum_lot": config.execution.min_lot,
        },
        "portfolio": {
            "rebalance_frequency": "monthly",
            "signal": "last_actual_session_close",
            "entry": "next_actual_session_open",
            "top_n": 10,
            "rank_buffer": 30,
            "max_per_sector": 2,
        },
        "risk": {
            "research_max_drawdown_limit": 0.10,
            "warning_drawdown": 0.07,
            "warning_equity_cap": 0.40,
            "defensive_drawdown": 0.09,
            "defensive_equity_cap": 0.20,
            "recovery_drawdown": 0.05,
            "recovery_days": 3,
            "loss_guarantee": False,
        },
        "decision_boundary": (
            "本阶段只建设和审计研究基础设施；因子尚未重新准入，"
            "不得生成正式买入授权或覆盖v1模拟盘。"
        ),
    }


def ensure_v2_protocol(
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    """Create the protocol once and reject any silent later mutation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "research_protocol.json"
    expected = v2_research_protocol()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError("现有v2研究协议与代码不一致，拒绝静默改写")
    else:
        path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class V2MonthlyBacktestEngine(BacktestEngine):
    """Backtest engine that adds the v2 tradability screen after membership."""

    def __init__(self, config: StrategyConfig | None = None):
        super().__init__(config or build_v2_config())
        self.universe_audit_rows: list[dict] = []
        self._universe_exclusion_frames: list[pd.DataFrame] = []

    @property
    def universe_exclusion_rows(self) -> pd.DataFrame:
        if not self._universe_exclusion_frames:
            return pd.DataFrame()
        return pd.concat(
            self._universe_exclusion_frames, ignore_index=True
        )

    def _tradability_rules(self) -> TradabilityRules:
        cfg = self.config
        return TradabilityRules(
            min_average_amount=cfg.risk_control.liquidity_min_daily_volume,
            amount_lookback=20,
            top_n=cfg.portfolio.top_n,
            initial_cash=cfg.execution.initial_cash,
            min_lot=cfg.execution.min_lot,
            commission=cfg.execution.commission,
            min_commission=cfg.execution.min_commission,
            transfer_fee_rate=cfg.execution.transfer_fee_rate,
            slippage_bps=cfg.execution.slippage_bps,
        )

    def build_drawdown_policy(self) -> DrawdownRiskPolicy:
        cfg = self.config
        return DrawdownRiskPolicy(
            initial_peak=cfg.execution.initial_cash,
            warning_threshold=cfg.risk_control.warning_dd_threshold,
            defensive_threshold=(
                cfg.risk_control.circuit_breaker_dd_threshold
            ),
            recovery_threshold=cfg.risk_control.recovery_dd_threshold,
            warning_equity_cap=cfg.risk_control.warning_equity_cap,
            defensive_equity_cap=(
                cfg.risk_control.circuit_breaker_equity_cap
            ),
            recovery_days=cfg.risk_control.recovery_days,
        )

    def filter_factor_frame_for_date(
        self,
        factor_date: str | pd.Timestamp,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        point_in_time = super().filter_factor_frame_for_date(
            factor_date, frame
        )
        if self.universe_store is None:
            return point_in_time.iloc[0:0].copy()
        snapshot = self.universe_store.snapshot_for(factor_date)
        eligible, details, audit = screen_tradable_codes(
            snapshot.members,
            self.price_data,
            factor_date,
            self._tradability_rules(),
        )
        self.universe_audit_rows.append(audit)
        self._universe_exclusion_frames.append(details)
        return point_in_time[
            point_in_time["code"].astype(str).str.zfill(6).isin(eligible)
        ].copy()
