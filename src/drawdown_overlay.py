"""Portfolio-level drawdown risk state machine without look-ahead."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskDecision:
    date: pd.Timestamp
    state: str
    equity_cap: float
    drawdown: float
    reason: str


class DrawdownRiskPolicy:
    """Translate prior-close drawdown into the next-open exposure cap."""

    def __init__(
        self,
        initial_peak: float,
        warning_threshold: float,
        defensive_threshold: float,
        recovery_threshold: float,
        warning_equity_cap: float,
        defensive_equity_cap: float,
        recovery_days: int,
    ):
        if initial_peak <= 0:
            raise ValueError("initial_peak必须为正数")
        if not 0 <= recovery_threshold < warning_threshold < defensive_threshold < 1:
            raise ValueError("回撤阈值必须满足 recovery < warning < defensive")
        if not 0 <= defensive_equity_cap <= warning_equity_cap <= 1:
            raise ValueError("股票仓位上限必须满足 defensive <= warning <= 1")
        if recovery_days <= 0:
            raise ValueError("recovery_days必须为正整数")
        self.peak_equity = float(initial_peak)
        self.warning_threshold = float(warning_threshold)
        self.defensive_threshold = float(defensive_threshold)
        self.recovery_threshold = float(recovery_threshold)
        self.warning_equity_cap = float(warning_equity_cap)
        self.defensive_equity_cap = float(defensive_equity_cap)
        self.recovery_days = int(recovery_days)
        self.state = "normal"
        self.equity_cap = 1.0
        self._recovery_step = 0
        self._recovery_start_cap = 1.0

    def observe(
        self,
        date: pd.Timestamp,
        prior_close_equity: float,
    ) -> RiskDecision:
        equity = float(prior_close_equity)
        if equity <= 0:
            raise ValueError("prior_close_equity必须为正数")
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = max(0.0, 1.0 - equity / self.peak_equity)
        tolerance = 1e-12

        if drawdown + tolerance >= self.defensive_threshold:
            self.state = "defensive"
            self.equity_cap = self.defensive_equity_cap
            self._recovery_step = 0
            reason = "drawdown_reached_defensive_threshold"
        elif drawdown + tolerance >= self.warning_threshold:
            self.state = "warning"
            self.equity_cap = self.warning_equity_cap
            self._recovery_step = 0
            reason = "drawdown_reached_warning_threshold"
        elif (
            self.state in {"warning", "defensive", "recovery"}
            and drawdown <= self.recovery_threshold + tolerance
        ):
            if self.state != "recovery":
                self._recovery_start_cap = self.equity_cap
                self._recovery_step = 0
            self._recovery_step += 1
            fraction = min(self._recovery_step / self.recovery_days, 1.0)
            self.equity_cap = self._recovery_start_cap + (
                1.0 - self._recovery_start_cap
            ) * fraction
            if self._recovery_step >= self.recovery_days:
                self.state = "normal"
                self.equity_cap = 1.0
                reason = "recovery_completed"
            else:
                self.state = "recovery"
                reason = "staged_recovery"
        elif self.state in {"warning", "defensive", "recovery"}:
            reason = "hold_risk_cap_until_recovery_threshold"
        else:
            self.state = "normal"
            self.equity_cap = 1.0
            reason = "normal"

        return RiskDecision(
            date=pd.Timestamp(date),
            state=self.state,
            equity_cap=float(self.equity_cap),
            drawdown=float(drawdown),
            reason=reason,
        )
