"""冻结参数的真正未见样本审计。

研究样本在 2025-05-12 截止。本模块只检验该日之后的价格，不在看到
未见样本结果后增删候选、修改参数或放宽门槛。三类组合分别代表当前正式
基准、此前保留的低换手执行参考，以及历史总收益较好的低波动研究候选。

注意：三者在既有研究样本上都没有通过完整晋级门槛。因此，即使某一组合
通过本次 holdout 门槛，也只能进入重新论证，不能自动替换正式策略。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig
from src.low_volatility_portfolio_validation import (
    LowVolatilityBacktestEngine,
)
from src.price_feed import load_index_data
from src.walk_forward_attribution import _benchmark_return


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "frozen_holdout_2025_2026"
)
RESEARCH_CUTOFF = "2025-05-12"
HOLDOUT_START = "2025-05-13"
HOLDOUT_END = "2026-07-31"
HISTORY_START = "2020-01-01"
FROZEN_AT = "2026-07-31 15:45 CST"
REFERENCE_INITIAL_CASH = 100_000.0
MIN_HOLDOUT_MONTHS = 12

CANDIDATES = {
    "current_weekly_top5": {
        "label": "当前正式周频 Top 5",
        "role": "formal_baseline",
        "engine": "price",
        "timeframe": "WEEKLY",
        "rebalance_frequency": "weekly",
        "top_n": 5,
        "rank_buffer": 0,
        "max_per_sector": 1,
        "momentum_weight": 1.0,
        "pullback_weight": 1.0,
        "effective_momentum_days": 8,
        "effective_pullback_days": 18,
        "prior_research_passed": False,
    },
    "pullback90_monthly_top10_b30": {
        "label": "90日回撤月频低换手参考",
        "role": "execution_reference",
        "engine": "price",
        "timeframe": "DAILY",
        "rebalance_frequency": "monthly",
        "top_n": 10,
        "rank_buffer": 30,
        "max_per_sector": 2,
        "momentum_weight": 0.0,
        "pullback_weight": 1.0,
        "effective_momentum_days": 40,
        "effective_pullback_days": 90,
        "prior_research_passed": False,
    },
    "low_volatility60_monthly_top10_b30": {
        "label": "60日低波动月频研究候选",
        "role": "research_candidate",
        "engine": "low_volatility_60",
        "timeframe": "DAILY",
        "rebalance_frequency": "monthly",
        "top_n": 10,
        "rank_buffer": 30,
        "max_per_sector": 2,
        "momentum_weight": 1.0,
        "pullback_weight": 0.0,
        "effective_momentum_days": 60,
        "effective_pullback_days": None,
        "prior_research_passed": False,
    },
}

HOLDOUT_GATE = {
    "minimum_observed_months": MIN_HOLDOUT_MONTHS,
    "continuous_total_return_gt_pct": 0.0,
    "continuous_excess_return_gt_pp": 0.0,
    "median_monthly_return_gt_pct": 0.0,
    "median_monthly_excess_gt_pp": 0.0,
    "positive_month_ratio_gte": 0.60,
    "positive_excess_month_ratio_gte": 0.60,
    "continuous_sharpe_gt": 0.0,
    "maximum_drawdown_gt_pct": -30.0,
    "annualized_selection_turnover_lte_pct": 1500.0,
}


def frozen_protocol() -> dict:
    """返回在读取新价格前固定的实验协议。"""
    return {
        "frozen_at": FROZEN_AT,
        "research_cutoff": RESEARCH_CUTOFF,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
        "history_load_start": HISTORY_START,
        "reference_initial_cash": REFERENCE_INITIAL_CASH,
        "benchmark": "CSI 300 / 000300",
        "candidates": CANDIDATES,
        "gate": HOLDOUT_GATE,
        "execution": {
            "signal": "period-end close",
            "entry": "next trading session open",
            "weighting": "equal_weight",
            "commission": 0.0005,
            "minimum_commission": 5.0,
            "transfer_fee_rate": 0.00001,
            "stamp_duty_rate": 0.0005,
            "historical_stamp_duty_rate": 0.001,
            "stamp_duty_cutover_date": "2023-08-28",
            "slippage_bps": 5.0,
            "minimum_lot": 100,
            "maximum_volume_participation": 0.05,
            "volume_lookback": 20,
            "price_limit_enforcement": True,
            "individual_stops": False,
            "market_timing": False,
            "sentiment": False,
        },
        "decision_boundary": (
            "候选在旧样本均未通过完整晋级门槛；holdout通过只允许重新论证，"
            "不自动修改正式策略或授权实盘。"
        ),
    }


def ensure_frozen_protocol(output_dir: Path) -> tuple[Path, str]:
    """写入或核对协议；已有协议不允许被静默改写。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "frozen_protocol.json"
    expected = frozen_protocol()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError(
                "冻结协议与代码不一致；拒绝运行，以免在看过结果后改规则"
            )
    else:
        path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def frozen_base_config() -> StrategyConfig:
    """建立不依赖 .env 研究参数的冻结基础配置。"""
    config = StrategyConfig()
    config.timeframe = "DAILY"
    config.baseline_mode = True
    config.factors.momentum_window = 40
    config.factors.pullback_window = 90
    config.market_timing.enabled = False
    config.execution.initial_cash = REFERENCE_INITIAL_CASH
    config.execution.commission = 0.0005
    config.execution.min_commission = 5.0
    config.execution.transfer_fee_rate = 0.00001
    config.execution.stamp_duty_rate = 0.0005
    config.execution.historical_stamp_duty_rate = 0.001
    config.execution.stamp_duty_cutover_date = "2023-08-28"
    config.execution.slippage_bps = 5.0
    config.execution.min_lot = 100
    config.execution.max_volume_participation = 0.05
    config.execution.volume_lookback = 20
    config.execution.enforce_price_limits = True
    config.execution.stop_loss_pct = 0.0
    config.execution.trailing_stop_pct = 0.0
    config.execution.time_stop_months = 0
    config.scoring.sentiment_weight = 0.0
    config.sentiment_data.allow_price_proxy = False
    config.portfolio.weighting = "equal_weight"
    config.sector_neutral.enabled = True
    config.universe.source = "hs300_pit"
    config.universe.snapshot_dir = "data/universe/hs300"
    config.universe.strict_point_in_time = True
    return config


def configure_candidate(
    base_config: StrategyConfig,
    candidate: str,
) -> StrategyConfig:
    if candidate not in CANDIDATES:
        raise ValueError(f"未知冻结候选: {candidate}")
    details = CANDIDATES[candidate]
    config = copy.deepcopy(base_config)
    config.timeframe = details["timeframe"]
    config.portfolio.rebalance_frequency = details[
        "rebalance_frequency"
    ]
    config.portfolio.top_n = int(details["top_n"])
    config.portfolio.rank_buffer = int(details["rank_buffer"])
    config.sector_neutral.max_per_sector = int(
        details["max_per_sector"]
    )
    config.scoring.momentum_weight = float(
        details["momentum_weight"]
    )
    config.scoring.pullback_weight = float(
        details["pullback_weight"]
    )
    return config


def _attach_shared_data(
    engine: BacktestEngine,
    shared: BacktestEngine,
) -> None:
    engine.price_data = shared.price_data
    engine.universe_store = shared.universe_store
    engine.code_to_name = shared.code_to_name
    engine.code_to_sector = shared.code_to_sector
    engine.name_to_code = shared.name_to_code


def _candidate_engine(
    base_config: StrategyConfig,
    candidate: str,
) -> BacktestEngine:
    config = configure_candidate(base_config, candidate)
    engine_type = CANDIDATES[candidate]["engine"]
    if engine_type == "low_volatility_60":
        return LowVolatilityBacktestEngine(
            config,
            "low_volatility_60",
        )
    return BacktestEngine(config)


def build_monthly_return_rows(
    *,
    candidate: str,
    equity_curve: pd.DataFrame,
    index_data: pd.DataFrame,
    first_execution: pd.Timestamp,
    initial_cash: float,
    end: str,
) -> pd.DataFrame:
    """从连续路径计算月收益，首月包含首次建仓费用。"""
    if equity_curve.empty or index_data.empty:
        return pd.DataFrame()
    equity = pd.to_numeric(
        equity_curve.loc[
            (equity_curve.index >= first_execution)
            & (equity_curve.index <= pd.Timestamp(end)),
            "equity",
        ],
        errors="coerce",
    ).dropna()
    index_window = index_data.loc[
        (index_data.index >= first_execution)
        & (index_data.index <= pd.Timestamp(end))
    ].copy()
    if equity.empty or index_window.empty:
        return pd.DataFrame()

    first_index_row = index_window.iloc[0]
    benchmark_start = float(first_index_row["open"])
    if benchmark_start <= 0:
        return pd.DataFrame()

    strategy_month_end = equity.resample("ME").last()
    benchmark_month_end = (
        pd.to_numeric(index_window["close"], errors="coerce")
        .dropna()
        .resample("ME")
        .last()
        / benchmark_start
        * initial_cash
    )
    aligned = pd.concat(
        [
            strategy_month_end.rename("strategy_value"),
            benchmark_month_end.rename("benchmark_value"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    previous_strategy = aligned["strategy_value"].shift(1)
    previous_benchmark = aligned["benchmark_value"].shift(1)
    previous_strategy.iloc[0] = initial_cash
    previous_benchmark.iloc[0] = initial_cash
    strategy_return = (
        aligned["strategy_value"] / previous_strategy - 1
    ) * 100
    benchmark_return = (
        aligned["benchmark_value"] / previous_benchmark - 1
    ) * 100
    return pd.DataFrame({
        "candidate": candidate,
        "month": aligned.index.strftime("%Y-%m"),
        "month_end": aligned.index,
        "strategy_return_pct": strategy_return.to_numpy(),
        "benchmark_return_pct": benchmark_return.to_numpy(),
        "excess_return_pp": (
            strategy_return - benchmark_return
        ).to_numpy(),
    })


def summarize_holdout(
    *,
    continuous: dict,
    monthly_rows: pd.DataFrame,
) -> dict:
    """按预先声明的 holdout 门槛汇总单一候选。"""
    monthly_return = pd.to_numeric(
        monthly_rows["strategy_return_pct"], errors="coerce"
    ).dropna()
    monthly_excess = pd.to_numeric(
        monthly_rows["excess_return_pp"], errors="coerce"
    ).dropna()
    summary = {
        **continuous,
        "observed_months": int(len(monthly_rows)),
        "median_monthly_return": (
            float(monthly_return.median())
            if len(monthly_return) else np.nan
        ),
        "median_monthly_excess_return": (
            float(monthly_excess.median())
            if len(monthly_excess) else np.nan
        ),
        "positive_month_ratio": (
            float((monthly_return > 0).mean())
            if len(monthly_return) else 0.0
        ),
        "positive_excess_month_ratio": (
            float((monthly_excess > 0).mean())
            if len(monthly_excess) else 0.0
        ),
    }
    summary["holdout_passed"] = bool(
        summary["observed_months"] >= MIN_HOLDOUT_MONTHS
        and summary["total_return"] > 0
        and summary["excess_return"] > 0
        and summary["median_monthly_return"] > 0
        and summary["median_monthly_excess_return"] > 0
        and summary["positive_month_ratio"] >= 0.60
        and summary["positive_excess_month_ratio"] >= 0.60
        and summary["sharpe"] > 0
        and summary["max_drawdown"] > -30
        and summary["annualized_selection_turnover_pct"] <= 1500
    )
    return summary


def run_candidate(
    *,
    candidate: str,
    base_config: StrategyConfig,
    shared: BacktestEngine,
    index_data: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    engine = _candidate_engine(base_config, candidate)
    _attach_shared_data(engine, shared)
    result = engine.run(
        HOLDOUT_START,
        HOLDOUT_END,
        mode="price_only",
    )
    performance = result["performance"]
    execution = result["stats"].get("execution", {})
    trades = result["trade_log"]
    first_execution = (
        pd.Timestamp(trades[0]["date"]) if trades else pd.NaT
    )
    benchmark = (
        _benchmark_return(index_data, first_execution, HOLDOUT_END)
        if pd.notna(first_execution)
        else np.nan
    )
    frequency = CANDIDATES[candidate]["rebalance_frequency"]
    annualization = 52 if frequency == "weekly" else 12
    turnover = float(performance.get("avg_turnover_pct", 0.0))
    total_return = float(performance.get("total_return", 0.0))
    continuous = {
        "first_execution": (
            str(first_execution.date())
            if pd.notna(first_execution) else None
        ),
        "total_return": total_return,
        "benchmark_return": float(benchmark),
        "excess_return": total_return - float(benchmark),
        "cagr": float(performance.get("cagr", 0.0)),
        "sharpe": float(performance.get("sharpe", 0.0)),
        "max_drawdown": float(
            performance.get("max_drawdown", 0.0)
        ),
        "avg_turnover_pct": turnover,
        "annualized_selection_turnover_pct": (
            turnover * annualization
        ),
        "total_fees": float(execution.get("total_fees", 0.0)),
        "slippage_cost": float(
            execution.get("slippage_cost", 0.0)
        ),
        "constrained_orders": int(
            execution.get("constrained_orders", 0)
        ),
        "avg_deployment_pct": float(
            performance.get("avg_deployment_pct", 0.0)
        ),
        "trade_periods": int(len(trades)),
    }
    monthly = build_monthly_return_rows(
        candidate=candidate,
        equity_curve=result["equity_curve"],
        index_data=index_data,
        first_execution=first_execution,
        initial_cash=REFERENCE_INITIAL_CASH,
        end=HOLDOUT_END,
    )
    summary = summarize_holdout(
        continuous=continuous,
        monthly_rows=monthly,
    )
    equity = result["equity_curve"].copy().reset_index()
    equity["candidate"] = candidate
    return summary, monthly, equity


def build_data_audit(
    *,
    shared: BacktestEngine,
    index_data: pd.DataFrame,
) -> dict:
    if shared.universe_store is None:
        raise ValueError("未见样本审计要求历史时点股票池")
    requested_end = pd.Timestamp(HOLDOUT_END)
    index_actual_end = (
        pd.Timestamp(index_data.index.max())
        if not index_data.empty else pd.NaT
    )
    index_lag_calendar_days = (
        int((requested_end - index_actual_end.normalize()).days)
        if pd.notna(index_actual_end) else None
    )
    # 行情供应商可能在收盘后仍晚一个交易日更新。冻结协议的请求终点
    # 不改写，但只要落后不超过3个自然日（覆盖周末）且期末成分新鲜，
    # 就按供应商最新完整交易日运行，并在报告中显式披露实际终点。
    index_within_update_lag = bool(
        index_lag_calendar_days is not None
        and 0 <= index_lag_calendar_days <= 3
    )
    end_snapshot = shared.universe_store.snapshot_for(HOLDOUT_END)
    fresh_cutoff = index_actual_end - pd.Timedelta(days=7)
    fresh_codes = []
    for code in end_snapshot.codes:
        frame = shared.price_data.get(code)
        if frame is not None and not frame.empty:
            if pd.Timestamp(frame.index.max()) >= fresh_cutoff:
                fresh_codes.append(code)
    freshness = len(fresh_codes) / len(end_snapshot.codes)
    return {
        "requested_end": HOLDOUT_END,
        "index_actual_end": (
            str(index_actual_end.date())
            if pd.notna(index_actual_end) else None
        ),
        "index_reaches_requested_end": bool(
            pd.notna(index_actual_end)
            and index_actual_end.normalize() >= requested_end
        ),
        "index_lag_calendar_days": index_lag_calendar_days,
        "index_within_allowed_update_lag": index_within_update_lag,
        "loaded_stock_count": len(shared.price_data),
        "end_universe_effective_date": str(
            end_snapshot.effective_date.date()
        ),
        "end_universe_count": len(end_snapshot.codes),
        "fresh_end_universe_count": len(fresh_codes),
        "fresh_end_universe_ratio": freshness,
        "freshness_rule": (
            "期末成分至少95%在指数最后交易日前7个自然日内有行情"
        ),
        "index_update_lag_rule": (
            "指数源可比协议请求终点落后最多3个自然日，并披露实际终点"
        ),
        "passed": bool(
            index_within_update_lag
            and freshness >= 0.95
        ),
    }


def validate_outputs(
    *,
    summaries: pd.DataFrame,
    monthly: pd.DataFrame,
    equities: pd.DataFrame,
    data_audit: dict,
    protocol_sha256: str,
) -> dict:
    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })

    record(
        "protocol_hash_present",
        len(protocol_sha256) == 64,
        protocol_sha256,
    )
    record(
        "exact_frozen_candidate_set",
        set(summaries["candidate"]) == set(CANDIDATES),
        f"actual={sorted(summaries['candidate'].tolist())}",
    )
    record(
        "minimum_holdout_months",
        bool((summaries["observed_months"] >= MIN_HOLDOUT_MONTHS).all()),
        (
            f"minimum={int(summaries['observed_months'].min())}, "
            f"required={MIN_HOLDOUT_MONTHS}"
        ),
    )
    first_execution = pd.to_datetime(
        summaries["first_execution"], errors="coerce"
    )
    record(
        "no_pre_holdout_execution",
        bool((first_execution >= pd.Timestamp(HOLDOUT_START)).all()),
        f"earliest={first_execution.min().date()}",
    )
    equity_dates = pd.to_datetime(equities["date"], errors="coerce")
    record(
        "no_pre_holdout_equity_path",
        bool(equity_dates.min() >= pd.Timestamp(HOLDOUT_START)),
        f"earliest={equity_dates.min().date()}",
    )
    duplicates = int(
        monthly.duplicated(["candidate", "month"]).sum()
    )
    record(
        "unique_candidate_month_grain",
        duplicates == 0,
        f"duplicate_rows={duplicates}",
    )
    critical = [
        "total_return",
        "benchmark_return",
        "excess_return",
        "sharpe",
        "max_drawdown",
        "annualized_selection_turnover_pct",
        "median_monthly_return",
        "median_monthly_excess_return",
    ]
    missing = int(summaries[critical].isna().sum().sum())
    record(
        "critical_metrics_complete",
        missing == 0,
        f"missing_cells={missing}",
    )
    excess_difference = (
        summaries["total_return"]
        - summaries["benchmark_return"]
        - summaries["excess_return"]
    ).abs().max()
    record(
        "continuous_excess_recomputed",
        float(excess_difference) < 1e-9,
        f"max_difference={float(excess_difference):.3g}",
    )
    record(
        "fresh_market_data",
        bool(data_audit["passed"]),
        (
            f"index_end={data_audit['index_actual_end']}, "
            f"fresh_ratio={data_audit['fresh_end_universe_ratio']:.1%}"
        ),
    )
    passed = all(item["passed"] for item in checks)
    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
    }


def _fmt(value: float, digits: int = 2, suffix: str = "%") -> str:
    return f"{float(value):+.{digits}f}{suffix}"


def _failed_gate_labels(row) -> list[str]:
    failures = []
    if row.observed_months < MIN_HOLDOUT_MONTHS:
        failures.append("观察月数")
    if row.total_return <= 0:
        failures.append("总收益")
    if row.excess_return <= 0:
        failures.append("连续超额")
    if row.median_monthly_return <= 0:
        failures.append("月收益中位")
    if row.median_monthly_excess_return <= 0:
        failures.append("月超额中位")
    if row.positive_month_ratio < 0.60:
        failures.append("正收益月份")
    if row.positive_excess_month_ratio < 0.60:
        failures.append("跑赢月份")
    if row.sharpe <= 0:
        failures.append("Sharpe")
    if row.max_drawdown <= -30:
        failures.append("最大回撤")
    if row.annualized_selection_turnover_pct > 1500:
        failures.append("换手")
    return failures


def build_html_report(
    *,
    result: dict,
    summaries: pd.DataFrame,
    monthly: pd.DataFrame,
    output_path: Path,
) -> None:
    passed = summaries[summaries["holdout_passed"]]
    if passed.empty:
        hero_tag = "HOLDOUT FAIL / 正式策略不变"
        hero_title = "冻结候选没有通过真正未见样本门槛"
        hero_body = (
            "这次结果没有给出可部署策略。继续运行当前正式基准只能用于"
            "记录模拟盘，不能把它解释为已验证的赚钱系统。"
        )
    else:
        labels = "、".join(passed["label"].tolist())
        hero_tag = "HOLDOUT SIGNAL / 仍不可部署"
        hero_title = f"{labels}通过本次 holdout，但仍需重新论证"
        hero_body = (
            "通过只说明 2025-05-13 之后的固定规则表现满足门槛。由于这些"
            "候选在旧样本均未通过完整历史晋级标准，正式策略仍不修改。"
        )

    cards = []
    rows = []
    for row in summaries.itertuples():
        state = "通过 holdout" if row.holdout_passed else "未通过"
        failed_labels = _failed_gate_labels(row)
        failed_text = (
            "无"
            if not failed_labels
            else "、".join(failed_labels)
        )
        cards.append(
            "<article class='metric'>"
            f"<span>{html.escape(row.label)}</span>"
            f"<b>{_fmt(row.total_return)}</b>"
            f"<small>相对沪深300 {_fmt(row.excess_return, suffix='pp')} · "
            f"{state}</small><small>关键失败：{html.escape(failed_text)}</small>"
            "</article>"
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(row.label)}</th>"
            f"<td>{row.total_return:+.2f}%</td>"
            f"<td>{row.benchmark_return:+.2f}%</td>"
            f"<td>{row.excess_return:+.2f}pp</td>"
            f"<td>{row.sharpe:+.2f}</td>"
            f"<td>{row.max_drawdown:.2f}%</td>"
            f"<td>{row.positive_excess_month_ratio:.0%}</td>"
            f"<td>{row.annualized_selection_turnover_pct:.0f}%</td>"
            f"<td>{row.total_fees + row.slippage_cost:,.0f}元</td>"
            f"<td><span class='pill {'pass' if row.holdout_passed else 'fail'}'>"
            f"{state}</span></td></tr>"
        )

    months = sorted(monthly["month"].unique())
    heat_header = "".join(
        f"<div class='head'>{html.escape(month)}</div>"
        for month in months
    )
    heat_rows = []
    for candidate, details in CANDIDATES.items():
        current = (
            monthly[monthly["candidate"] == candidate]
            .set_index("month")
            .reindex(months)
        )
        cells = []
        for value in current["excess_return_pp"]:
            if pd.isna(value):
                cells.append("<div class='neutral'>—</div>")
            else:
                cells.append(
                    f"<div class='{'good' if value > 0 else 'bad'}'>"
                    f"{float(value):+.1f}</div>"
                )
        heat_rows.append(
            f"<div class='row-label'>{html.escape(details['label'])}</div>"
            + "".join(cells)
        )

    audit = result["data_audit"]
    protocol = result["protocol"]
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结策略真正未见样本审计</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f3f6fa;color:#172033;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;
line-height:1.62}} main{{max-width:1180px;margin:0 auto;padding:46px 24px 72px}}
.eyebrow{{color:#2f65a7;font-weight:800;letter-spacing:.08em}} h1{{font-size:40px;
margin:7px 0}} .sub{{color:#66758d}} .hero{{background:#172d46;color:#fff;
padding:34px 38px;border-radius:20px;margin:26px 0 20px;box-shadow:0 14px 38px #1a355322}}
.tag{{display:inline-block;background:#f4d9c6;color:#6b2f1d;padding:5px 11px;
border-radius:999px;font-size:13px;font-weight:800}} .hero h2{{font-size:29px;
margin:14px 0 8px}} .hero p{{color:#d7e1ed;max-width:900px}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px}}
.metric,section{{background:#fff;border:1px solid #dce4ee;border-radius:16px}}
.metric{{padding:18px}} .metric span{{display:block;color:#68788e;font-size:13px}}
.metric b{{display:block;font-size:25px;margin:5px 0}} .metric small{{display:block;color:#68788e}}
section{{padding:28px;margin:18px 0;overflow:auto}} h2{{font-size:25px;margin:0 0 10px}}
table{{width:100%;border-collapse:collapse;margin-top:12px;min-width:900px}}
th,td{{padding:11px 9px;border-bottom:1px solid #dbe3ed;text-align:right;font-size:13px}}
th:first-child,td:first-child{{text-align:left}} thead th{{background:#f5f7fa;color:#58677a}}
.pill{{display:inline-block;border-radius:999px;padding:3px 8px;font-weight:800}}
.pill.pass{{background:#dce9f5;color:#174c7c}} .pill.fail{{background:#f5e6d2;color:#754512}}
.heat-wrap{{overflow:auto}} .heat{{display:grid;
grid-template-columns:210px repeat({len(months)},54px);gap:1px;background:#dbe3ed;
border:1px solid #dbe3ed;margin-top:14px;min-width:{210 + 54 * len(months)}px}}
.heat>*{{background:#fff;padding:10px 5px;text-align:center;font-size:12px}}
.heat .head{{background:#f4f6f9;font-weight:700}} .row-label{{text-align:left;font-weight:700}}
.heat .good{{background:#dce9f5;color:#174c7c}} .heat .bad{{background:#f5e6d2;color:#754512}}
.heat .neutral{{color:#8693a5}} .callout{{border-left:4px solid #b97826;
background:#fbf6ee;padding:14px 17px;margin-top:16px}} code{{font-size:12px;
word-break:break-all}} a{{color:#245e9d}} footer{{text-align:center;color:#77869a;
margin-top:28px;font-size:13px}}
@media(max-width:780px){{main{{padding:27px 14px}} h1{{font-size:31px}}
.metrics{{grid-template-columns:1fr}} section{{padding:20px}}}}
</style></head><body><main>
<div class="eyebrow">组合审计 · 真正未见样本</div>
<h1>冻结策略在未来数据上经得住吗？</h1>
<div class="sub">研究截止 {RESEARCH_CUTOFF} · Holdout {HOLDOUT_START} 至数据源最新完整交易日
{audit['index_actual_end']}（协议请求 {HOLDOUT_END}） ·
历史成分股 · 次日开盘 · 真实交易摩擦</div>
<div class="hero"><span class="tag">{hero_tag}</span><h2>{hero_title}</h2>
<p>{hero_body}</p></div>
<div class="metrics">{''.join(cards)}</div>
<section><h2>连续 holdout 结果</h2>
<p>每个候选只运行一次连续路径；收益从各自首次实际成交开始与沪深300比较，
策略在首次成交前保持现金。月度稳定性来自同一条连续净值路径，不按月重启。</p>
<table><thead><tr><th>冻结候选</th><th>策略收益</th><th>沪深300</th>
<th>超额</th><th>Sharpe</th><th>最大回撤</th><th>跑赢月份</th>
<th>年化选择换手</th><th>费用+滑点</th><th>判定</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></section>
<section><h2>逐月超额：稳定性比一次总收益更重要</h2>
<p>单元格为策略月收益减沪深300月收益（百分点）。蓝色为跑赢，棕色为跑输；
首月包含首次建仓费用，最后一月为截至样本终点的完整交易数据。</p>
<div class="heat-wrap"><div class="heat"><div class="head">候选 / 月份</div>
{heat_header}{''.join(heat_rows)}</div></div></section>
<section><h2>冻结门槛与决策边界</h2>
<ul>
<li>至少 {MIN_HOLDOUT_MONTHS} 个观察月份；连续总收益和相对沪深300超额均大于0。</li>
<li>月收益及月超额中位数大于0；正收益月和跑赢月占比均至少60%。</li>
<li>连续 Sharpe 大于0、最大回撤优于−30%、年化选择换手不超过1500%。</li>
<li><strong>双门槛：</strong>旧研究门槛与本次 holdout 均通过才可能进入部署评审。
本轮三个候选的旧研究门槛均未通过，所以不会自动修改正式策略。</li>
</ul>
<div class="callout"><strong>PM动作：</strong>{html.escape(result['verdict']['decision'])}
</div></section>
<section><h2>数据与可复现性检查</h2>
<ul>
<li>指数最后交易日：{audit['index_actual_end']}；期末沪深300成分行情新鲜度：
{audit['fresh_end_universe_count']}/{audit['end_universe_count']}
（{audit['fresh_end_universe_ratio']:.1%}）。</li>
<li>期末股票池快照生效日：{audit['end_universe_effective_date']}；加载历史并集
{audit['loaded_stock_count']}只。</li>
<li>协议冻结时间：{protocol['frozen_at']}；SHA-256：
<code>{result['protocol_sha256']}</code>。</li>
<li>输出复核：{result['validation']['status']}；检查候选集合、观察月数、
执行时点、结果粒度、关键字段、超额复算与数据新鲜度。</li>
</ul></section>
<section><h2>限制与下一动作</h2><ul>
<li>这是一段约15个月、单一市场环境的 holdout，不等于长期收益保证。</li>
<li>当前结果只评估10万元参考资金；更大资金仍需单独做容量、冲击成本与整手压力测试。</li>
<li>候选旧样本表现均未达到完整晋级标准；holdout结果不能抹去此前失败。</li>
<li>下一步应按本报告判定决定：淘汰、重新论证，或继续只做前瞻模拟盘记录。</li>
</ul></section>
<section><h2>来源账本</h2><ul>
<li><a href="frozen_protocol.json">frozen_protocol.json</a>：看新数据前固定的候选与门槛。</li>
<li><a href="strategy_summary.csv">strategy_summary.csv</a>：连续收益、风险、换手和判定。</li>
<li><a href="monthly_returns.csv">monthly_returns.csv</a>：同一连续路径的逐月收益与超额。</li>
<li><a href="equity_curves.csv">equity_curves.csv</a>：候选日频净值审计路径。</li>
<li><a href="data_audit.json">data_audit.json</a>：指数和期末成分行情覆盖。</li>
<li><a href="validation_checks.json">validation_checks.json</a>：独立复核结果。</li>
</ul></section>
<footer>冻结参数 holdout 审计 · 不构成个股建议、收益承诺或实盘授权</footer>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def run_validation(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    protocol_path, protocol_sha256 = ensure_frozen_protocol(output_dir)
    logger.info(
        "冻结协议已核对，开始加载截至 holdout 终点的历史价格与成分股"
    )
    base_config = frozen_base_config()
    shared = BacktestEngine(base_config)
    shared.load_price_data(HISTORY_START, HOLDOUT_END)
    if shared.universe_store is None:
        raise ValueError("未见样本审计要求历史时点股票池")
    index_data = load_index_data("000300", HISTORY_START, HOLDOUT_END)
    if index_data.empty:
        raise ValueError("沪深300行情为空")
    data_audit = build_data_audit(
        shared=shared,
        index_data=index_data,
    )
    if not data_audit["passed"]:
        (output_dir / "data_audit.json").write_text(
            json.dumps(data_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ValueError(f"未见样本数据覆盖检查未通过: {data_audit}")

    summary_rows = []
    monthly_frames = []
    equity_frames = []
    for candidate, details in CANDIDATES.items():
        logger.info(f"运行冻结候选: {details['label']}")
        summary, monthly, equity = run_candidate(
            candidate=candidate,
            base_config=base_config,
            shared=shared,
            index_data=index_data,
        )
        summary_rows.append({
            "candidate": candidate,
            "label": details["label"],
            "role": details["role"],
            "prior_research_passed": details[
                "prior_research_passed"
            ],
            **summary,
            "deployment_gate_passed": bool(
                details["prior_research_passed"]
                and summary["holdout_passed"]
            ),
        })
        monthly_frames.append(monthly)
        equity_frames.append(equity)

    summaries = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    equities = pd.concat(equity_frames, ignore_index=True)
    validation = validate_outputs(
        summaries=summaries,
        monthly=monthly,
        equities=equities,
        data_audit=data_audit,
        protocol_sha256=protocol_sha256,
    )
    if validation["status"] != "passed":
        raise ValueError(f"未见样本输出复核失败: {validation}")

    holdout_passed = summaries.loc[
        summaries["holdout_passed"], "candidate"
    ].tolist()
    deployment_passed = summaries.loc[
        summaries["deployment_gate_passed"], "candidate"
    ].tolist()
    if deployment_passed:
        status = "deployment_review"
        decision = (
            "存在同时通过旧研究与holdout的候选，可进入资金容量和模拟盘部署评审；"
            "在评审完成前仍不实盘。"
        )
    elif holdout_passed:
        status = "re_underwrite_only"
        labels = "、".join(
            CANDIDATES[name]["label"] for name in holdout_passed
        )
        decision = (
            f"{labels}仅通过本次holdout；由于旧研究门槛失败，只能重新论证，"
            "正式策略保持不变。"
        )
    else:
        status = "rejected"
        decision = (
            "三个冻结候选均未通过holdout；停止围绕现有参数微调，"
            "正式策略仅保留为模拟盘记录，不视为可部署赚钱策略。"
        )

    result = {
        "protocol": json.loads(
            protocol_path.read_text(encoding="utf-8")
        ),
        "protocol_sha256": protocol_sha256,
        "data_audit": data_audit,
        "validation": validation,
        "holdout_passed_candidates": holdout_passed,
        "deployment_gate_passed_candidates": deployment_passed,
        "verdict": {
            "status": status,
            "decision": decision,
            "formal_strategy_changed": False,
        },
    }
    summaries.to_csv(
        output_dir / "strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly.to_csv(
        output_dir / "monthly_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    equities.to_csv(
        output_dir / "equity_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "data_audit.json").write_text(
        json.dumps(data_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_html_report(
        result=result,
        summaries=summaries,
        monthly=monthly,
        output_path=output_dir / "report.html",
    )
    logger.info(f"冻结 holdout 审计完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行冻结参数的真正未见样本审计"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_validation(output_dir=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
