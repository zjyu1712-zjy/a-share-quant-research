"""走步样本外收益归因。

基于已经完成的走步验证方案，逐窗口拆分：
1. 零摩擦毛收益、真实净收益和交易摩擦；
2. momentum、pullback、score_price 的多持有期 Rank IC；
3. 新进入、继续持有、被卖出股票的下一持有期收益；
4. 选股轮换、仓位再平衡和成交限制；
5. 相对沪深300的样本外收益。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.config import StrategyConfig
from src.dual_factor import build_combined_factors, build_price_factors
from src.price_feed import load_index_data
from src.walk_forward_validation import _candidate_builders


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WALK_FORWARD_DIR = (
    PROJECT_ROOT / "data" / "backtest_results" / "walk_forward_2020_2025"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "backtest_results" / "walk_forward_attribution"
)
HORIZONS = {1: 5, 2: 10, 4: 20, 8: 40}


def _base_config() -> StrategyConfig:
    config = StrategyConfig.from_env()
    config.timeframe = "WEEKLY"
    config.portfolio.rebalance_frequency = "weekly"
    config.market_timing.enabled = False
    config.execution.stop_loss_pct = 0.0
    config.execution.trailing_stop_pct = 0.0
    config.execution.time_stop_months = 0
    config.scoring.sentiment_weight = 0.0
    config.sentiment_data.allow_price_proxy = False
    return config


def _attach_shared_data(engine: BacktestEngine, shared: BacktestEngine) -> None:
    engine.price_data = shared.price_data
    engine.universe_store = shared.universe_store
    engine.code_to_name = shared.code_to_name
    engine.code_to_sector = shared.code_to_sector
    engine.name_to_code = shared.name_to_code


def _run_candidate(
    *,
    base_config: StrategyConfig,
    candidate: str,
    shared: BacktestEngine,
    start: str,
    end: str,
    frictionless: bool,
) -> dict:
    config = copy.deepcopy(base_config)
    builders = _candidate_builders()
    if candidate not in builders:
        raise ValueError(f"未知候选策略: {candidate}")
    builders[candidate](config)
    if frictionless:
        config.execution.commission = 0.0
        config.execution.min_commission = 0.0
        config.execution.transfer_fee_rate = 0.0
        config.execution.stamp_duty_rate = 0.0
        config.execution.historical_stamp_duty_rate = 0.0
        config.execution.slippage_bps = 0.0
    engine = BacktestEngine(config)
    _attach_shared_data(engine, shared)
    return engine.run(start, end, mode="price_only")


def _compound(returns_pct: Iterable[float]) -> float:
    values = np.asarray(list(returns_pct), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float((np.prod(1 + values / 100.0) - 1) * 100)


def _open_forward_return(
    frame: pd.DataFrame,
    signal_date: pd.Timestamp,
    periods: int,
) -> float:
    """信号产生后的下一交易日开盘至第 periods 个交易日开盘。"""
    if frame.empty or "open" not in frame.columns:
        return np.nan
    index = pd.DatetimeIndex(frame.index)
    entry = int(index.searchsorted(pd.Timestamp(signal_date), side="right"))
    exit_position = entry + periods
    if entry >= len(index) or exit_position >= len(index):
        return np.nan
    entry_price = float(frame.iloc[entry]["open"])
    exit_price = float(frame.iloc[exit_position]["open"])
    if entry_price <= 0 or exit_price <= 0:
        return np.nan
    return (exit_price / entry_price - 1) * 100


def _between_open_return(
    frame: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """两个实际执行日的开盘到开盘收益。"""
    if frame.empty or "open" not in frame.columns:
        return np.nan
    try:
        start_price = frame.loc[pd.Timestamp(start_date), "open"]
        end_price = frame.loc[pd.Timestamp(end_date), "open"]
    except KeyError:
        return np.nan
    if isinstance(start_price, pd.Series):
        start_price = start_price.iloc[-1]
    if isinstance(end_price, pd.Series):
        end_price = end_price.iloc[-1]
    start_value = float(start_price)
    end_value = float(end_price)
    if start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value - 1) * 100


def build_factor_ic(
    factors: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    factor_names: Iterable[str] = ("momentum", "pullback", "score_price"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算逐调仓日、多持有期的截面 Spearman Rank IC。"""
    records: list[dict] = []
    for signal_date, date_frame in factors.groupby("date"):
        working = date_frame.copy()
        for weeks, periods in HORIZONS.items():
            working[f"return_{weeks}w"] = [
                _open_forward_return(
                    price_data.get(str(code), pd.DataFrame()),
                    pd.Timestamp(signal_date),
                    periods,
                )
                for code in working["code"]
            ]
        for factor in factor_names:
            if factor not in working.columns:
                continue
            for weeks in HORIZONS:
                pair = working[[factor, f"return_{weeks}w"]].dropna()
                can_correlate = (
                    len(pair) >= 20
                    and pair[factor].nunique() > 1
                    and pair[f"return_{weeks}w"].nunique() > 1
                )
                ic = (
                    pair[factor].corr(
                        pair[f"return_{weeks}w"], method="spearman"
                    )
                    if can_correlate
                    else np.nan
                )
                records.append({
                    "date": str(pd.Timestamp(signal_date).date()),
                    "factor": factor,
                    "horizon_weeks": weeks,
                    "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                    "stock_count": int(len(pair)),
                })
    daily = pd.DataFrame(records)
    summaries: list[dict] = []
    if not daily.empty:
        for (factor, weeks), group in daily.groupby(
            ["factor", "horizon_weeks"], sort=False
        ):
            values = group["rank_ic"].dropna()
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            t_stat = (
                float(values.mean() / (std / np.sqrt(len(values))))
                if std > 0
                else 0.0
            )
            summaries.append({
                "factor": factor,
                "horizon_weeks": int(weeks),
                "weeks": int(len(values)),
                "mean_rank_ic": float(values.mean()) if len(values) else np.nan,
                "median_rank_ic": (
                    float(values.median()) if len(values) else np.nan
                ),
                "positive_ic_ratio": (
                    float((values > 0).mean()) if len(values) else np.nan
                ),
                "ic_t_stat": t_stat,
            })
    return daily, pd.DataFrame(summaries)


def summarize_ic_by(
    daily: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """按指定维度汇总逐期 IC。"""
    summaries: list[dict] = []
    if daily.empty:
        return pd.DataFrame()
    for keys, group in daily.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["rank_ic"].dropna()
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row = dict(zip(group_columns, keys))
        row.update({
            "weeks": int(len(values)),
            "mean_rank_ic": float(values.mean()) if len(values) else np.nan,
            "median_rank_ic": (
                float(values.median()) if len(values) else np.nan
            ),
            "positive_ic_ratio": (
                float((values > 0).mean()) if len(values) else np.nan
            ),
            "ic_t_stat": (
                float(values.mean() / (std / np.sqrt(len(values))))
                if std > 0
                else 0.0
            ),
        })
        summaries.append(row)
    return pd.DataFrame(summaries)


def build_cohort_attribution(
    *,
    fold: int,
    trade_log: list[dict],
    price_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拆分新进入、继续持有和被卖出股票的下一调仓期收益。"""
    cohort_rows: list[dict] = []
    turnover_rows: list[dict] = []
    for position in range(len(trade_log)):
        current = trade_log[position]
        previous_target = (
            set(trade_log[position - 1].get("target", []))
            if position > 0
            else set()
        )
        current_target = set(current.get("target", []))
        retained = current_target & previous_target
        entered = current_target - previous_target
        exited = previous_target - current_target

        buys = current.get("buys", [])
        sells = current.get("sells", [])
        turnover_rows.append({
            "fold": fold,
            "execution_date": str(pd.Timestamp(current["date"]).date()),
            "is_initial": position == 0,
            "previous_count": len(previous_target),
            "target_count": len(current_target),
            "entered_count": len(entered),
            "retained_count": len(retained),
            "exited_count": len(exited),
            "selection_turnover_pct": (
                (len(entered) + len(exited)) / len(previous_target) * 100
                if previous_target
                else np.nan
            ),
            "entry_buy_value": float(sum(
                item.get("gross_value", 0.0)
                for item in buys if item.get("code") in entered
            )),
            "retained_topup_value": float(sum(
                item.get("gross_value", 0.0)
                for item in buys if item.get("code") in retained
            )),
            "exit_sell_value": float(sum(
                item.get("gross_value", 0.0)
                for item in sells if item.get("note") == "rebalance_exit"
            )),
            "retained_trim_value": float(sum(
                item.get("gross_value", 0.0)
                for item in sells if item.get("note") == "rebalance_trim"
            )),
            "constrained_orders": len(current.get("constraints", [])),
        })

        if position + 1 >= len(trade_log):
            continue
        next_date = pd.Timestamp(trade_log[position + 1]["date"])
        execution_date = pd.Timestamp(current["date"])
        groups = {
            "new_entry": entered,
            "retained": retained,
            "exited_missed": exited,
        }
        for cohort, codes in groups.items():
            for code in codes:
                value = _between_open_return(
                    price_data.get(code, pd.DataFrame()),
                    execution_date,
                    next_date,
                )
                if pd.notna(value):
                    cohort_rows.append({
                        "fold": fold,
                        "execution_date": str(execution_date.date()),
                        "next_execution_date": str(next_date.date()),
                        "cohort": cohort,
                        "code": code,
                        "return_pct": float(value),
                        "is_initial": position == 0,
                    })
    return pd.DataFrame(cohort_rows), pd.DataFrame(turnover_rows)


def _benchmark_return(
    index_data: pd.DataFrame,
    first_execution_date: pd.Timestamp,
    end: str,
) -> float:
    if index_data.empty:
        return np.nan
    end_date = pd.Timestamp(end)
    start_rows = index_data[index_data.index >= first_execution_date]
    end_rows = index_data[index_data.index <= end_date]
    if start_rows.empty or end_rows.empty:
        return np.nan
    start_price = float(start_rows.iloc[0]["open"])
    end_price = float(end_rows.iloc[-1]["close"])
    if start_price <= 0 or end_price <= 0:
        return np.nan
    return (end_price / start_price - 1) * 100


def _load_plan(path: Path) -> pd.DataFrame:
    plan = pd.read_csv(path)
    required = {
        "fold", "selected_candidate", "test_start", "test_end",
    }
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"走步选择文件缺少字段: {sorted(missing)}")
    return plan.sort_values("fold").reset_index(drop=True)


def _round_records(frame: pd.DataFrame, digits: int = 6) -> list[dict]:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(digits)
    return result.replace({np.nan: None}).to_dict(orient="records")


def run_attribution(
    *,
    walk_forward_dir: Path = DEFAULT_WALK_FORWARD_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    plan = _load_plan(walk_forward_dir / "adaptive_selection_folds.csv")
    history_start = "2020-01-01"
    history_end = str(plan["test_end"].max())
    oos_start = str(plan["test_start"].min())

    base_config = _base_config()
    shared = BacktestEngine(base_config)
    logger.info("加载价格、历史股票池与沪深300基准")
    shared.load_price_data(history_start, history_end)
    if shared.universe_store is None:
        raise ValueError("收益归因要求历史时点股票池")
    index_data = load_index_data("000300", history_start, history_end)
    if index_data.empty:
        raise ValueError("沪深300数据为空，无法完成相对收益归因")

    fold_rows: list[dict] = []
    cohort_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []
    selected_horizon_rows: list[dict] = []
    for row in plan.itertuples(index=False):
        logger.info(
            f"归因 Fold {row.fold}: {row.selected_candidate}, "
            f"{row.test_start}~{row.test_end}"
        )
        realistic = _run_candidate(
            base_config=base_config,
            candidate=row.selected_candidate,
            shared=shared,
            start=row.test_start,
            end=row.test_end,
            frictionless=False,
        )
        frictionless = _run_candidate(
            base_config=base_config,
            candidate=row.selected_candidate,
            shared=shared,
            start=row.test_start,
            end=row.test_end,
            frictionless=True,
        )
        net_return = float(realistic["performance"]["total_return"])
        gross_return = float(frictionless["performance"]["total_return"])
        execution = realistic["stats"]["execution"]
        trade_log = realistic["trade_log"]
        first_execution = (
            pd.Timestamp(trade_log[0]["date"]) if trade_log else pd.NaT
        )
        previous_target: set[str] = set()
        for trade in trade_log:
            signal_date = pd.Timestamp(trade["signal_date"])
            current_target = set(trade.get("target", []))
            for code in current_target:
                cohort = (
                    "retained" if code in previous_target else "new_entry"
                )
                for horizon_weeks, periods in HORIZONS.items():
                    selected_horizon_rows.append({
                        "fold": int(row.fold),
                        "signal_date": str(signal_date.date()),
                        "code": code,
                        "cohort": cohort,
                        "horizon_weeks": horizon_weeks,
                        "return_pct": _open_forward_return(
                            shared.price_data.get(code, pd.DataFrame()),
                            signal_date,
                            periods,
                        ),
                    })
            previous_target = current_target
        benchmark = (
            _benchmark_return(index_data, first_execution, row.test_end)
            if pd.notna(first_execution)
            else np.nan
        )
        fold_rows.append({
            "fold": int(row.fold),
            "candidate": row.selected_candidate,
            "test_start": row.test_start,
            "test_end": row.test_end,
            "gross_return_pct": gross_return,
            "net_return_pct": net_return,
            "friction_drag_pp": gross_return - net_return,
            "benchmark_return_pct": benchmark,
            "net_excess_return_pp": (
                net_return - benchmark if pd.notna(benchmark) else np.nan
            ),
            "gross_excess_return_pp": (
                gross_return - benchmark if pd.notna(benchmark) else np.nan
            ),
            "total_fees": float(execution.get("total_fees", 0.0)),
            "slippage_cost": float(execution.get("slippage_cost", 0.0)),
            "gross_traded_value": float(execution.get("gross_turnover", 0.0)),
            "constrained_orders": int(
                execution.get("constrained_orders", 0)
            ),
            "avg_selection_turnover_pct": float(
                realistic["performance"].get("avg_turnover_pct", 0.0)
            ),
        })
        cohorts, turnovers = build_cohort_attribution(
            fold=int(row.fold),
            trade_log=trade_log,
            price_data=shared.price_data,
        )
        cohort_frames.append(cohorts)
        turnover_frames.append(turnovers)

    folds = pd.DataFrame(fold_rows)
    cohorts = (
        pd.concat(cohort_frames, ignore_index=True)
        if cohort_frames else pd.DataFrame()
    )
    turnovers = (
        pd.concat(turnover_frames, ignore_index=True)
        if turnover_frames else pd.DataFrame()
    )
    selected_horizons = pd.DataFrame(selected_horizon_rows)

    rebalance_dates = _generate_rebalance_dates(
        oos_start,
        history_end,
        base_config.portfolio.rebalance_frequency,
        shared.price_data,
    )
    factors = build_price_factors(
        shared.price_data,
        rebalance_dates,
        base_config.factors.for_timeframe(base_config.timeframe),
    )
    combined = build_combined_factors(
        factors,
        None,
        base_config.scoring,
    )
    filtered: list[pd.DataFrame] = []
    for date, date_frame in combined.groupby("date"):
        eligible = shared.universe_store.snapshot_for(date).codes
        filtered.append(
            date_frame[
                date_frame["code"].astype(str).str.zfill(6).isin(eligible)
            ]
        )
    combined = (
        pd.concat(filtered, ignore_index=True)
        if filtered else combined.iloc[0:0].copy()
    )
    ic_daily, ic_summary = build_factor_ic(combined, shared.price_data)
    ic_daily["fold"] = pd.NA
    ic_dates = pd.to_datetime(ic_daily["date"])
    for row in plan.itertuples(index=False):
        mask = (
            (ic_dates >= pd.Timestamp(row.test_start))
            & (ic_dates <= pd.Timestamp(row.test_end))
        )
        ic_daily.loc[mask, "fold"] = int(row.fold)
    ic_daily["fold"] = ic_daily["fold"].astype("Int64")
    ic_by_fold = summarize_ic_by(
        ic_daily.dropna(subset=["fold"]),
        ["fold", "factor", "horizon_weeks"],
    )

    cohort_summary = pd.DataFrame()
    if not cohorts.empty:
        cohort_summary = (
            cohorts[
                ~(
                    cohorts["is_initial"]
                    & cohorts["cohort"].eq("new_entry")
                )
            ]
            .groupby("cohort")["return_pct"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(columns={
                "count": "observations",
                "mean": "mean_return_pct",
                "median": "median_return_pct",
            })
        )

    holding_period_summary = pd.DataFrame()
    if not selected_horizons.empty:
        holding_period_summary = (
            selected_horizons.dropna(subset=["return_pct"])
            .groupby("horizon_weeks")["return_pct"]
            .agg(
                observations="count",
                mean_return_pct="mean",
                median_return_pct="median",
                positive_return_ratio=lambda values: (values > 0).mean(),
            )
            .reset_index()
        )

    non_initial_turnover = turnovers[~turnovers["is_initial"]].copy()
    traded_source_columns = [
        "entry_buy_value",
        "retained_topup_value",
        "exit_sell_value",
        "retained_trim_value",
    ]
    traded_sources = {
        column: float(non_initial_turnover[column].sum())
        for column in traded_source_columns
    }
    traded_total = sum(traded_sources.values())
    traded_source_ratios = {
        column.replace("_value", "_ratio"): (
            value / traded_total if traded_total > 0 else 0.0
        )
        for column, value in traded_sources.items()
    }

    gross_compound = _compound(folds["gross_return_pct"])
    net_compound = _compound(folds["net_return_pct"])
    benchmark_compound = _compound(folds["benchmark_return_pct"])
    score_ic_1w = ic_summary[
        (ic_summary["factor"] == "score_price")
        & (ic_summary["horizon_weeks"] == 1)
    ]
    score_ic_1w_value = (
        float(score_ic_1w.iloc[0]["mean_rank_ic"])
        if not score_ic_1w.empty
        else np.nan
    )
    primary_cause = (
        "factor_and_selection"
        if gross_compound <= 0
        else "trading_friction"
    )
    summary = {
        "period": {
            "start": oos_start,
            "end": history_end,
            "folds": int(len(folds)),
        },
        "portfolio_attribution": {
            "compounded_gross_return_pct": gross_compound,
            "compounded_net_return_pct": net_compound,
            "compounded_friction_drag_pp": gross_compound - net_compound,
            "compounded_benchmark_return_pct": benchmark_compound,
            "compounded_net_excess_return_pp": (
                net_compound - benchmark_compound
            ),
            "profitable_gross_folds": int(
                (folds["gross_return_pct"] > 0).sum()
            ),
            "profitable_net_folds": int(
                (folds["net_return_pct"] > 0).sum()
            ),
            "average_selection_turnover_pct": float(
                folds["avg_selection_turnover_pct"].mean()
            ),
            "total_fees_across_reset_folds": float(folds["total_fees"].sum()),
            "total_slippage_across_reset_folds": float(
                folds["slippage_cost"].sum()
            ),
        },
        "factor_diagnostic": {
            "score_price_1w_mean_rank_ic": score_ic_1w_value,
            "primary_cause": primary_cause,
            "interpretation": (
                "零摩擦样本外收益仍为负，主要问题是价格因子与选股，"
                "交易摩擦是次要放大项。"
                if primary_cause == "factor_and_selection"
                else "零摩擦样本外收益为正但真实收益转负，主要问题是交易摩擦。"
            ),
        },
        "cohort_summary": _round_records(cohort_summary),
        "selected_holding_period_summary": _round_records(
            holding_period_summary
        ),
        "turnover_sources": {
            **traded_sources,
            **traded_source_ratios,
            "average_retention_ratio": (
                float(
                    (
                        non_initial_turnover["retained_count"]
                        / non_initial_turnover["previous_count"].replace(0, np.nan)
                    ).mean()
                )
                if not non_initial_turnover.empty
                else 0.0
            ),
            "full_replacement_ratio": (
                float((non_initial_turnover["retained_count"] == 0).mean())
                if not non_initial_turnover.empty
                else 0.0
            ),
        },
        "factor_ic_summary": _round_records(ic_summary),
        "factor_ic_by_fold": _round_records(ic_by_fold),
    }

    score_ic_fold_table = (
        ic_by_fold[ic_by_fold["factor"] == "score_price"]
        .pivot(
            index="fold",
            columns="horizon_weeks",
            values="mean_rank_ic",
        )
        .rename(columns=lambda weeks: f"{weeks}w")
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(
        output_dir / "fold_attribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cohorts.to_csv(
        output_dir / "cohort_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    turnovers.to_csv(
        output_dir / "turnover_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_horizons.to_csv(
        output_dir / "selected_holding_period_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ic_daily.to_csv(
        output_dir / "factor_ic_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ic_summary.to_csv(
        output_dir / "factor_ic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ic_by_fold.to_csv(
        output_dir / "factor_ic_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# 走步样本外收益归因",
        "",
        f"- 样本外区间：{oos_start} 至 {history_end}",
        f"- 窗口数：{len(folds)}",
        "- 毛收益：保留整手、停牌、涨跌停和容量限制，仅移除费用与滑点",
        "- 净收益：下一交易日开盘成交，包含全部真实费用和滑点",
        "- 基准：从各窗口首次策略成交日开盘持有沪深300至窗口末收盘",
        "",
        "## 核心结论",
        "",
        f"- 零摩擦复合收益：{gross_compound:.2f}%",
        f"- 真实净复合收益：{net_compound:.2f}%",
        f"- 交易摩擦拖累：{gross_compound - net_compound:.2f}个百分点",
        f"- 沪深300复合收益：{benchmark_compound:.2f}%",
        f"- 策略净超额收益：{net_compound - benchmark_compound:.2f}个百分点",
        f"- 价格综合分1周平均 Rank IC：{score_ic_1w_value:.4f}",
        f"- 主要原因：{summary['factor_diagnostic']['interpretation']}",
        "",
        "## 各窗口毛收益、净收益与基准",
        "",
        folds[[
            "fold", "candidate", "test_start", "test_end",
            "gross_return_pct", "net_return_pct", "friction_drag_pp",
            "benchmark_return_pct", "net_excess_return_pp",
            "avg_selection_turnover_pct",
        ]].to_markdown(index=False),
        "",
        "## 因子 Rank IC",
        "",
        ic_summary.to_markdown(index=False),
        "",
        "完整的逐窗口、逐因子结果见 `factor_ic_by_fold.csv`。",
        "",
        "### 各窗口价格综合分 Rank IC",
        "",
        score_ic_fold_table.to_markdown(index=False),
        "",
        "## 入选股票的不同持有期收益",
        "",
        (
            holding_period_summary.to_markdown(index=False)
            if not holding_period_summary.empty
            else "无可用持有期数据。"
        ),
        "",
        "## 信号群组下一持有期收益",
        "",
        (
            cohort_summary.to_markdown(index=False)
            if not cohort_summary.empty
            else "无可用群组数据。"
        ),
        "",
        "## 换手来源",
        "",
        f"- 平均持仓保留比例：{summary['turnover_sources']['average_retention_ratio']:.1%}",
        f"- 完全换仓事件比例：{summary['turnover_sources']['full_replacement_ratio']:.1%}",
        f"- 新进入股票买入占成交来源：{traded_source_ratios['entry_buy_ratio']:.1%}",
        f"- 继续持有股票加仓占成交来源：{traded_source_ratios['retained_topup_ratio']:.1%}",
        f"- 退出股票卖出占成交来源：{traded_source_ratios['exit_sell_ratio']:.1%}",
        f"- 继续持有股票减仓占成交来源：{traded_source_ratios['retained_trim_ratio']:.1%}",
        "",
        "注：各走步窗口独立以10万元重新开始，费用金额用于窗口归因，"
        "不能直接视作一条连续实盘账户的绝对费用。",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    logger.info(f"走步收益归因完成: {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="走步样本外收益归因")
    parser.add_argument(
        "--walk-forward-dir",
        type=Path,
        default=DEFAULT_WALK_FORWARD_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_attribution(
        walk_forward_dir=args.walk_forward_dir,
        output_dir=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
