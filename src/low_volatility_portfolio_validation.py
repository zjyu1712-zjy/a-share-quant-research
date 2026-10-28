"""低波动候选因子的组合级研究验证。

所有候选共用月频 Top 10、排名缓冲30、每行业最多2只、等权、真实费用与
下一交易日开盘成交。候选因子来自此前筛选，因此七个窗口用于组合级稳健性
诊断，不应被描述为全新未见样本。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig
from src.price_feed import load_index_data
from src.walk_forward_attribution import _benchmark_return
from src.walk_forward_validation import Fold, generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "low_volatility_portfolio_validation"
)
CANDIDATES = {
    "pullback_90_reference": {
        "label": "90日回撤参考",
        "momentum_weight": 1.0,
        "pullback_weight": 0.0,
        "eligible_for_advancement": False,
    },
    "low_volatility_60": {
        "label": "60日低波动",
        "momentum_weight": 1.0,
        "pullback_weight": 0.0,
        "eligible_for_advancement": True,
    },
    "sector_neutral_low_volatility_60": {
        "label": "行业中性60日低波动",
        "momentum_weight": 1.0,
        "pullback_weight": 0.0,
        "eligible_for_advancement": True,
    },
    "balanced_low_volatility_60": {
        "label": "原始/行业中性低波动各50%",
        "momentum_weight": 0.5,
        "pullback_weight": 0.5,
        "eligible_for_advancement": True,
    },
}


def build_low_volatility_factor_frame(
    *,
    price_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
    candidate: str,
) -> pd.DataFrame:
    """只使用信号日及此前日线，生成可交给统一回测器的因子表。"""
    if candidate not in CANDIDATES:
        raise ValueError(f"未知低波动组合候选: {candidate}")

    records: list[dict] = []
    for code, frame in price_data.items():
        if frame.empty or "close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").sort_index()
        daily_return = close.pct_change()
        low_volatility = -daily_return.rolling(
            60, min_periods=60
        ).std() * np.sqrt(252)
        pullback_90 = close / close.rolling(
            90, min_periods=90
        ).max() - 1
        for date in rebalance_dates:
            if date not in close.index:
                continue
            raw_low_vol = low_volatility.get(date, np.nan)
            reference = pullback_90.get(date, np.nan)
            if pd.isna(raw_low_vol) or pd.isna(reference):
                continue
            records.append({
                "date": pd.Timestamp(date),
                "code": str(code).zfill(6),
                "close": float(close.loc[date]),
                "low_volatility_60": float(raw_low_vol),
                "pullback_90_reference": float(reference),
            })

    raw = pd.DataFrame(records)
    if raw.empty:
        return pd.DataFrame(
            columns=["date", "code", "close", "momentum", "pullback"]
        )

    filtered: list[pd.DataFrame] = []
    for date, date_frame in raw.groupby("date"):
        snapshot = universe_store.snapshot_for(date)
        current = date_frame[
            date_frame["code"].isin(snapshot.codes)
        ].copy()
        current["sector"] = (
            current["code"].map(snapshot.code_to_sector).fillna("未知")
        )
        current["sector_neutral_low_volatility_60"] = (
            current["low_volatility_60"]
            - current.groupby("sector")["low_volatility_60"].transform(
                "median"
            )
        )
        filtered.append(current)

    result = pd.concat(filtered, ignore_index=True)
    if candidate == "pullback_90_reference":
        result["momentum"] = result["pullback_90_reference"]
        result["pullback"] = 0.0
    elif candidate == "low_volatility_60":
        result["momentum"] = result["low_volatility_60"]
        result["pullback"] = 0.0
    elif candidate == "sector_neutral_low_volatility_60":
        result["momentum"] = (
            result["sector_neutral_low_volatility_60"]
        )
        result["pullback"] = 0.0
    else:
        result["momentum"] = result["low_volatility_60"]
        result["pullback"] = (
            result["sector_neutral_low_volatility_60"]
        )

    logger.info(
        f"{candidate}: {len(result)}条因子记录, "
        f"{result['code'].nunique()}只股票, "
        f"{result['date'].nunique()}个调仓日"
    )
    return result


class LowVolatilityBacktestEngine(BacktestEngine):
    """仅覆写因子构建；选股、成交、费用和绩效均复用统一回测器。"""

    def __init__(self, config: StrategyConfig, candidate: str):
        super().__init__(config)
        self.candidate = candidate

    def build_price_factor_frame(
        self,
        rebalance_dates: pd.DatetimeIndex,
        adjusted_factors,
    ) -> pd.DataFrame:
        if self.universe_store is None:
            raise ValueError("低波动组合验证要求历史时点股票池")
        return build_low_volatility_factor_frame(
            price_data=self.price_data,
            rebalance_dates=rebalance_dates,
            universe_store=self.universe_store,
            candidate=self.candidate,
        )


def _base_config(initial_cash: float) -> StrategyConfig:
    config = StrategyConfig.from_env()
    config.timeframe = "DAILY"
    config.market_timing.enabled = False
    config.execution.stop_loss_pct = 0.0
    config.execution.trailing_stop_pct = 0.0
    config.execution.time_stop_months = 0
    config.execution.initial_cash = float(initial_cash)
    config.scoring.sentiment_weight = 0.0
    config.sentiment_data.allow_price_proxy = False
    config.portfolio.rebalance_frequency = "monthly"
    config.portfolio.top_n = 10
    config.portfolio.rank_buffer = 30
    config.portfolio.weighting = "equal_weight"
    config.sector_neutral.enabled = True
    config.sector_neutral.max_per_sector = 2
    return config


def candidate_metadata(initial_cash: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "candidate": name,
            "label": details["label"],
            "eligible_for_advancement": details[
                "eligible_for_advancement"
            ],
            "momentum_weight": details["momentum_weight"],
            "pullback_weight": details["pullback_weight"],
            "rebalance_frequency": "monthly",
            "top_n": 10,
            "rank_buffer": 30,
            "eligible_old_rank": 40,
            "max_per_sector": 2,
            "weighting": "equal_weight",
            "reference_initial_cash": float(initial_cash),
        }
        for name, details in CANDIDATES.items()
    ])


def _attach_shared_data(
    engine: BacktestEngine,
    shared: BacktestEngine,
) -> None:
    engine.price_data = shared.price_data
    engine.universe_store = shared.universe_store
    engine.code_to_name = shared.code_to_name
    engine.code_to_sector = shared.code_to_sector
    engine.name_to_code = shared.name_to_code


def _run_period(
    *,
    base_config: StrategyConfig,
    candidate: str,
    shared: BacktestEngine,
    index_data: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    config = copy.deepcopy(base_config)
    details = CANDIDATES[candidate]
    config.scoring.momentum_weight = float(details["momentum_weight"])
    config.scoring.pullback_weight = float(details["pullback_weight"])
    engine = LowVolatilityBacktestEngine(config, candidate)
    _attach_shared_data(engine, shared)
    result = engine.run(start, end, mode="price_only")
    performance = result["performance"]
    stats = result["stats"]
    execution = stats.get("execution", {})
    trades = result["trade_log"]
    first_execution = pd.Timestamp(trades[0]["date"]) if trades else pd.NaT
    benchmark = (
        _benchmark_return(index_data, first_execution, end)
        if pd.notna(first_execution)
        else np.nan
    )
    strategy_return = float(performance.get("total_return", 0.0))
    initial_cash = float(config.execution.initial_cash)
    total_fees = float(execution.get("total_fees", 0.0))
    slippage_cost = float(execution.get("slippage_cost", 0.0))
    turnover = float(performance.get("avg_turnover_pct", 0.0))
    order_count = sum(
        len(log.get("buys", [])) + len(log.get("sells", []))
        for log in trades
    )
    return {
        "total_return": strategy_return,
        "cagr": float(performance.get("cagr", 0.0)),
        "sharpe": float(performance.get("sharpe", 0.0)),
        "max_drawdown": float(
            performance.get("max_drawdown", 0.0)
        ),
        "benchmark_return": float(benchmark),
        "excess_return": strategy_return - float(benchmark),
        "avg_turnover_pct": turnover,
        "annualized_selection_turnover_pct": turnover * 12,
        "gross_execution_turnover_pct": (
            float(execution.get("gross_turnover", 0.0))
            / initial_cash
            * 100
        ),
        "total_fees": total_fees,
        "slippage_cost": slippage_cost,
        "friction_drag_pct_of_initial_cash": (
            (total_fees + slippage_cost) / initial_cash * 100
        ),
        "constrained_orders": int(
            execution.get("constrained_orders", 0)
        ),
        "executed_orders": int(order_count),
        "avg_deployment_pct": float(
            performance.get("avg_deployment_pct", 0.0)
        ),
    }


def _compound(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float((np.prod(1 + numeric / 100.0) - 1) * 100)


def summarize_rows(rows: pd.DataFrame) -> dict:
    """按运行前固定的组合晋级门槛汇总不重叠窗口。"""
    if rows.empty:
        return {}
    strategy_return = _compound(rows["test_total_return"])
    benchmark_return = _compound(rows["test_benchmark_return"])
    summary = {
        "folds": int(len(rows)),
        "compounded_oos_return": strategy_return,
        "compounded_benchmark_return": benchmark_return,
        "compounded_excess_return_pp": (
            strategy_return - benchmark_return
        ),
        "median_fold_return": float(
            rows["test_total_return"].median()
        ),
        "median_fold_excess_return": float(
            rows["test_excess_return"].median()
        ),
        "positive_return_fold_ratio": float(
            (rows["test_total_return"] > 0).mean()
        ),
        "positive_excess_fold_ratio": float(
            (rows["test_excess_return"] > 0).mean()
        ),
        "mean_fold_sharpe": float(rows["test_sharpe"].mean()),
        "worst_fold_drawdown": float(
            rows["test_max_drawdown"].min()
        ),
        "mean_annualized_selection_turnover_pct": float(
            rows["test_annualized_selection_turnover_pct"].mean()
        ),
        "total_fees": float(rows["test_total_fees"].sum()),
        "total_slippage_cost": float(
            rows["test_slippage_cost"].sum()
        ),
        "mean_friction_drag_pct_of_initial_cash": float(
            rows["test_friction_drag_pct_of_initial_cash"].mean()
        ),
        "mean_deployment_pct": float(
            rows["test_avg_deployment_pct"].mean()
        ),
    }
    summary["passed"] = bool(
        summary["compounded_oos_return"] > 0
        and summary["compounded_excess_return_pp"] > 0
        and summary["median_fold_return"] > 0
        and summary["median_fold_excess_return"] > 0
        and summary["positive_return_fold_ratio"] >= 0.60
        and summary["positive_excess_fold_ratio"] >= 0.60
        and summary["mean_fold_sharpe"] > 0
        and summary["worst_fold_drawdown"] > -30
        and summary[
            "mean_annualized_selection_turnover_pct"
        ] <= 1500
    )
    return summary


def _plot_fold_excess(
    fold_results: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "pullback_90_reference": "90d pullback reference",
        "low_volatility_60": "60d low volatility",
        "sector_neutral_low_volatility_60": "Sector-neutral 60d low vol",
        "balanced_low_volatility_60": "50/50 raw + sector-neutral",
    }
    pivot = fold_results.pivot(
        index="fold",
        columns="candidate",
        values="test_excess_return",
    )
    ordered = list(CANDIDATES)
    pivot = pivot[ordered]
    period_labels = (
        fold_results[["fold", "test_start", "test_end"]]
        .drop_duplicates("fold")
        .sort_values("fold")
    )
    pivot.index = [
        (
            f"{pd.Timestamp(row.test_start):%Y-%m}\n"
            f"to {pd.Timestamp(row.test_end):%Y-%m}"
        )
        for row in period_labels.itertuples()
    ]
    colors = ["#6B7280", "#2563EB", "#D97706", "#6B7D3A"]
    ax = pivot.rename(columns=labels).plot(
        kind="bar",
        figsize=(12, 6.8),
        width=0.82,
        color=colors,
        edgecolor="#374151",
        linewidth=0.5,
    )
    for container, hatch in zip(
        ax.containers,
        ["//", "..", "xx", "\\\\"],
    ):
        for patch in container:
            patch.set_hatch(hatch)
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_title("Excess return versus CSI 300 by research fold")
    ax.set_xlabel("Non-overlapping research folds")
    ax.set_ylabel("Excess return (percentage points)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title="Candidate portfolio",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def validate_result_tables(
    *,
    folds_frame: pd.DataFrame,
    summary_table: pd.DataFrame,
    expected_folds: int,
    prior_reference_path: Path | None = None,
) -> dict:
    """复算关键汇总并检查比较口径是否一致。"""
    critical = [
        "test_total_return",
        "test_benchmark_return",
        "test_excess_return",
        "test_sharpe",
        "test_max_drawdown",
        "test_annualized_selection_turnover_pct",
    ]
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })

    expected_rows = len(CANDIDATES) * expected_folds
    record(
        "expected_candidate_fold_rows",
        len(folds_frame) == expected_rows,
        f"actual={len(folds_frame)}, expected={expected_rows}",
    )
    duplicates = int(
        folds_frame.duplicated(["fold", "candidate"]).sum()
    )
    record(
        "unique_candidate_fold_grain",
        duplicates == 0,
        f"duplicate_rows={duplicates}",
    )
    missing = int(folds_frame[critical].isna().sum().sum())
    record(
        "critical_metrics_complete",
        missing == 0,
        f"missing_cells={missing}",
    )
    benchmark_counts = folds_frame.groupby("fold")[
        "test_benchmark_return"
    ].nunique(dropna=False)
    record(
        "same_benchmark_within_each_fold",
        bool((benchmark_counts == 1).all()),
        f"max_distinct_benchmarks={int(benchmark_counts.max())}",
    )

    max_recompute_difference = 0.0
    for candidate in CANDIDATES:
        recomputed = summarize_rows(
            folds_frame[folds_frame["candidate"] == candidate]
        )
        saved = summary_table[
            summary_table["strategy"] == candidate
        ].iloc[0]
        for metric in (
            "compounded_oos_return",
            "compounded_benchmark_return",
            "compounded_excess_return_pp",
            "median_fold_excess_return",
            "mean_fold_sharpe",
            "worst_fold_drawdown",
        ):
            difference = abs(
                float(recomputed[metric]) - float(saved[metric])
            )
            max_recompute_difference = max(
                max_recompute_difference,
                difference,
            )
    record(
        "headline_metrics_recomputed",
        max_recompute_difference < 1e-9,
        f"max_absolute_difference={max_recompute_difference:.3g}",
    )

    if (
        prior_reference_path is not None
        and prior_reference_path.exists()
    ):
        prior = pd.read_csv(prior_reference_path)
        prior_row = prior[
            prior["strategy"] == "pullback90_monthly_top10_b30"
        ]
        current_row = summary_table[
            summary_table["strategy"] == "pullback_90_reference"
        ]
        if not prior_row.empty and not current_row.empty:
            difference = abs(
                float(prior_row.iloc[0]["compounded_oos_return"])
                - float(current_row.iloc[0]["compounded_oos_return"])
            )
            record(
                "reference_reconciles_to_prior_validation",
                difference < 1e-9,
                f"compounded_return_difference={difference:.3g}",
            )

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
    }


def run_validation(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    train_months: int = 24,
    test_months: int = 6,
    initial_cash: float = 100_000,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    folds: list[Fold] = generate_folds(
        start,
        end,
        train_months,
        test_months,
    )
    if not folds:
        raise ValueError("区间不足以形成走步测试窗口")

    base_config = _base_config(initial_cash)
    metadata = candidate_metadata(initial_cash)
    logger.info("一次性加载价格、历史股票池与沪深300")
    shared = BacktestEngine(base_config)
    shared.load_price_data(start, end)
    if shared.universe_store is None:
        raise ValueError("低波动组合验证要求历史时点股票池")
    index_data = load_index_data("000300", start, end)
    if index_data.empty:
        raise ValueError("沪深300数据为空")

    cache: dict[tuple[str, str, str], dict] = {}

    def run_cached(candidate: str, period_start: str, period_end: str) -> dict:
        key = (candidate, period_start, period_end)
        if key not in cache:
            cache[key] = _run_period(
                base_config=base_config,
                candidate=candidate,
                shared=shared,
                index_data=index_data,
                start=period_start,
                end=period_end,
            )
        return cache[key]

    fold_rows: list[dict] = []
    for fold in folds:
        logger.info(
            f"Fold {fold.fold}: {fold.test_start}~{fold.test_end}"
        )
        for candidate in CANDIDATES:
            test = run_cached(
                candidate,
                fold.test_start,
                fold.test_end,
            )
            fold_rows.append({
                "fold": fold.fold,
                "candidate": candidate,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **{f"test_{key}": value for key, value in test.items()},
            })

    folds_frame = pd.DataFrame(fold_rows)
    fixed_summaries = {
        candidate: summarize_rows(
            folds_frame[folds_frame["candidate"] == candidate]
        )
        for candidate in CANDIDATES
    }
    summary_table = pd.DataFrame([
        {
            "strategy": candidate,
            "label": CANDIDATES[candidate]["label"],
            "eligible_for_advancement": CANDIDATES[candidate][
                "eligible_for_advancement"
            ],
            **metrics,
        }
        for candidate, metrics in fixed_summaries.items()
    ])

    continuous_start = folds[0].test_start
    continuous_rows = pd.DataFrame([
        {
            "candidate": candidate,
            "label": CANDIDATES[candidate]["label"],
            "start": continuous_start,
            "end": end,
            **run_cached(candidate, continuous_start, end),
        }
        for candidate in CANDIDATES
    ])
    validation = validate_result_tables(
        folds_frame=folds_frame,
        summary_table=summary_table,
        expected_folds=len(folds),
        prior_reference_path=(
            PROJECT_ROOT
            / "data"
            / "backtest_results"
            / "low_turnover_validation"
            / "strategy_summary.csv"
            if (
                start == "2020-01-01"
                and end == "2025-05-12"
                and train_months == 24
                and test_months == 6
            )
            else None
        ),
    )
    if validation["status"] != "passed":
        failed_checks = [
            check["check"]
            for check in validation["checks"]
            if not check["passed"]
        ]
        raise ValueError(f"验证结果复核失败: {failed_checks}")

    eligible = summary_table[
        summary_table["eligible_for_advancement"]
    ].copy()
    passing = eligible[eligible["passed"]].copy()
    if passing.empty:
        selected_candidate = None
        status = "failed"
        decision = "没有低波动候选满足组合级晋级门槛。"
    else:
        passing = passing.sort_values(
            [
                "compounded_excess_return_pp",
                "compounded_oos_return",
                "mean_fold_sharpe",
            ],
            ascending=False,
        )
        selected_candidate = str(passing.iloc[0]["strategy"])
        status = "passed_research_gate"
        decision = (
            f"{CANDIDATES[selected_candidate]['label']}满足门槛，"
            "进入参数敏感性与稳健性测试。"
        )

    rule = (
        "复合收益>0、复合超额收益>0、中位窗口收益和超额收益>0、"
        "正收益及正超额窗口均>=60%、平均Sharpe>0、最差回撤>-30%、"
        "年化选择换手<=1500%"
    )
    result = {
        "period": {
            "history_start": start,
            "research_fold_start": continuous_start,
            "end": end,
            "fold_count": len(folds),
            "fold_months": test_months,
        },
        "reference_initial_cash": float(initial_cash),
        "shared_execution": {
            "rebalance_frequency": "monthly",
            "top_n": 10,
            "rank_buffer": 30,
            "eligible_old_rank": 40,
            "max_per_sector": 2,
            "weighting": "equal_weight",
            "entry": "signal后的下一交易日开盘",
            "frictions": "真实佣金、最低佣金、过户费、印花税、滑点与成交限制",
        },
        "advancement_rule": rule,
        "fixed_candidate_summaries": fixed_summaries,
        "validation": validation,
        "selected_candidate": selected_candidate,
        "verdict": {
            "status": status,
            "decision": decision,
            "validation_assessment": "Share with caveats",
            "research_boundary": (
                "候选因子来自同一历史区间的前序筛选；本结果可用于研究取舍，"
                "不能替代未来3–6个月未见数据验证。"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(
        output_dir / "candidate_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds_frame.to_csv(
        output_dir / "fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_table.to_csv(
        output_dir / "strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    continuous_rows.to_csv(
        output_dir / "continuous_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chart_path = output_dir / "fold_excess_returns.png"
    _plot_fold_excess(folds_frame, chart_path)

    visible_columns = [
        "label",
        "compounded_oos_return",
        "compounded_benchmark_return",
        "compounded_excess_return_pp",
        "median_fold_excess_return",
        "positive_return_fold_ratio",
        "positive_excess_fold_ratio",
        "mean_fold_sharpe",
        "worst_fold_drawdown",
        "mean_annualized_selection_turnover_pct",
        "passed",
    ]
    raw_summary = fixed_summaries["low_volatility_60"]
    report = [
        "# 低波动候选组合级验证",
        "",
        "## Executive Summary",
        "",
        f"- **组合级判定：{status}。** {decision}",
        (
            "- **原始低波动有总量优势但缺少稳定性。** 七窗口复合收益"
            f"{raw_summary['compounded_oos_return']:+.1f}%，相对基准"
            f"{raw_summary['compounded_excess_return_pp']:+.1f}个百分点；"
            f"但仅{int(raw_summary['positive_return_fold_ratio'] * 7)}/7"
            "窗口赚钱、"
            f"{int(raw_summary['positive_excess_fold_ratio'] * 7)}/7"
            "窗口跑赢基准，不能由少数强窗口替代稳定性门槛。"
        ),
        (
            "- **统一比较口径。** 所有候选均使用月频Top 10、"
            "排名缓冲30（旧持仓最高保留至Top 40）、每行业最多2只、"
            "等权、下一交易日开盘和真实摩擦。"
        ),
        (
            "- **结论需带保留条件。** 因子先在同一历史区间筛选，"
            "所以本报告是研究验证，不是全新样本外证明。"
        ),
        (
            "- **结果表复核通过。** 候选×窗口粒度、关键字段完整性、"
            "同窗基准一致性、头部指标独立复算及旧参考策略对账均通过。"
        ),
        "",
        "## 七个窗口决定候选能否晋级",
        "",
        (
            "下图展示每个不重叠6个月窗口相对沪深300的超额收益。"
            "稳定候选应在多数窗口位于零线上方，而不是依赖单一时期。"
        ),
        "",
        "![各窗口超额收益](fold_excess_returns.png)",
        "",
        summary_table[visible_columns].to_markdown(index=False),
        "",
        "## 连续区间结果用于核对经济意义",
        "",
        (
            "连续区间结果保留持仓缓冲的完整路径，用于核对分段重启"
            "是否改变方向；晋级仍以预先声明的七窗口规则为准。"
        ),
        "",
        continuous_rows.to_markdown(index=False),
        "",
        "## 推荐的下一步",
        "",
        (
            "1. 对通过者测试40/60/90日窗口、Top 8/10/15、"
            "缓冲20/30/40及等权/风险权重。"
            if selected_candidate
            else (
                "1. 不把低波动作为独立策略晋级；转向估值与质量"
                "因子筛选，后续只把低波动作为风险特征或组合成分复验。"
            )
        ),
        "2. 对费用和滑点翻倍、去掉最佳年份、行业剔除做压力测试。",
        "3. 只有稳健性通过后，冻结策略并开始未来3–6个月模拟盘。",
        "",
        "## 仍需回答的问题",
        "",
        "- 不同资金规模下整手约束和最低佣金是否改变持仓可执行性？",
        "- 低波动收益是否集中在特定行业或市场状态？",
        "- 加入估值或质量因子后是否能提供独立增量？",
        "",
        "## Caveats and Assumptions",
        "",
        f"- 参考资金为{initial_cash:,.0f}元；股票排名不依赖资金，"
        "但费用、整手和资金利用率依赖资金规模。",
        f"- 晋级门槛：{rule}",
        f"- 验证评估：Share with caveats。{result['verdict']['research_boundary']}",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    logger.info(f"低波动组合验证完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="低波动候选组合级验证")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_validation(
        start=args.start,
        end=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        initial_cash=args.initial_cash,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
