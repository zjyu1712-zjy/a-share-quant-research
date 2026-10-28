"""行业中性 1/PB 及其与低波动组合的组合级研究验证。

所有候选共用历史时点沪深300、月频 Top 10、排名缓冲30、每行业最多2只、
等权、下一交易日开盘及真实交易摩擦。因子已在同一历史区间完成前置筛选，
因此本模块属于组合级研究复验，不应描述为全新未见样本。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.config import StrategyConfig
from src.fundamental_factor_screening import (
    DEFAULT_CACHE,
    build_fundamental_factors,
    load_valuation_data,
)
from src.low_volatility_portfolio_validation import (
    _attach_shared_data,
    _base_config,
    build_low_volatility_factor_frame,
    summarize_rows,
)
from src.price_feed import load_index_data
from src.walk_forward_attribution import _benchmark_return
from src.walk_forward_validation import Fold, generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "fundamental_portfolio_validation"
)

# 候选及权重在运行前固定。混合时统一回测器会先按调仓日做横截面 z-score。
CANDIDATES = {
    "low_volatility_60_reference": {
        "label": "60日低波动参考",
        "book_weight": 0.0,
        "low_vol_weight": 1.0,
        "eligible_for_advancement": False,
    },
    "book_to_price_mrq": {
        "label": "行业中性1/PB",
        "book_weight": 1.0,
        "low_vol_weight": 0.0,
        "eligible_for_advancement": True,
    },
    "book25_lowvol75": {
        "label": "1/PB 25% + 低波动75%",
        "book_weight": 0.25,
        "low_vol_weight": 0.75,
        "eligible_for_advancement": True,
    },
    "book50_lowvol50": {
        "label": "1/PB 50% + 低波动50%",
        "book_weight": 0.50,
        "low_vol_weight": 0.50,
        "eligible_for_advancement": True,
    },
    "book75_lowvol25": {
        "label": "1/PB 75% + 低波动25%",
        "book_weight": 0.75,
        "low_vol_weight": 0.25,
        "eligible_for_advancement": True,
    },
}


def _close_frame(
    price_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    records: list[dict] = []
    signal_dates = set(pd.DatetimeIndex(rebalance_dates))
    for code, frame in price_data.items():
        if frame.empty or "close" not in frame.columns:
            continue
        current = frame.loc[frame.index.intersection(signal_dates), ["close"]]
        for date, row in current.iterrows():
            records.append({
                "date": pd.Timestamp(date),
                "code": str(code).zfill(6),
                "close": float(row["close"]),
            })
    return pd.DataFrame(records)


def build_candidate_factor_frames(
    *,
    price_data: dict[str, pd.DataFrame],
    valuation_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> dict[str, pd.DataFrame]:
    """一次性构造所有候选，避免不同候选使用不同数据时点。"""
    fundamentals = build_fundamental_factors(
        valuation_data=valuation_data,
        rebalance_dates=rebalance_dates,
        universe_store=universe_store,
    )
    closes = _close_frame(price_data, rebalance_dates)
    book = fundamentals[
        [
            "date",
            "code",
            "source_date",
            "value_book_to_price_mrq",
        ]
    ].merge(closes, on=["date", "code"], how="inner", validate="one_to_one")
    book = book.dropna(subset=["value_book_to_price_mrq"]).copy()

    low_volatility = build_low_volatility_factor_frame(
        price_data=price_data,
        rebalance_dates=rebalance_dates,
        universe_store=universe_store,
        candidate="low_volatility_60",
    )
    low_volatility = low_volatility[
        ["date", "code", "close", "momentum"]
    ].rename(columns={"momentum": "low_volatility_60"})

    combined = book.merge(
        low_volatility[["date", "code", "low_volatility_60"]],
        on=["date", "code"],
        how="inner",
        validate="one_to_one",
    )
    frames: dict[str, pd.DataFrame] = {}
    reference = low_volatility.copy()
    reference["momentum"] = reference["low_volatility_60"]
    reference["pullback"] = 0.0
    frames["low_volatility_60_reference"] = reference

    book_only = book.copy()
    book_only["momentum"] = book_only["value_book_to_price_mrq"]
    book_only["pullback"] = 0.0
    frames["book_to_price_mrq"] = book_only

    for candidate in ("book25_lowvol75", "book50_lowvol50", "book75_lowvol25"):
        frame = combined.copy()
        frame["momentum"] = frame["value_book_to_price_mrq"]
        frame["pullback"] = frame["low_volatility_60"]
        frames[candidate] = frame

    for candidate, frame in frames.items():
        logger.info(
            f"{candidate}: {len(frame)}条因子记录, "
            f"{frame['code'].nunique()}只股票, "
            f"{frame['date'].nunique()}个调仓日"
        )
    return frames


class FundamentalPortfolioBacktestEngine(BacktestEngine):
    """仅覆写候选因子；其余选股、成交、费用和绩效均复用统一回测器。"""

    def __init__(
        self,
        config: StrategyConfig,
        candidate: str,
        factor_frames: dict[str, pd.DataFrame],
    ):
        super().__init__(config)
        self.candidate = candidate
        self.factor_frames = factor_frames

    def build_price_factor_frame(
        self,
        rebalance_dates: pd.DatetimeIndex,
        adjusted_factors,
    ) -> pd.DataFrame:
        del adjusted_factors
        wanted = set(pd.DatetimeIndex(rebalance_dates))
        frame = self.factor_frames[self.candidate]
        return frame[frame["date"].isin(wanted)].copy()


def candidate_metadata(initial_cash: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "candidate": candidate,
            "label": details["label"],
            "eligible_for_advancement": details["eligible_for_advancement"],
            "book_weight": details["book_weight"],
            "low_volatility_weight": details["low_vol_weight"],
            "normalization": "cross_sectional_zscore_by_rebalance_date",
            "rebalance_frequency": "monthly",
            "top_n": 10,
            "rank_buffer": 30,
            "eligible_old_rank": 40,
            "max_per_sector": 2,
            "weighting": "equal_weight",
            "reference_initial_cash": float(initial_cash),
        }
        for candidate, details in CANDIDATES.items()
    ])


def _run_period(
    *,
    base_config: StrategyConfig,
    candidate: str,
    factor_frames: dict[str, pd.DataFrame],
    shared: BacktestEngine,
    index_data: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    config = copy.deepcopy(base_config)
    details = CANDIDATES[candidate]
    if candidate == "low_volatility_60_reference":
        config.scoring.momentum_weight = 1.0
        config.scoring.pullback_weight = 0.0
    else:
        config.scoring.momentum_weight = float(details["book_weight"])
        config.scoring.pullback_weight = float(details["low_vol_weight"])
    engine = FundamentalPortfolioBacktestEngine(
        config,
        candidate,
        factor_frames,
    )
    _attach_shared_data(engine, shared)
    result = engine.run(start, end, mode="price_only")
    performance = result["performance"]
    execution = result["stats"].get("execution", {})
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
        "max_drawdown": float(performance.get("max_drawdown", 0.0)),
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
        "constrained_orders": int(execution.get("constrained_orders", 0)),
        "executed_orders": int(order_count),
        "avg_deployment_pct": float(
            performance.get("avg_deployment_pct", 0.0)
        ),
    }


def profile_factor_frames(
    factor_frames: dict[str, pd.DataFrame],
    universe_store,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    duplicates = 0
    future_matches = 0
    for candidate, frame in factor_frames.items():
        duplicates += int(frame.duplicated(["date", "code"]).sum())
        if "source_date" in frame:
            future_matches += int(
                (
                    pd.to_datetime(frame["source_date"])
                    > pd.to_datetime(frame["date"])
                ).sum()
            )
        for date, group in frame.groupby("date"):
            universe_count = len(universe_store.snapshot_for(date).codes)
            available = int(group["code"].nunique())
            rows.append({
                "candidate": candidate,
                "date": str(pd.Timestamp(date).date()),
                "universe_count": universe_count,
                "available_count": available,
                "coverage_ratio": available / universe_count,
            })
    quality = pd.DataFrame(rows)
    summary = {
        "duplicate_date_code_rows": duplicates,
        "future_fundamental_matches": future_matches,
        "minimum_coverage_ratio": float(quality["coverage_ratio"].min()),
        "median_coverage_ratio": float(quality["coverage_ratio"].median()),
        "passed": bool(
            duplicates == 0
            and future_matches == 0
            and quality["coverage_ratio"].median() >= 0.60
        ),
    }
    return quality, summary


def validate_result_tables(
    *,
    folds_frame: pd.DataFrame,
    summary_table: pd.DataFrame,
    factor_quality: dict,
    expected_folds: int,
    prior_reference_path: Path | None = None,
) -> dict:
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
    duplicates = int(folds_frame.duplicated(["fold", "candidate"]).sum())
    record(
        "unique_candidate_fold_grain",
        duplicates == 0,
        f"duplicate_rows={duplicates}",
    )
    critical = [
        "test_total_return",
        "test_benchmark_return",
        "test_excess_return",
        "test_sharpe",
        "test_max_drawdown",
        "test_annualized_selection_turnover_pct",
    ]
    missing = int(folds_frame[critical].isna().sum().sum())
    record("critical_metrics_complete", missing == 0, f"missing_cells={missing}")
    benchmark_counts = folds_frame.groupby("fold")[
        "test_benchmark_return"
    ].nunique(dropna=False)
    record(
        "same_benchmark_within_each_fold",
        bool((benchmark_counts == 1).all()),
        f"max_distinct_benchmarks={int(benchmark_counts.max())}",
    )

    max_difference = 0.0
    for candidate in CANDIDATES:
        recalculated = summarize_rows(
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
            max_difference = max(
                max_difference,
                abs(float(recalculated[metric]) - float(saved[metric])),
            )
    record(
        "headline_metrics_recomputed",
        max_difference < 1e-9,
        f"max_absolute_difference={max_difference:.3g}",
    )
    record(
        "factor_data_point_in_time_and_covered",
        bool(factor_quality["passed"]),
        (
            f"duplicates={factor_quality['duplicate_date_code_rows']}, "
            f"future={factor_quality['future_fundamental_matches']}, "
            f"median_coverage={factor_quality['median_coverage_ratio']:.3f}"
        ),
    )
    if prior_reference_path is not None and prior_reference_path.exists():
        prior = pd.read_csv(prior_reference_path)
        prior_row = prior[prior["strategy"] == "low_volatility_60"]
        current_row = summary_table[
            summary_table["strategy"] == "low_volatility_60_reference"
        ]
        if not prior_row.empty and not current_row.empty:
            difference = abs(
                float(prior_row.iloc[0]["compounded_oos_return"])
                - float(current_row.iloc[0]["compounded_oos_return"])
            )
            record(
                "low_vol_reference_reconciles_to_prior_validation",
                difference < 1e-9,
                f"compounded_return_difference={difference:.3g}",
            )
    passed = all(check["passed"] for check in checks)
    return {"status": "passed" if passed else "failed", "checks": checks}


def _plot_fold_excess(folds_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = folds_frame.pivot(
        index="fold",
        columns="candidate",
        values="test_excess_return",
    )[list(CANDIDATES)]
    periods = (
        folds_frame[["fold", "test_start", "test_end"]]
        .drop_duplicates("fold")
        .sort_values("fold")
    )
    pivot.index = [
        f"{pd.Timestamp(row.test_start):%Y-%m}\nto\n{pd.Timestamp(row.test_end):%Y-%m}"
        for row in periods.itertuples()
    ]
    labels = {
        "low_volatility_60_reference": "60d low-vol reference",
        "book_to_price_mrq": "Sector-neutral 1/PB",
        "book25_lowvol75": "1/PB 25% + low-vol 75%",
        "book50_lowvol50": "1/PB 50% + low-vol 50%",
        "book75_lowvol25": "1/PB 75% + low-vol 25%",
    }
    ax = pivot.rename(columns=labels).plot(
        kind="bar",
        figsize=(13.5, 7.2),
        width=0.86,
        color=["#6B7280", "#B45309", "#2563EB", "#0F766E", "#7C3AED"],
        edgecolor="#374151",
        linewidth=0.4,
    )
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_title("Excess return versus CSI 300 by research fold")
    ax.set_xlabel("Non-overlapping six-month research folds")
    ax.set_ylabel("Excess return (percentage points)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title="Candidate portfolio",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
    )
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def run_validation(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    train_months: int = 24,
    test_months: int = 6,
    initial_cash: float = 100_000,
    valuation_cache: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    folds: list[Fold] = generate_folds(start, end, train_months, test_months)
    if not folds:
        raise ValueError("区间不足以形成走步测试窗口")

    base_config = _base_config(initial_cash)
    shared = BacktestEngine(base_config)
    logger.info("一次性加载价格、历史股票池、估值缓存与沪深300")
    shared.load_price_data(start, end)
    if shared.universe_store is None:
        raise ValueError("基本面组合验证要求历史时点股票池")
    valuation_data = load_valuation_data(
        list(shared.price_data),
        cache_dir=valuation_cache,
    )
    if not valuation_data:
        raise ValueError("历史估值缓存为空；请先运行 ./run_mac.sh fundamentals")
    index_data = load_index_data("000300", start, end)
    if index_data.empty:
        raise ValueError("沪深300数据为空")
    rebalance_dates = _generate_rebalance_dates(
        start,
        end,
        "monthly",
        shared.price_data,
    )
    factor_frames = build_candidate_factor_frames(
        price_data=shared.price_data,
        valuation_data=valuation_data,
        rebalance_dates=rebalance_dates,
        universe_store=shared.universe_store,
    )
    quality_by_date, factor_quality = profile_factor_frames(
        factor_frames,
        shared.universe_store,
    )
    if not factor_quality["passed"]:
        raise ValueError(f"候选因子数据质量未通过: {factor_quality}")

    cache: dict[tuple[str, str, str], dict] = {}

    def run_cached(candidate: str, period_start: str, period_end: str) -> dict:
        key = (candidate, period_start, period_end)
        if key not in cache:
            cache[key] = _run_period(
                base_config=base_config,
                candidate=candidate,
                factor_frames=factor_frames,
                shared=shared,
                index_data=index_data,
                start=period_start,
                end=period_end,
            )
        return cache[key]

    fold_rows: list[dict] = []
    for fold in folds:
        logger.info(f"Fold {fold.fold}: {fold.test_start}~{fold.test_end}")
        for candidate in CANDIDATES:
            metrics = run_cached(candidate, fold.test_start, fold.test_end)
            fold_rows.append({
                "fold": fold.fold,
                "candidate": candidate,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **{f"test_{key}": value for key, value in metrics.items()},
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
        factor_quality=factor_quality,
        expected_folds=len(folds),
        prior_reference_path=(
            PROJECT_ROOT
            / "data"
            / "backtest_results"
            / "low_volatility_portfolio_validation"
            / "strategy_summary.csv"
        ),
    )
    if validation["status"] != "passed":
        failures = [
            check["check"]
            for check in validation["checks"]
            if not check["passed"]
        ]
        raise ValueError(f"验证结果复核失败: {failures}")

    eligible = summary_table[summary_table["eligible_for_advancement"]]
    passing = eligible[eligible["passed"]].sort_values(
        [
            "compounded_excess_return_pp",
            "compounded_oos_return",
            "mean_fold_sharpe",
        ],
        ascending=False,
    )
    if passing.empty:
        selected_candidate = None
        status = "failed"
        decision = "没有1/PB或混合候选满足预先声明的组合级晋级门槛。"
    else:
        selected_candidate = str(passing.iloc[0]["strategy"])
        status = "passed_research_gate"
        decision = (
            f"{CANDIDATES[selected_candidate]['label']}满足组合级门槛，"
            "进入费用、时期和行业压力测试。"
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
        "factor_quality": factor_quality,
        "fixed_candidate_summaries": fixed_summaries,
        "validation": validation,
        "selected_candidate": selected_candidate,
        "verdict": {
            "status": status,
            "decision": decision,
            "validation_assessment": "Share with caveats",
            "research_boundary": (
                "1/PB先在同一历史区间完成前置筛选；本结果可用于研究取舍，"
                "不能替代未来3–6个月未见数据验证。"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_metadata(initial_cash).to_csv(
        output_dir / "candidate_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality_by_date.to_csv(
        output_dir / "factor_quality_by_date.csv",
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
    _plot_fold_excess(folds_frame, output_dir / "fold_excess_returns.png")

    visible = [
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
    report = [
        "# 1/PB与低波动组合级验证",
        "",
        "## 结论先行",
        "",
        f"- **组合级判定：{status}。** {decision}",
        (
            "- **比较口径一致。** 月频Top 10、缓冲30（旧持仓可保留至"
            "Top 40）、每行业最多2只、等权、下一交易日开盘和真实摩擦。"
        ),
        (
            "- **混合权重可解释。** 1/PB与低波动先逐日横截面标准化，"
            "再按25/75、50/50、75/25加权。"
        ),
        (
            "- **结果复核通过。** 候选×窗口粒度、指标完整性、同窗基准、"
            "头部指标复算、数据时点及旧低波动参考对账均通过。"
        ),
        f"- **研究边界。** {result['verdict']['research_boundary']}",
        "",
        "## 七个不重叠窗口",
        "",
        "![各窗口超额收益](fold_excess_returns.png)",
        "",
        summary_table[visible].to_markdown(index=False),
        "",
        "## 连续区间核对",
        "",
        continuous_rows.to_markdown(index=False),
        "",
        "## 下一步",
        "",
        (
            "1. 对晋级候选做费用和滑点翻倍、去掉最佳年份、行业剔除"
            "及不同资金规模压力测试。"
            if selected_candidate
            else (
                "1. 不把本批候选加入正式策略；回到因子研究，优先获取"
                "更可靠的质量/盈利数据或扩大因子来源。"
            )
        ),
        "2. 只有压力测试仍通过，才冻结策略并开始未来3–6个月模拟盘。",
        "",
        "## Caveats and Assumptions",
        "",
        f"- 参考资金为{initial_cash:,.0f}元；费用与整手约束依赖资金规模。",
        f"- 晋级门槛：{rule}",
        "- 本报告属于研究候选筛选，不构成买入建议或收益承诺。",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    logger.info(f"1/PB组合级验证完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1/PB与低波动组合级验证")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--valuation-cache", type=Path, default=DEFAULT_CACHE)
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
        valuation_cache=args.valuation_cache,
        output_dir=args.output,
    )
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
