"""低换手价格策略的探索性走步验证。

候选方案在运行前固定，围绕收益归因暴露出的两个问题展开：
1. 去除样本外负 IC 的短期动量；
2. 用 Top 10、排名缓冲和月度调仓降低轮换；
3. 对照当前被缩短为 8/18 日的因子窗口与原始 40/90 日窗口。

注意：候选设计使用了此前 2022-2025 归因结论，因此本报告属于研究迭代，
不是一份全新的、未被观察过的最终样本外证明。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Callable

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
    PROJECT_ROOT / "data" / "backtest_results" / "low_turnover_validation"
)


def _set_common(config: StrategyConfig) -> None:
    config.market_timing.enabled = False
    config.execution.stop_loss_pct = 0.0
    config.execution.trailing_stop_pct = 0.0
    config.execution.time_stop_months = 0
    config.scoring.sentiment_weight = 0.0
    config.sentiment_data.allow_price_proxy = False
    config.portfolio.weighting = "equal_weight"


def candidate_builders() -> dict[str, Callable[[StrategyConfig], None]]:
    """返回运行前固定的六个候选配置。"""

    def current_weekly_top5(config: StrategyConfig) -> None:
        _set_common(config)
        config.timeframe = "WEEKLY"
        config.portfolio.rebalance_frequency = "weekly"
        config.portfolio.top_n = 5
        config.portfolio.rank_buffer = 0
        config.sector_neutral.enabled = True
        config.sector_neutral.max_per_sector = 1
        config.scoring.momentum_weight = 1.0
        config.scoring.pullback_weight = 1.0

    def new_common(config: StrategyConfig) -> None:
        _set_common(config)
        config.portfolio.top_n = 10
        config.sector_neutral.enabled = True
        config.sector_neutral.max_per_sector = 2
        config.scoring.sentiment_weight = 0.0

    def pullback18_weekly_top10_b20(config: StrategyConfig) -> None:
        new_common(config)
        config.timeframe = "WEEKLY"
        config.portfolio.rebalance_frequency = "weekly"
        config.portfolio.rank_buffer = 20
        config.scoring.momentum_weight = 0.0
        config.scoring.pullback_weight = 1.0

    def pullback18_monthly_top10_b20(config: StrategyConfig) -> None:
        pullback18_weekly_top10_b20(config)
        config.portfolio.rebalance_frequency = "monthly"

    def pullback90_monthly_top10_b20(config: StrategyConfig) -> None:
        new_common(config)
        # 价格源始终为日频；DAILY 保留配置中的 40/90 日窗口，不做 /5。
        config.timeframe = "DAILY"
        config.portfolio.rebalance_frequency = "monthly"
        config.portfolio.rank_buffer = 20
        config.scoring.momentum_weight = 0.0
        config.scoring.pullback_weight = 1.0

    def price40_90_monthly_top10_b20(config: StrategyConfig) -> None:
        pullback90_monthly_top10_b20(config)
        config.scoring.momentum_weight = 1.0
        config.scoring.pullback_weight = 1.0

    def pullback90_monthly_top10_b30(config: StrategyConfig) -> None:
        pullback90_monthly_top10_b20(config)
        config.portfolio.rank_buffer = 30

    return {
        "current_weekly_top5": current_weekly_top5,
        "pullback18_weekly_top10_b20": pullback18_weekly_top10_b20,
        "pullback18_monthly_top10_b20": pullback18_monthly_top10_b20,
        "pullback90_monthly_top10_b20": pullback90_monthly_top10_b20,
        "price40_90_monthly_top10_b20": price40_90_monthly_top10_b20,
        "pullback90_monthly_top10_b30": pullback90_monthly_top10_b30,
    }


def candidate_metadata(
    base_config: StrategyConfig,
    builders: dict[str, Callable[[StrategyConfig], None]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for name, configure in builders.items():
        config = copy.deepcopy(base_config)
        configure(config)
        effective = config.factors.for_timeframe(config.timeframe)
        rows.append({
            "candidate": name,
            "timeframe_scaling": config.timeframe,
            "rebalance_frequency": config.portfolio.rebalance_frequency,
            "top_n": config.portfolio.top_n,
            "rank_buffer": config.portfolio.rank_buffer,
            "max_per_sector": config.sector_neutral.max_per_sector,
            "momentum_weight": config.scoring.momentum_weight,
            "pullback_weight": config.scoring.pullback_weight,
            "effective_momentum_days": effective.momentum_window,
            "effective_pullback_days": effective.pullback_window,
        })
    return pd.DataFrame(rows)


def _attach_shared_data(engine: BacktestEngine, shared: BacktestEngine) -> None:
    engine.price_data = shared.price_data
    engine.universe_store = shared.universe_store
    engine.code_to_name = shared.code_to_name
    engine.code_to_sector = shared.code_to_sector
    engine.name_to_code = shared.name_to_code


def _run_period(
    *,
    base_config: StrategyConfig,
    configure: Callable[[StrategyConfig], None],
    shared: BacktestEngine,
    index_data: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    config = copy.deepcopy(base_config)
    configure(config)
    engine = BacktestEngine(config)
    _attach_shared_data(engine, shared)
    result = engine.run(start, end, mode="price_only")
    performance = result["performance"]
    stats = result["stats"]
    trades = result["trade_log"]
    first_execution = pd.Timestamp(trades[0]["date"]) if trades else pd.NaT
    benchmark = (
        _benchmark_return(index_data, first_execution, end)
        if pd.notna(first_execution)
        else np.nan
    )
    periods_per_year = (
        52 if config.portfolio.rebalance_frequency == "weekly" else 12
    )
    turnover = float(performance.get("avg_turnover_pct", 0.0))
    return {
        "total_return": float(performance.get("total_return", 0.0)),
        "cagr": float(performance.get("cagr", 0.0)),
        "sharpe": float(performance.get("sharpe", 0.0)),
        "max_drawdown": float(performance.get("max_drawdown", 0.0)),
        "avg_turnover_pct": turnover,
        "annualized_selection_turnover_pct": turnover * periods_per_year,
        "total_fees": float(
            stats.get("execution", {}).get("total_fees", 0.0)
        ),
        "benchmark_return": float(benchmark),
        "excess_return": (
            float(performance.get("total_return", 0.0)) - float(benchmark)
        ),
    }


def _compound(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float((np.prod(1 + numeric / 100.0) - 1) * 100)


def summarize_rows(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {}
    strategy_return = _compound(rows["test_total_return"])
    benchmark_return = _compound(rows["test_benchmark_return"])
    summary = {
        "folds": int(len(rows)),
        "compounded_oos_return": strategy_return,
        "compounded_benchmark_return": benchmark_return,
        "compounded_excess_return_pp": strategy_return - benchmark_return,
        "median_fold_return": float(rows["test_total_return"].median()),
        "positive_fold_ratio": float(
            (rows["test_total_return"] > 0).mean()
        ),
        "mean_fold_sharpe": float(rows["test_sharpe"].mean()),
        "worst_fold_drawdown": float(rows["test_max_drawdown"].min()),
        "mean_annualized_selection_turnover_pct": float(
            rows["test_annualized_selection_turnover_pct"].mean()
        ),
        "mean_fold_fees": float(rows["test_total_fees"].mean()),
    }
    summary["passed"] = bool(
        summary["compounded_oos_return"] > 0
        and summary["compounded_excess_return_pp"] > 0
        and summary["median_fold_return"] > 0
        and summary["positive_fold_ratio"] >= 0.60
        and summary["mean_fold_sharpe"] > 0
        and summary["worst_fold_drawdown"] > -30
        and summary["mean_annualized_selection_turnover_pct"] <= 1500
    )
    return summary


def run_validation(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    train_months: int = 24,
    test_months: int = 6,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    folds: list[Fold] = generate_folds(
        start, end, train_months, test_months
    )
    if not folds:
        raise ValueError("区间不足以形成走步测试窗口")

    base_config = StrategyConfig.from_env()
    _set_common(base_config)
    builders = candidate_builders()
    metadata = candidate_metadata(base_config, builders)

    logger.info("一次性加载价格、历史股票池与沪深300")
    shared = BacktestEngine(base_config)
    shared.load_price_data(start, end)
    if shared.universe_store is None:
        raise ValueError("低换手验证要求历史时点股票池")
    index_data = load_index_data("000300", start, end)
    if index_data.empty:
        raise ValueError("沪深300数据为空")

    cache: dict[tuple[str, str, str], dict] = {}

    def run_cached(candidate: str, period_start: str, period_end: str) -> dict:
        key = (candidate, period_start, period_end)
        if key not in cache:
            cache[key] = _run_period(
                base_config=base_config,
                configure=builders[candidate],
                shared=shared,
                index_data=index_data,
                start=period_start,
                end=period_end,
            )
        return cache[key]

    detailed_rows: list[dict] = []
    adaptive_rows: list[dict] = []
    for fold in folds:
        logger.info(
            f"Fold {fold.fold}: train {fold.train_start}~{fold.train_end}, "
            f"test {fold.test_start}~{fold.test_end}"
        )
        train_results = {
            candidate: run_cached(
                candidate, fold.train_start, fold.train_end
            )
            for candidate in builders
        }
        selected = max(
            train_results,
            key=lambda name: (
                train_results[name]["sharpe"],
                train_results[name]["cagr"],
            ),
        )
        for candidate in builders:
            train = train_results[candidate]
            test = run_cached(candidate, fold.test_start, fold.test_end)
            detailed_rows.append({
                "fold": fold.fold,
                "candidate": candidate,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"test_{key}": value for key, value in test.items()},
                "selected_by_train": candidate == selected,
            })
        adaptive_rows.append({
            "fold": fold.fold,
            "selected_candidate": selected,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            **{
                f"train_{key}": value
                for key, value in train_results[selected].items()
            },
            **{
                f"test_{key}": value
                for key, value in run_cached(
                    selected, fold.test_start, fold.test_end
                ).items()
            },
        })

    details = pd.DataFrame(detailed_rows)
    adaptive = pd.DataFrame(adaptive_rows)
    fixed_summaries = {
        candidate: summarize_rows(
            details[details["candidate"] == candidate]
        )
        for candidate in builders
    }
    adaptive_summary = summarize_rows(adaptive)
    summary_table = pd.DataFrame([
        {"strategy": candidate, **metrics}
        for candidate, metrics in fixed_summaries.items()
    ] + [{"strategy": "adaptive_train_selected", **adaptive_summary}])
    verdict = {
        "status": (
            "passed" if adaptive_summary.get("passed") else "failed"
        ),
        "rule": (
            "复合收益和超额收益>0、中位窗口收益>0、正收益窗口>=60%、"
            "平均Sharpe>0、最差回撤>-30%、年化选择换手<=1500%"
        ),
        "research_status": (
            "本轮使用了此前归因结论设计候选，属于研究迭代；"
            "即使通过也需新的未见数据验证。"
        ),
    }
    summary = {
        "start": start,
        "end": end,
        "fold_count": len(folds),
        "candidate_metadata": metadata.to_dict(orient="records"),
        "fixed_candidate_summaries": fixed_summaries,
        "adaptive_summary": adaptive_summary,
        "adaptive_selected_candidates": adaptive[
            ["fold", "selected_candidate", "test_start", "test_end"]
        ].to_dict(orient="records"),
        "verdict": verdict,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(
        output_dir / "candidate_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    details.to_csv(
        output_dir / "all_candidate_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adaptive.to_csv(
        output_dir / "adaptive_selection_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_table.to_csv(
        output_dir / "strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# 低换手价格策略探索性走步验证",
        "",
        f"- 历史区间：{start} 至 {end}",
        f"- 样本外窗口：{len(folds)}个",
        "- 成交：下一交易日开盘，包含真实费用与限制",
        "- 新闻、价格代理、止损和大盘择时：关闭",
        "- 研究边界：候选设计参考了此前样本外归因，不是全新未见样本",
        "",
        "## 候选配置",
        "",
        metadata.to_markdown(index=False),
        "",
        "## 固定候选与训练期动态选择",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## 每期训练后选择",
        "",
        adaptive[[
            "fold", "selected_candidate", "train_sharpe",
            "test_total_return", "test_benchmark_return",
            "test_excess_return", "test_sharpe",
            "test_max_drawdown",
            "test_annualized_selection_turnover_pct",
        ]].to_markdown(index=False),
        "",
        "## 判定",
        "",
        f"**{verdict['status']}**",
        "",
        verdict["rule"],
        "",
        verdict["research_status"],
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    logger.info(f"低换手验证完成: {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="低换手价格策略走步验证")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_validation(
        start=args.start,
        end=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        output_dir=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
