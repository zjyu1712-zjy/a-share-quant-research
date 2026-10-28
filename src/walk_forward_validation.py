"""价格策略走步样本外验证。

使用扩展训练窗口和不重叠测试窗口。每个测试窗口开始前，只根据截至训练
期末的表现选择候选方案，再把选中的方案用于下一段未见数据。
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "walk_forward_2020_2025"
)


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def generate_folds(
    start: str,
    end: str,
    train_months: int = 24,
    test_months: int = 6,
) -> list[Fold]:
    if train_months < 6 or test_months < 1:
        raise ValueError("训练窗口至少6个月，测试窗口至少1个月")
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    train_end = start_date + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
    folds: list[Fold] = []
    fold_number = 1

    while train_end < end_date:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(
            test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1),
            end_date,
        )
        if test_end <= test_start:
            break
        folds.append(Fold(
            fold=fold_number,
            train_start=str(start_date.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
        ))
        fold_number += 1
        train_end = test_end
    return folds


def _candidate_builders() -> dict[str, Callable[[StrategyConfig], None]]:
    def baseline(config: StrategyConfig) -> None:
        config.sector_neutral.enabled = True
        config.sector_neutral.max_per_sector = 1
        config.portfolio.rank_buffer = 0
        config.portfolio.weighting = "equal_weight"

    def sector_max_2(config: StrategyConfig) -> None:
        baseline(config)
        config.sector_neutral.max_per_sector = 2

    def sector_off(config: StrategyConfig) -> None:
        baseline(config)
        config.sector_neutral.enabled = False

    def rank_buffer_5(config: StrategyConfig) -> None:
        baseline(config)
        config.portfolio.rank_buffer = 5

    def inverse_volatility(config: StrategyConfig) -> None:
        baseline(config)
        config.portfolio.weighting = "inverse_volatility"
        config.portfolio.risk_weight_window = 60
        config.portfolio.max_position_weight = 0.35

    def sector_off_buffer_5(config: StrategyConfig) -> None:
        baseline(config)
        config.sector_neutral.enabled = False
        config.portfolio.rank_buffer = 5

    return {
        "baseline": baseline,
        "sector_max_2": sector_max_2,
        "sector_off": sector_off,
        "rank_buffer_5": rank_buffer_5,
        "inverse_volatility": inverse_volatility,
        "sector_off_buffer_5": sector_off_buffer_5,
    }


def _run_period(
    *,
    base_config: StrategyConfig,
    configure: Callable[[StrategyConfig], None],
    price_data: dict,
    universe_store,
    code_to_name: dict[str, str],
    code_to_sector: dict[str, str],
    name_to_code: dict[str, str],
    start: str,
    end: str,
) -> dict:
    config = copy.deepcopy(base_config)
    configure(config)
    engine = BacktestEngine(config)
    engine.price_data = price_data
    engine.universe_store = universe_store
    engine.code_to_name = code_to_name
    engine.code_to_sector = code_to_sector
    engine.name_to_code = name_to_code
    result = engine.run(start, end, mode="price_only")
    performance = result["performance"]
    stats = result["stats"]
    return {
        "total_return": float(performance.get("total_return", 0.0)),
        "cagr": float(performance.get("cagr", 0.0)),
        "sharpe": float(performance.get("sharpe", 0.0)),
        "max_drawdown": float(performance.get("max_drawdown", 0.0)),
        "avg_turnover_pct": float(performance.get("avg_turnover_pct", 0.0)),
        "final_equity": float(stats.get("final_equity", 0.0)),
        "total_fees": float(
            stats.get("execution", {}).get("total_fees", 0.0)
        ),
    }


def summarize_test_rows(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {}
    returns = pd.to_numeric(rows["test_total_return"], errors="coerce").dropna()
    sharpes = pd.to_numeric(rows["test_sharpe"], errors="coerce").dropna()
    drawdowns = pd.to_numeric(rows["test_max_drawdown"], errors="coerce").dropna()
    compounded = (np.prod(1 + returns / 100) - 1) * 100
    summary = {
        "folds": int(len(returns)),
        "compounded_oos_return": float(compounded),
        "mean_fold_return": float(returns.mean()),
        "median_fold_return": float(returns.median()),
        "positive_fold_ratio": float((returns > 0).mean()),
        "mean_fold_sharpe": float(sharpes.mean()) if len(sharpes) else 0.0,
        "worst_fold_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "mean_turnover_pct": float(
            pd.to_numeric(rows["test_avg_turnover_pct"], errors="coerce").mean()
        ),
    }
    summary["passed"] = bool(
        summary["compounded_oos_return"] > 0
        and summary["median_fold_return"] > 0
        and summary["positive_fold_ratio"] >= 0.60
        and summary["mean_fold_sharpe"] > 0
        and summary["worst_fold_drawdown"] > -30
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
    folds = generate_folds(start, end, train_months, test_months)
    if not folds:
        raise ValueError("区间不足以形成走步测试窗口")

    base_config = StrategyConfig.from_env()
    base_config.timeframe = "WEEKLY"
    base_config.portfolio.rebalance_frequency = "weekly"
    base_config.market_timing.enabled = False
    base_config.execution.stop_loss_pct = 0.0
    base_config.execution.trailing_stop_pct = 0.0
    base_config.execution.time_stop_months = 0
    base_config.scoring.sentiment_weight = 0.0
    base_config.sentiment_data.allow_price_proxy = False

    logger.info("一次性加载全区间价格与历史股票池")
    shared_engine = BacktestEngine(base_config)
    shared_engine.load_price_data(start, end)
    price_data = shared_engine.price_data
    universe_store = shared_engine.universe_store
    if universe_store is None:
        raise ValueError("走步测试要求历史时点股票池")

    builders = _candidate_builders()
    cache: dict[tuple[str, str, str], dict] = {}

    def run_cached(candidate: str, period_start: str, period_end: str) -> dict:
        key = (candidate, period_start, period_end)
        if key not in cache:
            cache[key] = _run_period(
                base_config=base_config,
                configure=builders[candidate],
                price_data=price_data,
                universe_store=universe_store,
                code_to_name=shared_engine.code_to_name,
                code_to_sector=shared_engine.code_to_sector,
                name_to_code=shared_engine.name_to_code,
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
            test = run_cached(candidate, fold.test_start, fold.test_end)
            train = train_results[candidate]
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

        selected_test = run_cached(selected, fold.test_start, fold.test_end)
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
                for key, value in selected_test.items()
            },
        })

    details = pd.DataFrame(detailed_rows)
    adaptive = pd.DataFrame(adaptive_rows)
    fixed_summaries = {
        candidate: summarize_test_rows(
            details.loc[details["candidate"] == candidate]
        )
        for candidate in builders
    }
    adaptive_summary = summarize_test_rows(adaptive)
    verdict = {
        "status": (
            "passed" if adaptive_summary.get("passed") else "failed"
        ),
        "rule": (
            "复合样本外收益>0、中位窗口收益>0、正收益窗口占比>=60%、"
            "平均窗口Sharpe>0、最差窗口回撤优于-30%"
        ),
    }
    verdict["reason"] = (
        "训练期选参后的样本外组合满足全部预设门槛。"
        if verdict["status"] == "passed"
        else "训练期选参后的样本外组合未满足全部预设门槛。"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
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
    summary = {
        "start": start,
        "end": end,
        "train_months": train_months,
        "test_months": test_months,
        "fold_count": len(folds),
        "fixed_candidate_summaries": fixed_summaries,
        "adaptive_summary": adaptive_summary,
        "adaptive_selected_candidates": adaptive[
            ["fold", "selected_candidate", "test_start", "test_end"]
        ].to_dict(orient="records"),
        "verdict": verdict,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_table = pd.DataFrame([
        {"strategy": candidate, **metrics}
        for candidate, metrics in fixed_summaries.items()
    ] + [{"strategy": "adaptive_train_selected", **adaptive_summary}])
    summary_table.to_csv(
        output_dir / "strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = [
        "# 走步样本外验证报告",
        "",
        f"- 历史区间：{start} 至 {end}",
        f"- 首个训练窗：{train_months}个月",
        f"- 后续测试窗：{test_months}个月",
        f"- 样本外窗口：{len(folds)}个",
        "- 新闻和价格代理：未使用",
        "- 成交：下一交易日开盘，包含真实费用与限制",
        "",
        "## 固定策略与训练期选参结果",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## 每期训练后选择",
        "",
        adaptive[[
            "fold", "selected_candidate", "train_sharpe",
            "test_total_return", "test_sharpe", "test_max_drawdown",
        ]].to_markdown(index=False),
        "",
        "## 判定",
        "",
        f"**{verdict['status']}**",
        "",
        verdict["reason"],
        "",
        f"判定规则：{verdict['rule']}",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    logger.info(f"走步验证完成: {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="价格策略走步样本外验证")
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
