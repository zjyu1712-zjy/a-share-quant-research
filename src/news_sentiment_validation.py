"""验证真实新闻情绪是否在价格因子之外提供增量收益。

该模块只用于研究，不降低正式策略的新闻覆盖门槛。验证时：
1. 仅使用 SQLite 中真实新闻分析结果，禁止价格代理；
2. 所有方案使用相同历史股票池、价格数据和成交规则；
3. 比较多个情绪权重，并计算情绪分数对下一周收益的 Rank IC；
4. 单独报告每个信号日的新闻覆盖，避免把稀疏样本误当作充分证据。
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig
from src.dual_factor import (
    build_combined_factors,
    build_price_factors,
    build_sentiment_factors,
    load_sentiment_series,
)
from src.execution_engine import generate_rebalance_dates
from src.price_feed import load_pool_data
from src.universe import PointInTimeUniverse
import src.price_feed as price_feed


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "financial_news.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "backtest_results" / "news_incremental"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "news_validation_price_cache"


def _database_range(db_path: Path) -> tuple[str, str]:
    if not db_path.exists():
        raise FileNotFoundError(f"新闻数据库不存在: {db_path}")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT MIN(fetch_date), MAX(fetch_date)
            FROM news_archive
            WHERE sentiment_score IS NOT NULL
              AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
              AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
            """
        ).fetchone()
    if not row or not row[0] or not row[1]:
        raise ValueError("新闻数据库没有可用的真实情绪记录")
    return str(row[0]), str(row[1])


def _load_all_real_sentiment(
    code_to_name: dict[str, str],
    start: str,
    end: str,
    db_path: Path,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    sentiment: dict[str, pd.Series] = {}
    rows: list[dict] = []
    for code, name in code_to_name.items():
        series = load_sentiment_series(name, start, end, db_path=db_path)
        count = int(len(series))
        rows.append({
            "code": code,
            "name": name,
            "news_days": count,
            "first_news_date": (
                str(series.index.min().date()) if count else None
            ),
            "last_news_date": (
                str(series.index.max().date()) if count else None
            ),
        })
        if count:
            sentiment[code] = series
    return sentiment, pd.DataFrame(rows)


def _filter_point_in_time(
    frame: pd.DataFrame,
    universe_store: PointInTimeUniverse,
) -> pd.DataFrame:
    filtered: list[pd.DataFrame] = []
    for date, date_frame in frame.groupby("date"):
        eligible = universe_store.snapshot_for(date).codes
        filtered.append(date_frame[date_frame["code"].isin(eligible)])
    return pd.concat(filtered, ignore_index=True) if filtered else frame.iloc[0:0]


def _forward_open_returns(
    price_data: dict[str, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """信号后下一交易日开盘买入，下个信号后的下一交易日开盘卖出。"""
    all_dates: set[pd.Timestamp] = set()
    for frame in price_data.values():
        all_dates.update(pd.DatetimeIndex(frame.index).normalize())
    calendar = pd.DatetimeIndex(sorted(all_dates))
    rows: list[dict] = []

    for index in range(len(signal_dates) - 1):
        signal_date = pd.Timestamp(signal_dates[index])
        next_signal = pd.Timestamp(signal_dates[index + 1])
        entries = calendar[calendar > signal_date]
        exits = calendar[calendar > next_signal]
        if not len(entries) or not len(exits):
            continue
        entry_date = pd.Timestamp(entries[0])
        exit_date = pd.Timestamp(exits[0])
        for code, frame in price_data.items():
            if entry_date not in frame.index or exit_date not in frame.index:
                continue
            entry = float(frame.loc[entry_date, "open"])
            exit_price = float(frame.loc[exit_date, "open"])
            if entry <= 0 or not np.isfinite(entry) or not np.isfinite(exit_price):
                continue
            rows.append({
                "date": signal_date,
                "code": code,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "forward_return": exit_price / entry - 1,
            })
    return pd.DataFrame(rows)


def _rank_ic_analysis(
    combined: pd.DataFrame,
    sentiment_factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    sent_keys = sentiment_factors[["date", "code"]].drop_duplicates()
    analysis = (
        combined.merge(sent_keys.assign(has_sentiment=True), on=["date", "code"])
        .merge(forward_returns, on=["date", "code"])
    )
    weekly_rows: list[dict] = []
    for date, group in analysis.groupby("date"):
        clean = group[
            ["score_sentiment", "score_price", "forward_return"]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 5:
            continue
        sentiment_ic = clean["score_sentiment"].corr(
            clean["forward_return"], method="spearman"
        )
        price_ic = clean["score_price"].corr(
            clean["forward_return"], method="spearman"
        )

        design = np.column_stack([
            np.ones(len(clean)),
            clean["score_price"].to_numpy(),
        ])
        residual = (
            clean["forward_return"].to_numpy()
            - design @ np.linalg.lstsq(
                design, clean["forward_return"].to_numpy(), rcond=None
            )[0]
        )
        residual_series = pd.Series(residual, index=clean.index)
        partial_ic = clean["score_sentiment"].corr(
            residual_series, method="spearman"
        )

        ranked = clean.sort_values("score_sentiment")
        bucket = max(len(ranked) // 3, 1)
        spread = (
            ranked.tail(bucket)["forward_return"].mean()
            - ranked.head(bucket)["forward_return"].mean()
        )
        weekly_rows.append({
            "date": date,
            "n_stocks": len(clean),
            "sentiment_rank_ic": sentiment_ic,
            "price_rank_ic": price_ic,
            "sentiment_partial_rank_ic": partial_ic,
            "top_bottom_spread": spread,
        })

    weekly = pd.DataFrame(weekly_rows)

    def summarize(column: str) -> dict:
        values = weekly[column].dropna() if column in weekly else pd.Series(dtype=float)
        if values.empty:
            return {
                "mean": None, "median": None, "positive_ratio": None,
                "t_stat": None, "observations": 0,
            }
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        t_stat = (
            float(values.mean() / (std / np.sqrt(len(values))))
            if len(values) > 1 and std > 0 else None
        )
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "positive_ratio": float((values > 0).mean()),
            "t_stat": t_stat,
            "observations": int(len(values)),
        }

    summary = {
        "sentiment_rank_ic": summarize("sentiment_rank_ic"),
        "price_rank_ic_on_covered_stocks": summarize("price_rank_ic"),
        "sentiment_partial_rank_ic": summarize("sentiment_partial_rank_ic"),
        "top_bottom_weekly_spread": summarize("top_bottom_spread"),
    }
    return weekly, summary


def _coverage_by_week(
    price_factors: pd.DataFrame,
    sentiment_factors: pd.DataFrame,
    universe_store: PointInTimeUniverse,
) -> pd.DataFrame:
    rows: list[dict] = []
    for date, price_group in price_factors.groupby("date"):
        pit_codes = universe_store.snapshot_for(date).codes
        available_price = set(price_group["code"]) & pit_codes
        sent_codes = set(
            sentiment_factors.loc[
                sentiment_factors["date"] == date, "code"
            ]
        ) & available_price
        rows.append({
            "date": date,
            "price_stock_count": len(available_price),
            "sentiment_stock_count": len(sent_codes),
            "sentiment_coverage": (
                len(sent_codes) / len(available_price)
                if available_price else 0.0
            ),
        })
    return pd.DataFrame(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def run_validation(
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    weights: list[float],
    db_path: Path = DEFAULT_DB,
    output_dir: Path = DEFAULT_OUTPUT,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict:
    db_start, db_end = _database_range(db_path)
    analysis_start = max(pd.Timestamp(start or db_start), pd.Timestamp(db_start))
    analysis_end = min(pd.Timestamp(end or db_end), pd.Timestamp(db_end))
    if analysis_start >= analysis_end:
        raise ValueError("新闻有效区间不足，无法验证")

    config = StrategyConfig.from_env()
    config.timeframe = "WEEKLY"
    config.portfolio.rebalance_frequency = "weekly"
    config.market_timing.enabled = False
    config.execution.stop_loss_pct = 0.0
    config.execution.trailing_stop_pct = 0.0
    config.execution.time_stop_months = 0
    config.sentiment_data.allow_price_proxy = False

    snapshot_dir = Path(config.universe.snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir
    universe_store = PointInTimeUniverse(snapshot_dir)
    members = universe_store.union_for_range(
        str(analysis_start.date()), str(analysis_end.date())
    )
    code_to_name = dict(zip(members["code"], members["name"]))
    code_to_sector = dict(zip(members["code"], members["sector"]))
    name_to_code = {name: code for code, name in code_to_name.items()}

    sentiment_data, stock_coverage = _load_all_real_sentiment(
        code_to_name,
        str(analysis_start.date()),
        str(analysis_end.date()),
        db_path,
    )
    if not sentiment_data:
        raise ValueError("没有股票匹配到真实新闻情绪")

    # 使用独立缓存，避免短区间验证覆盖长期正式回测缓存。
    cache_dir.mkdir(parents=True, exist_ok=True)
    price_feed.PRICE_CACHE_DIR = cache_dir
    warmup_start = analysis_start - pd.Timedelta(days=180)
    price_data = load_pool_data(
        list(code_to_name),
        str(warmup_start.date()),
        str(analysis_end.date()),
    )
    if len(price_data) < 20:
        raise ValueError(f"有效价格数据仅 {len(price_data)} 只，无法验证")

    signal_dates = generate_rebalance_dates(
        price_data,
        str(analysis_start.date()),
        str(analysis_end.date()),
        frequency="weekly",
    )
    factors = config.factors.for_timeframe(config.timeframe)
    price_factors = build_price_factors(price_data, signal_dates, factors)
    sentiment_factors = build_sentiment_factors(
        sentiment_data,
        signal_dates,
        price_data,
        factors,
        rebalance_freq="weekly",
    )
    price_factors = _filter_point_in_time(price_factors, universe_store)
    sentiment_factors = _filter_point_in_time(sentiment_factors, universe_store)
    coverage_weekly = _coverage_by_week(
        price_factors, sentiment_factors, universe_store
    )

    scoring_for_ic = copy.deepcopy(config.scoring)
    scoring_for_ic.sentiment_weight = 1.0
    combined_for_ic = build_combined_factors(
        price_factors, sentiment_factors, scoring_for_ic
    )
    forward_returns = _forward_open_returns(price_data, signal_dates)
    rank_ic_weekly, rank_ic_summary = _rank_ic_analysis(
        combined_for_ic, sentiment_factors, forward_returns
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict] = []
    baseline_targets: Optional[dict[str, set[str]]] = None
    tested_weights = [0.0] + [
        float(weight) for weight in weights if float(weight) > 0
    ]
    tested_weights = list(dict.fromkeys(tested_weights))

    for weight in tested_weights:
        run_config = copy.deepcopy(config)
        run_config.scoring.sentiment_weight = weight
        engine = BacktestEngine(run_config)
        engine.price_data = price_data
        engine.sentiment_data = sentiment_data
        engine.code_to_name = code_to_name
        engine.code_to_sector = code_to_sector
        engine.name_to_code = name_to_code
        engine.universe_store = universe_store
        mode = "price_only" if weight == 0 else "hybrid"
        result = engine.run(
            str(analysis_start.date()),
            str(analysis_end.date()),
            mode=mode,
        )
        label = "price_only" if weight == 0 else f"hybrid_w{weight:g}"
        engine.export_results(output_dir / label)
        current_targets = {
            str(pd.Timestamp(item["signal_date"]).date()): set(item["target"])
            for item in result["trade_log"]
        }
        if weight == 0:
            baseline_targets = current_targets
        changed_weeks = 0
        substitutions = 0
        if baseline_targets is not None:
            all_signal_dates = set(baseline_targets) | set(current_targets)
            for signal_date in all_signal_dates:
                baseline_set = baseline_targets.get(signal_date, set())
                current_set = current_targets.get(signal_date, set())
                if baseline_set != current_set:
                    changed_weeks += 1
                    substitutions += max(
                        len(current_set - baseline_set),
                        len(baseline_set - current_set),
                    )
        performance = result["performance"]
        stats = result["stats"]
        comparisons.append({
            "label": label,
            "sentiment_weight": weight,
            "total_return": performance.get("total_return"),
            "cagr": performance.get("cagr"),
            "sharpe": performance.get("sharpe"),
            "max_drawdown": performance.get("max_drawdown"),
            "avg_turnover_pct": performance.get("avg_turnover_pct"),
            "final_equity": stats.get("final_equity"),
            "total_fees": stats.get("execution", {}).get("total_fees"),
            "changed_weeks_vs_price": changed_weeks,
            "substitutions_vs_price": substitutions,
        })

    comparison_df = pd.DataFrame(comparisons)
    baseline_return = float(
        comparison_df.loc[
            comparison_df["sentiment_weight"] == 0, "total_return"
        ].iloc[0]
    )
    comparison_df["incremental_return_pp"] = (
        comparison_df["total_return"] - baseline_return
    )
    comparisons = comparison_df.to_dict(orient="records")
    comparison_df.to_csv(
        output_dir / "weight_comparison.csv", index=False, encoding="utf-8-sig"
    )
    stock_coverage.to_csv(
        output_dir / "stock_news_coverage.csv", index=False, encoding="utf-8-sig"
    )
    coverage_weekly.to_csv(
        output_dir / "weekly_factor_coverage.csv", index=False, encoding="utf-8-sig"
    )
    rank_ic_weekly.to_csv(
        output_dir / "weekly_rank_ic.csv", index=False, encoding="utf-8-sig"
    )

    coverage_summary = {
        "universe_union_count": int(len(code_to_name)),
        "stocks_with_any_news": int((stock_coverage["news_days"] > 0).sum()),
        "stocks_with_5_news_days": int((stock_coverage["news_days"] >= 5).sum()),
        "weekly_mean_factor_coverage": (
            float(coverage_weekly["sentiment_coverage"].mean())
            if len(coverage_weekly) else 0.0
        ),
        "weekly_min_factor_coverage": (
            float(coverage_weekly["sentiment_coverage"].min())
            if len(coverage_weekly) else 0.0
        ),
        "weekly_max_factor_coverage": (
            float(coverage_weekly["sentiment_coverage"].max())
            if len(coverage_weekly) else 0.0
        ),
        "signal_weeks": int(len(signal_dates)),
        "rank_ic_weeks": int(len(rank_ic_weekly)),
    }
    summary = {
        "analysis_start": str(analysis_start.date()),
        "analysis_end": str(analysis_end.date()),
        "database_path": str(db_path),
        "price_proxy_used": False,
        "formal_quality_gate_changed": False,
        "coverage": coverage_summary,
        "rank_ic": rank_ic_summary,
        "portfolio_comparison": comparisons,
    }
    mean_ic = rank_ic_summary["sentiment_rank_ic"]["mean"]
    partial_ic = rank_ic_summary["sentiment_partial_rank_ic"]["mean"]
    spread = rank_ic_summary["top_bottom_weekly_spread"]["mean"]
    insufficient_sample = (
        coverage_summary["rank_ic_weeks"] < 26
        or coverage_summary["weekly_mean_factor_coverage"] < 0.50
    )
    predictive_support = (
        mean_ic is not None and mean_ic > 0.02
        and partial_ic is not None and partial_ic > 0
        and spread is not None and spread > 0
    )
    summary["verdict"] = {
        "status": (
            "not_validated"
            if insufficient_sample or not predictive_support
            else "validated"
        ),
        "insufficient_sample": insufficient_sample,
        "predictive_support": predictive_support,
        "reason": (
            "覆盖率和有效周数不足，且Rank IC、价格残差IC与多空收益差"
            "未形成一致的正向证据。"
            if insufficient_sample or not predictive_support
            else "覆盖率、样本量与三项预测统计均达到预设方向。"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = comparison_df.sort_values("total_return", ascending=False).iloc[0]
    baseline = comparison_df.loc[
        comparison_df["sentiment_weight"] == 0
    ].iloc[0]
    report = [
        "# 真实新闻情绪增量收益验证",
        "",
        f"- 区间：{analysis_start.date()} 至 {analysis_end.date()}",
        f"- 信号周数：{coverage_summary['signal_weeks']}",
        f"- Rank IC有效周数：{coverage_summary['rank_ic_weeks']}",
        f"- 股票池区间并集：{coverage_summary['universe_union_count']}只",
        f"- 至少匹配一次真实新闻：{coverage_summary['stocks_with_any_news']}只",
        (
            "- 周度情绪因子平均覆盖率："
            f"{coverage_summary['weekly_mean_factor_coverage']:.1%}"
        ),
        "- 价格代理：未使用",
        "- 正式覆盖门槛：未修改",
        "",
        "## 组合对照",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Rank IC",
        "",
        "```json",
        json.dumps(_json_ready(rank_ic_summary), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 初步判定",
        "",
        (
            f"全期收益最高的是 {best['label']}（{best['total_return']:.2f}%），"
            f"价格基准为 {baseline['total_return']:.2f}%。"
        ),
        (
            "样本周数较少或覆盖率偏低时，该结果只能视为探索性证据，"
            "不能据此启用实盘情绪权重。"
        ),
        "",
        f"正式判定：**{summary['verdict']['status']}**。",
        summary["verdict"]["reason"],
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    logger.info(f"真实新闻增量验证完成: {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实新闻情绪增量收益验证")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 1.0],
        help="待测试的正情绪权重；价格基准0会自动加入",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_validation(
        start=args.start,
        end=args.end,
        weights=args.weights,
        db_path=args.db,
        output_dir=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
