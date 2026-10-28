"""独立价格因子筛选。

在历史时点沪深300股票池中检验：
- 60日低波动；
- 行业中性化后的60日低波动；
- 约6个月、跳过最近1个月的中期动量；
- 60日行业相对强度；
- 90日回撤参考因子。

先检查 Rank IC、Top/Bottom 分组收益差、窗口稳定性和因子相关性；本模块
不运行组合回测，也不修改正式策略。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.config import StrategyConfig
from src.walk_forward_attribution import HORIZONS, _open_forward_return
from src.walk_forward_validation import generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "factor_screening"
)
FACTOR_NAMES = [
    "low_volatility_60",
    "sector_neutral_low_volatility_60",
    "momentum_6_1",
    "sector_relative_strength_60",
    "pullback_90_reference",
]


def _hac_t_stat(values: pd.Series, max_lag: int) -> float:
    """Newey-West/HAC 均值 t 统计量，缓解多周收益重叠造成的高估。"""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    count = len(clean)
    if count < 3:
        return 0.0
    centered = clean - clean.mean()
    long_run_variance = float(np.dot(centered, centered) / count)
    lag_limit = min(max(int(max_lag), 0), count - 1)
    for lag in range(1, lag_limit + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
        weight = 1.0 - lag / (lag_limit + 1)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance / count, 0.0)
    if variance_of_mean <= 0:
        return 0.0
    return float(clean.mean() / np.sqrt(variance_of_mean))


def build_candidate_factors(
    *,
    price_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> pd.DataFrame:
    """只使用信号日及此前数据构建候选因子。"""
    records: list[dict] = []
    for code, frame in price_data.items():
        if frame.empty or "close" not in frame.columns:
            continue
        close = frame["close"].sort_index().astype(float)
        daily_return = close.pct_change()
        low_volatility = -daily_return.rolling(
            60, min_periods=60
        ).std() * np.sqrt(252)
        # 120日总回看、跳过最近20个交易日：P(t-20)/P(t-120)-1。
        momentum_6_1 = close.shift(20) / close.shift(120) - 1
        strength_60 = close.pct_change(60)
        rolling_high_90 = close.rolling(90, min_periods=90).max()
        pullback_90 = close / rolling_high_90 - 1

        for date in rebalance_dates:
            if date not in close.index:
                continue
            values = {
                "low_volatility_60": low_volatility.get(date, np.nan),
                "momentum_6_1": momentum_6_1.get(date, np.nan),
                "raw_strength_60": strength_60.get(date, np.nan),
                "pullback_90_reference": pullback_90.get(date, np.nan),
            }
            if all(pd.isna(value) for value in values.values()):
                continue
            records.append({
                "date": pd.Timestamp(date),
                "code": str(code).zfill(6),
                **values,
            })

    raw = pd.DataFrame(records)
    if raw.empty:
        return raw
    filtered: list[pd.DataFrame] = []
    for date, date_frame in raw.groupby("date"):
        snapshot = universe_store.snapshot_for(date)
        current = date_frame[
            date_frame["code"].isin(snapshot.codes)
        ].copy()
        current["sector"] = current["code"].map(snapshot.code_to_sector)
        current["sector_relative_strength_60"] = (
            current["raw_strength_60"]
            - current.groupby("sector")["raw_strength_60"].transform("median")
        )
        current["sector_neutral_low_volatility_60"] = (
            current["low_volatility_60"]
            - current.groupby("sector")["low_volatility_60"].transform("median")
        )
        filtered.append(current)
    result = pd.concat(filtered, ignore_index=True)
    logger.info(
        f"候选因子: {len(result)} 条, "
        f"{result['code'].nunique()}只股票, {result['date'].nunique()}个信号日"
    )
    return result


def add_forward_returns(
    factors: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = factors.copy()
    for weeks, periods in HORIZONS.items():
        result[f"forward_{weeks}w_pct"] = [
            _open_forward_return(
                price_data.get(str(code), pd.DataFrame()),
                pd.Timestamp(date),
                periods,
            )
            for code, date in zip(result["code"], result["date"])
        ]
    return result


def calculate_signal_diagnostics(
    enriched: pd.DataFrame,
    factor_names: list[str] = FACTOR_NAMES,
) -> pd.DataFrame:
    """逐信号日计算 Rank IC 与最高/最低20%收益差。"""
    records: list[dict] = []
    for date, date_frame in enriched.groupby("date"):
        for factor in factor_names:
            for weeks in HORIZONS:
                return_column = f"forward_{weeks}w_pct"
                pair = date_frame[[factor, return_column]].dropna()
                can_measure = (
                    len(pair) >= 50
                    and pair[factor].nunique() > 4
                    and pair[return_column].nunique() > 4
                )
                if not can_measure:
                    ic = np.nan
                    spread = np.nan
                    top_return = np.nan
                    bottom_return = np.nan
                else:
                    ic = pair[factor].corr(
                        pair[return_column], method="spearman"
                    )
                    percentile = pair[factor].rank(pct=True, method="average")
                    top_return = float(
                        pair.loc[percentile >= 0.80, return_column].mean()
                    )
                    bottom_return = float(
                        pair.loc[percentile <= 0.20, return_column].mean()
                    )
                    spread = top_return - bottom_return
                records.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "horizon_weeks": weeks,
                    "stock_count": int(len(pair)),
                    "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                    "top_quintile_return_pct": top_return,
                    "bottom_quintile_return_pct": bottom_return,
                    "top_bottom_spread_pct": spread,
                })
    return pd.DataFrame(records)


def assign_folds(
    diagnostics: pd.DataFrame,
    *,
    start: str,
    end: str,
    train_months: int = 24,
    test_months: int = 6,
) -> pd.DataFrame:
    result = diagnostics.copy()
    result["fold"] = pd.NA
    dates = pd.to_datetime(result["date"])
    for fold in generate_folds(start, end, train_months, test_months):
        mask = (
            (dates >= pd.Timestamp(fold.test_start))
            & (dates <= pd.Timestamp(fold.test_end))
        )
        result.loc[mask, "fold"] = fold.fold
    result["fold"] = result["fold"].astype("Int64")
    return result


def summarize_diagnostics(
    diagnostics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in diagnostics.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ic = group["rank_ic"].dropna()
        spread = group["top_bottom_spread_pct"].dropna()
        horizon = int(group["horizon_weeks"].iloc[0])
        row = dict(zip(group_columns, keys))
        row.update({
            "signal_dates": int(len(ic)),
            "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
            "median_rank_ic": float(ic.median()) if len(ic) else np.nan,
            "positive_ic_ratio": (
                float((ic > 0).mean()) if len(ic) else np.nan
            ),
            "hac_t_stat": _hac_t_stat(ic, max_lag=horizon - 1),
            "mean_top_bottom_spread_pct": (
                float(spread.mean()) if len(spread) else np.nan
            ),
            "positive_spread_ratio": (
                float((spread > 0).mean()) if len(spread) else np.nan
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_factor_correlations(
    factors: pd.DataFrame,
    factor_names: list[str] = FACTOR_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict] = []
    for date, date_frame in factors.groupby("date"):
        correlation = date_frame[factor_names].corr(method="spearman")
        for left_index, left in enumerate(factor_names):
            for right in factor_names[left_index + 1:]:
                value = correlation.loc[left, right]
                daily_rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor_left": left,
                    "factor_right": right,
                    "rank_correlation": (
                        float(value) if pd.notna(value) else np.nan
                    ),
                })
    daily = pd.DataFrame(daily_rows)
    summary = (
        daily.groupby(["factor_left", "factor_right"])["rank_correlation"]
        .agg(signal_dates="count", mean="mean", median="median")
        .reset_index()
    )
    summary["mean_absolute_correlation"] = (
        daily.assign(abs_corr=daily["rank_correlation"].abs())
        .groupby(["factor_left", "factor_right"])["abs_corr"]
        .mean()
        .to_numpy()
    )
    return daily, summary


def screen_factors(
    aggregate: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> pd.DataFrame:
    """按运行前固定的月频因子门槛筛选。"""
    rows: list[dict] = []
    for factor in FACTOR_NAMES:
        four_week = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 4)
        ]
        eight_week = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 8)
        ]
        fold_values = by_fold[
            (by_fold["factor"] == factor)
            & (by_fold["horizon_weeks"] == 4)
        ]["mean_rank_ic"].dropna()
        if four_week.empty or eight_week.empty:
            continue
        row4 = four_week.iloc[0]
        row8 = eight_week.iloc[0]
        positive_fold_ratio = (
            float((fold_values > 0).mean()) if len(fold_values) else 0.0
        )
        passed = bool(
            row4["mean_rank_ic"] >= 0.015
            and row4["positive_ic_ratio"] >= 0.55
            and row4["hac_t_stat"] >= 1.50
            and row4["mean_top_bottom_spread_pct"] > 0
            and positive_fold_ratio >= 5 / 7
            and row8["mean_rank_ic"] > 0
        )
        rows.append({
            "factor": factor,
            "four_week_mean_rank_ic": float(row4["mean_rank_ic"]),
            "four_week_positive_ic_ratio": float(
                row4["positive_ic_ratio"]
            ),
            "four_week_hac_t_stat": float(row4["hac_t_stat"]),
            "four_week_mean_spread_pct": float(
                row4["mean_top_bottom_spread_pct"]
            ),
            "positive_fold_ratio": positive_fold_ratio,
            "eight_week_mean_rank_ic": float(row8["mean_rank_ic"]),
            "status": "promising" if passed else "rejected",
        })
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict]:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(6)
    return result.replace({np.nan: None}).to_dict(orient="records")


def run_screening(
    *,
    history_start: str = "2020-01-01",
    oos_start: str = "2022-01-01",
    end: str = "2025-05-12",
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    config.portfolio.rebalance_frequency = "weekly"
    logger.info("加载行情与历史时点股票池")
    engine = BacktestEngine(config)
    engine.load_price_data(history_start, end)
    if engine.universe_store is None:
        raise ValueError("因子筛选要求历史时点股票池")
    rebalance_dates = _generate_rebalance_dates(
        oos_start, end, "weekly", engine.price_data
    )
    factors = build_candidate_factors(
        price_data=engine.price_data,
        rebalance_dates=rebalance_dates,
        universe_store=engine.universe_store,
    )
    enriched = add_forward_returns(factors, engine.price_data)
    diagnostics = calculate_signal_diagnostics(enriched)
    diagnostics = assign_folds(
        diagnostics,
        start=history_start,
        end=end,
    )
    aggregate = summarize_diagnostics(
        diagnostics,
        ["factor", "horizon_weeks"],
    )
    by_fold = summarize_diagnostics(
        diagnostics.dropna(subset=["fold"]),
        ["fold", "factor", "horizon_weeks"],
    )
    correlation_daily, correlation_summary = calculate_factor_correlations(
        factors
    )
    screening = screen_factors(aggregate, by_fold)
    promising = screening.loc[
        screening["status"] == "promising", "factor"
    ].tolist()
    result = {
        "period": {
            "history_start": history_start,
            "oos_start": oos_start,
            "end": end,
            "signal_dates": int(factors["date"].nunique()),
        },
        "screening_rule": (
            "4周平均IC>=0.015、IC为正周占比>=55%、HAC t>=1.5、"
            "Top-Bottom收益差>0、至少5/7窗口IC为正、8周平均IC>0"
        ),
        "screening": _records(screening),
        "promising_factors": promising,
        "decision": (
            "进入月频Top10+缓冲30组合回测"
            if promising
            else "没有因子通过门槛，停止进入组合回测"
        ),
        "research_note": (
            "筛选区间此前已用于策略归因，结果属于研究诊断；"
            "通过者仍需新的未见数据确认。"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    factors.to_csv(
        output_dir / "factor_values.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        output_dir / "diagnostics_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.to_csv(
        output_dir / "diagnostics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_fold.to_csv(
        output_dir / "diagnostics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlation_daily.to_csv(
        output_dir / "factor_correlations_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlation_summary.to_csv(
        output_dir / "factor_correlations_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    screening.to_csv(
        output_dir / "screening_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# 独立价格因子筛选报告",
        "",
        f"- 样本区间：{oos_start} 至 {end}",
        f"- 周度信号日：{factors['date'].nunique()}个",
        "- 收益口径：信号后下一交易日开盘至未来开盘",
        "- 显著性：使用HAC/Newey-West t统计量处理多周收益重叠",
        "- 本步骤只筛因子，不运行或修改正式组合",
        "",
        "## 筛选结论",
        "",
        screening.to_markdown(index=False),
        "",
        f"**{result['decision']}**",
        "",
        f"门槛：{result['screening_rule']}",
        "",
        "## 各期限总体诊断",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## 因子相关性",
        "",
        correlation_summary.to_markdown(index=False),
        "",
        "完整逐窗口结果见 `diagnostics_by_fold.csv`。",
        "",
        result["research_note"],
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    logger.info(f"因子筛选完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立价格因子筛选")
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screening(
        history_start=args.history_start,
        oos_start=args.oos_start,
        end=args.end,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
