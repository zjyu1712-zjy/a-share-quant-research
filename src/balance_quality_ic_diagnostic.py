"""解释资产负债质量因子为何没有转化为组合收益。

本模块只做研究归因，不修改正式策略。诊断口径在结果产生前固定：

1. 三个质量候选按月分为五组，检查未来4周和8周收益单调性；
2. 检查最高分组的行业与规模暴露；
3. 只测试一个预先声明的替代用法：在同池低波动策略中剔除低应计
   最差20%的股票；
4. 过滤器必须相对同池低波动在至少5/7窗口胜出，且复合与窗口中位
   增量为正，换手不超过基准110%，回撤恶化不超过3个百分点。
"""

from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.balance_quality_factor_screening import (
    DEFAULT_CACHE as DEFAULT_STATEMENT_CACHE,
    DEFAULT_INCOME_CACHE,
    join_statement_sources,
    load_statement_supplements,
)
from src.balance_quality_portfolio_validation import (
    CANDIDATES,
    MATCHED_REFERENCE,
    _run_period,
    build_candidate_factor_frames,
)
from src.indicators import zscore
from src.low_volatility_portfolio_validation import (
    _base_config,
    summarize_rows,
)
from src.statement_factor_screening import (
    annual_report_dates,
    load_annual_reports,
)
from src.walk_forward_attribution import _open_forward_return
from src.walk_forward_validation import Fold, generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "balance_quality_ic_diagnostic"
)

QUALITY_CANDIDATES = [
    "low_accruals",
    "working_capital_discipline",
    "accruals50_working_capital50",
]
FILTER_NAME = "low_accruals_bottom20_filter"
FILTER_LABEL = "低波动＋剔除低应计最差20%"


def _score_frame(candidate: str, frame: pd.DataFrame) -> pd.DataFrame:
    """按与组合回测一致的截面 z-score 权重构造最终分数。"""
    details = CANDIDATES[candidate]
    result = frame.copy()
    result["z_factor_one"] = result.groupby("date")[
        "factor_one"
    ].transform(zscore)
    result["z_factor_two"] = result.groupby("date")[
        "factor_two"
    ].transform(zscore)
    result["quality_score"] = (
        result["z_factor_one"] * details["factor_one_weight"]
        + result["z_factor_two"] * details["factor_two_weight"]
    )
    return result


def estimate_market_cap(
    frame: pd.DataFrame,
    income: pd.DataFrame,
) -> pd.DataFrame:
    """用公告时点年报净利润/基本EPS估算股本，再乘信号日收盘价。

    这是规模暴露代理，不是交易所自由流通市值。异常或符号不一致的
    股本估算会被置空，避免伪精确。
    """
    source = income[[
        "code",
        "report_date",
        "eps",
        "net_profit",
    ]].copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["report_date"] = pd.to_datetime(source["report_date"])
    source = source.rename(columns={
        "report_date": "source_report_date",
        "net_profit": "income_net_profit",
    })
    result = frame.merge(
        source,
        on=["code", "source_report_date"],
        how="left",
        validate="many_to_one",
    )
    shares = result["income_net_profit"] / result["eps"]
    valid = (
        np.isfinite(shares)
        & (shares > 1_000_000)
        & (shares < 100_000_000_000)
        & (result["close"] > 0)
    )
    result["estimated_shares"] = shares.where(valid)
    result["estimated_market_cap"] = (
        result["close"] * result["estimated_shares"]
    )
    return result


def build_enriched_candidates(
    *,
    factor_frames: dict[str, pd.DataFrame],
    price_data: dict[str, pd.DataFrame],
    income: pd.DataFrame,
    folds: list[Fold],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for candidate in QUALITY_CANDIDATES:
        current = _score_frame(candidate, factor_frames[candidate])
        current = estimate_market_cap(current, income)
        current["candidate"] = candidate
        current["label"] = CANDIDATES[candidate]["label"]
        for weeks, periods in {4: 20, 8: 40}.items():
            current[f"forward_{weeks}w_pct"] = [
                _open_forward_return(
                    price_data.get(str(code), pd.DataFrame()),
                    pd.Timestamp(date),
                    periods,
                )
                for code, date in zip(current["code"], current["date"])
            ]
        current["quintile"] = current.groupby("date")[
            "quality_score"
        ].transform(assign_quintiles)
        current["size_percentile"] = current.groupby("date")[
            "estimated_market_cap"
        ].rank(pct=True, method="average")
        current["fold"] = pd.NA
        for fold in folds:
            mask = (
                (current["date"] >= pd.Timestamp(fold.test_start))
                & (current["date"] <= pd.Timestamp(fold.test_end))
            )
            current.loc[mask, "fold"] = fold.fold
        current["fold"] = current["fold"].astype("Int64")
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def assign_quintiles(values: pd.Series) -> pd.Series:
    """稳定地将高分方向分成1（最差）到5（最好）。"""
    valid = values.notna()
    output = pd.Series(pd.NA, index=values.index, dtype="Int64")
    if valid.sum() < 5:
        return output
    ranks = values.loc[valid].rank(method="first")
    output.loc[valid] = (
        pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        .astype(int)
        .astype("Int64")
    )
    return output


def calculate_quintile_returns(
    enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """先做日期内等权收益，再汇总，避免大样本月份获得更高权重。"""
    date_rows: list[dict] = []
    for candidate, candidate_frame in enriched.groupby("candidate"):
        for date, group in candidate_frame.groupby("date"):
            fold = group["fold"].dropna()
            fold_value = int(fold.iloc[0]) if len(fold) else pd.NA
            for horizon in [4, 8]:
                return_column = f"forward_{horizon}w_pct"
                for quintile, bucket in group.groupby(
                    "quintile", dropna=True
                ):
                    returns = pd.to_numeric(
                        bucket[return_column], errors="coerce"
                    ).dropna()
                    if returns.empty:
                        continue
                    date_rows.append({
                        "candidate": candidate,
                        "date": str(pd.Timestamp(date).date()),
                        "fold": fold_value,
                        "horizon_weeks": horizon,
                        "quintile": int(quintile),
                        "stock_count": int(len(returns)),
                        "equal_weight_return_pct": float(returns.mean()),
                    })
    by_date = pd.DataFrame(date_rows)
    summary = (
        by_date.groupby(
            ["candidate", "horizon_weeks", "quintile"],
            as_index=False,
        )
        .agg(
            mean_return_pct=("equal_weight_return_pct", "mean"),
            median_return_pct=("equal_weight_return_pct", "median"),
            positive_month_ratio=(
                "equal_weight_return_pct",
                lambda values: float((values > 0).mean()),
            ),
            month_count=("date", "nunique"),
        )
    )
    diagnostics = summarize_quintile_shape(by_date, summary)
    return by_date, summary, diagnostics


def summarize_quintile_shape(
    by_date: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in QUALITY_CANDIDATES:
        for horizon in [4, 8]:
            current = summary[
                (summary["candidate"] == candidate)
                & (summary["horizon_weeks"] == horizon)
            ].sort_values("quintile")
            returns = current.set_index("quintile")["mean_return_pct"]
            complete = all(value in returns.index for value in range(1, 6))
            monotonic_correlation = (
                float(pd.Series(range(1, 6)).corr(
                    pd.Series(
                        [returns.loc[value] for value in range(1, 6)]
                    ),
                    method="spearman",
                ))
                if complete else np.nan
            )
            fold_spreads = []
            candidate_dates = by_date[
                (by_date["candidate"] == candidate)
                & (by_date["horizon_weeks"] == horizon)
                & by_date["fold"].notna()
            ]
            for _, fold_group in candidate_dates.groupby("fold"):
                pivot = fold_group.pivot_table(
                    index="date",
                    columns="quintile",
                    values="equal_weight_return_pct",
                )
                if 1 in pivot and 5 in pivot:
                    fold_spreads.append(
                        float((pivot[5] - pivot[1]).mean())
                    )
            bottom_is_worst = bool(
                complete and returns.loc[1] == returns.min()
            )
            top_bottom = (
                float(returns.loc[5] - returns.loc[1])
                if complete else np.nan
            )
            positive_folds = int(
                sum(value > 0 for value in fold_spreads)
            )
            rows.append({
                "candidate": candidate,
                "label": CANDIDATES[candidate]["label"],
                "horizon_weeks": horizon,
                "monotonic_spearman": monotonic_correlation,
                "top_bottom_spread_pct": top_bottom,
                "positive_spread_folds": positive_folds,
                "measured_folds": int(len(fold_spreads)),
                "bottom_quintile_is_worst": bottom_is_worst,
                "stable_monotonic_signal": bool(
                    np.isfinite(monotonic_correlation)
                    and monotonic_correlation >= 0.70
                    and top_bottom > 0
                    and positive_folds >= 5
                    and bottom_is_worst
                ),
            })
    return pd.DataFrame(rows)


def calculate_sector_exposure(
    enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for (candidate, date), group in enriched.groupby(
        ["candidate", "date"]
    ):
        usable = group.dropna(subset=["quintile", "sector"])
        top = usable[usable["quintile"] == 5]
        if top.empty or usable.empty:
            continue
        universe_share = usable["sector"].value_counts(normalize=True)
        top_share = top["sector"].value_counts(normalize=True)
        sectors = sorted(set(universe_share.index) | set(top_share.index))
        for sector in sectors:
            rows.append({
                "candidate": candidate,
                "date": str(pd.Timestamp(date).date()),
                "sector": sector,
                "top_quintile_share": float(top_share.get(sector, 0.0)),
                "universe_share": float(universe_share.get(sector, 0.0)),
                "excess_share_pp": float(
                    (top_share.get(sector, 0.0)
                     - universe_share.get(sector, 0.0)) * 100
                ),
            })
    by_date = pd.DataFrame(rows)
    summary = (
        by_date.groupby(["candidate", "sector"], as_index=False)
        .agg(
            mean_top_share=("top_quintile_share", "mean"),
            mean_universe_share=("universe_share", "mean"),
            mean_excess_share_pp=("excess_share_pp", "mean"),
            mean_abs_excess_share_pp=(
                "excess_share_pp",
                lambda values: float(np.abs(values).mean()),
            ),
        )
        .sort_values(
            ["candidate", "mean_excess_share_pp"],
            ascending=[True, False],
        )
    )
    return by_date, summary


def calculate_size_exposure(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate, group in enriched.groupby("candidate"):
        correlations = []
        top_percentiles = []
        bottom_percentiles = []
        for _, date_group in group.groupby("date"):
            pair = date_group[[
                "quality_score",
                "estimated_market_cap",
                "size_percentile",
                "quintile",
            ]].dropna()
            if len(pair) < 50:
                continue
            correlations.append(
                pair["quality_score"].corr(
                    np.log(pair["estimated_market_cap"]),
                    method="spearman",
                )
            )
            top_percentiles.extend(
                pair.loc[pair["quintile"] == 5, "size_percentile"]
            )
            bottom_percentiles.extend(
                pair.loc[pair["quintile"] == 1, "size_percentile"]
            )
        coverage = float(group["estimated_market_cap"].notna().mean())
        rows.append({
            "candidate": candidate,
            "label": CANDIDATES[candidate]["label"],
            "market_cap_proxy_coverage": coverage,
            "mean_score_log_size_rank_corr": float(
                pd.Series(correlations).mean()
            ),
            "top_quintile_mean_size_percentile": float(
                pd.Series(top_percentiles).mean()
            ),
            "bottom_quintile_mean_size_percentile": float(
                pd.Series(bottom_percentiles).mean()
            ),
            "top_minus_bottom_size_percentile_pp": float(
                (
                    pd.Series(top_percentiles).mean()
                    - pd.Series(bottom_percentiles).mean()
                ) * 100
            ),
        })
    return pd.DataFrame(rows)


def build_bottom_filter_frame(
    factor_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    quality = _score_frame(
        "low_accruals", factor_frames["low_accruals"]
    )
    bottom_codes: dict[pd.Timestamp, set[str]] = {}
    for date, group in quality.groupby("date"):
        cutoff = group["quality_score"].quantile(0.20)
        bottom_codes[pd.Timestamp(date)] = set(
            group.loc[group["quality_score"] <= cutoff, "code"]
        )
    reference = factor_frames[MATCHED_REFERENCE].copy()
    keep = [
        str(code) not in bottom_codes.get(pd.Timestamp(date), set())
        for code, date in zip(reference["code"], reference["date"])
    ]
    return reference.loc[keep].reset_index(drop=True)


def evaluate_filter_gate(
    folds_frame: pd.DataFrame,
    summary_table: pd.DataFrame,
) -> dict:
    reference = summary_table.set_index("strategy").loc[MATCHED_REFERENCE]
    filtered = summary_table.set_index("strategy").loc[FILTER_NAME]
    paired = (
        folds_frame.pivot(
            index="fold",
            columns="strategy",
            values="test_total_return",
        )
        .dropna()
    )
    increments = paired[FILTER_NAME] - paired[MATCHED_REFERENCE]
    turnover_limit = (
        float(reference["mean_annualized_selection_turnover_pct"]) * 1.10
    )
    drawdown_floor = float(reference["worst_fold_drawdown"]) - 3.0
    result = {
        "compounded_incremental_return_pp": (
            float(filtered["compounded_oos_return"])
            - float(reference["compounded_oos_return"])
        ),
        "median_fold_incremental_return_pp": float(increments.median()),
        "positive_incremental_folds": int((increments > 0).sum()),
        "measured_folds": int(len(increments)),
        "filtered_annualized_turnover_pct": float(
            filtered["mean_annualized_selection_turnover_pct"]
        ),
        "turnover_limit_pct": turnover_limit,
        "filtered_worst_drawdown_pct": float(
            filtered["worst_fold_drawdown"]
        ),
        "drawdown_floor_pct": drawdown_floor,
    }
    result["passed"] = bool(
        result["compounded_incremental_return_pp"] > 0
        and result["median_fold_incremental_return_pp"] > 0
        and result["positive_incremental_folds"] >= 5
        and result["filtered_annualized_turnover_pct"] <= turnover_limit
        and result["filtered_worst_drawdown_pct"] >= drawdown_floor
    )
    return result


def run_filter_backtest(
    *,
    factor_frames: dict[str, pd.DataFrame],
    shared: BacktestEngine,
    index_data: pd.DataFrame,
    folds: list[Fold],
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_config = _base_config(initial_cash)
    filtered_frame = build_bottom_filter_frame(factor_frames)
    reference_frames = {
        MATCHED_REFERENCE: factor_frames[MATCHED_REFERENCE],
    }
    filtered_frames = {MATCHED_REFERENCE: filtered_frame}
    rows: list[dict] = []
    for fold in folds:
        for strategy, frames in [
            (MATCHED_REFERENCE, reference_frames),
            (FILTER_NAME, filtered_frames),
        ]:
            metrics = _run_period(
                base_config=copy.deepcopy(base_config),
                candidate=MATCHED_REFERENCE,
                factor_frames=frames,
                shared=shared,
                index_data=index_data,
                start=fold.test_start,
                end=fold.test_end,
            )
            rows.append({
                "fold": fold.fold,
                "strategy": strategy,
                "label": (
                    CANDIDATES[MATCHED_REFERENCE]["label"]
                    if strategy == MATCHED_REFERENCE else FILTER_LABEL
                ),
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **{f"test_{key}": value for key, value in metrics.items()},
            })
    fold_results = pd.DataFrame(rows)
    summaries = []
    for strategy, label in [
        (
            MATCHED_REFERENCE,
            CANDIDATES[MATCHED_REFERENCE]["label"],
        ),
        (FILTER_NAME, FILTER_LABEL),
    ]:
        summaries.append({
            "strategy": strategy,
            "label": label,
            **summarize_rows(
                fold_results[fold_results["strategy"] == strategy]
            ),
        })
    summary = pd.DataFrame(summaries)
    gate = evaluate_filter_gate(fold_results, summary)
    return fold_results, summary, gate


def validate_outputs(
    *,
    enriched: pd.DataFrame,
    quintile_by_date: pd.DataFrame,
    size_exposure: pd.DataFrame,
    filter_folds: pd.DataFrame,
    filter_gate: dict,
    expected_folds: int,
) -> dict:
    checks = [
        {
            "check": "候选日期股票唯一",
            "passed": bool(
                enriched.duplicated(
                    ["candidate", "date", "code"]
                ).sum() == 0
            ),
        },
        {
            "check": "无未来财报匹配",
            "passed": bool(
                (
                    pd.to_datetime(enriched["source_availability_date"])
                    <= pd.to_datetime(enriched["date"])
                ).all()
            ),
        },
        {
            "check": "五分位标签合法",
            "passed": bool(
                set(
                    quintile_by_date["quintile"].dropna().astype(int)
                ).issubset({1, 2, 3, 4, 5})
            ),
        },
        {
            "check": "规模代理覆盖率至少80%",
            "passed": bool(
                (
                    size_exposure["market_cap_proxy_coverage"] >= 0.80
                ).all()
            ),
        },
        {
            "check": "过滤策略窗口完整",
            "passed": bool(
                len(filter_folds)
                == expected_folds * 2
                and filter_folds["fold"].nunique() == expected_folds
            ),
        },
        {
            "check": "过滤门槛字段完整",
            "passed": bool(
                all(
                    key in filter_gate
                    for key in [
                        "compounded_incremental_return_pp",
                        "median_fold_incremental_return_pp",
                        "positive_incremental_folds",
                        "passed",
                    ]
                )
            ),
        },
    ]
    return {
        "status": (
            "passed" if all(item["passed"] for item in checks)
            else "failed"
        ),
        "checks": checks,
    }


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{float(value):+.{digits}f}%"


def build_html_report(
    *,
    result: dict,
    quintile_summary: pd.DataFrame,
    shape: pd.DataFrame,
    sector_summary: pd.DataFrame,
    size_exposure: pd.DataFrame,
    filter_folds: pd.DataFrame,
    filter_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    low_accrual_4 = shape[
        (shape["candidate"] == "low_accruals")
        & (shape["horizon_weeks"] == 4)
    ].iloc[0]
    gate = result["filter_gate"]
    if gate["passed"]:
        hero_tag = "WATCHLIST / 仅过滤器候选"
        hero_title = "低应计不适合主排名，但底部排除值得继续验证"
        hero_body = (
            "五分位形态不足以支持Top 10主排名；预先声明的底部20%过滤"
            "在七窗口增量门槛上通过。它仍不是正式策略，只能进入压力测试。"
        )
    else:
        hero_tag = "REJECT / 不进入正式策略"
        hero_title = "低应计既没有形成稳定排序，也未证明过滤价值"
        hero_body = (
            "因子层面的正Rank IC没有形成稳定五分位阶梯；剔除最差20%"
            "后也未通过同池低波动增量门槛。继续调权或扩大剔除比例缺乏依据。"
        )

    quintile_cards = []
    labels = {
        "low_accruals": "低应计",
        "working_capital_discipline": "营运资本纪律",
        "accruals50_working_capital50": "50/50复合",
    }
    for candidate in QUALITY_CANDIDATES:
        current = quintile_summary[
            (quintile_summary["candidate"] == candidate)
            & (quintile_summary["horizon_weeks"] == 4)
        ].sort_values("quintile")
        bars = []
        max_abs = max(
            float(current["mean_return_pct"].abs().max()), 0.01
        )
        for _, row in current.iterrows():
            value = float(row["mean_return_pct"])
            width = 12 + 78 * abs(value) / max_abs
            bars.append(
                f"<div class='qrow'><span>Q{int(row['quintile'])}</span>"
                f"<div class='track'><i class='{'pos' if value >= 0 else 'neg'}' "
                f"style='width:{width:.1f}%'></i></div>"
                f"<b>{value:+.2f}%</b></div>"
            )
        diagnostic = shape[
            (shape["candidate"] == candidate)
            & (shape["horizon_weeks"] == 4)
        ].iloc[0]
        quintile_cards.append(
            "<article class='factor-card'>"
            f"<h3>{html.escape(labels[candidate])}</h3>"
            + "".join(bars)
            + f"<p>Q5−Q1 {float(diagnostic['top_bottom_spread_pct']):+.2f}pp · "
            f"单调相关 {float(diagnostic['monotonic_spearman']):+.2f} · "
            f"正价差窗口 {int(diagnostic['positive_spread_folds'])}/"
            f"{int(diagnostic['measured_folds'])}</p></article>"
        )

    filter_pivot = filter_folds.pivot(
        index="fold",
        columns="strategy",
        values="test_total_return",
    )
    fold_cells = []
    for fold, row in filter_pivot.iterrows():
        delta = float(row[FILTER_NAME] - row[MATCHED_REFERENCE])
        fold_cells.append(
            f"<div class='{'good' if delta > 0 else 'bad'}'>"
            f"{delta:+.2f}pp</div>"
        )
    sector_rows = []
    for candidate in QUALITY_CANDIDATES:
        current = (
            sector_summary[sector_summary["candidate"] == candidate]
            .sort_values("mean_excess_share_pp", ascending=False)
            .head(3)
        )
        sector_rows.append(
            "<tr>"
            f"<th>{html.escape(labels[candidate])}</th>"
            f"<td>{'；'.join(html.escape(str(row.sector)) + ' ' + f'{row.mean_excess_share_pp:+.1f}pp' for row in current.itertuples())}</td>"
            "</tr>"
        )
    size_rows = []
    for row in size_exposure.itertuples():
        size_rows.append(
            "<tr>"
            f"<th>{html.escape(str(row.label))}</th>"
            f"<td>{row.market_cap_proxy_coverage:.1%}</td>"
            f"<td>{row.mean_score_log_size_rank_corr:+.2f}</td>"
            f"<td>{row.top_quintile_mean_size_percentile:.1%}</td>"
            f"<td>{row.top_minus_bottom_size_percentile_pp:+.1f}pp</td>"
            "</tr>"
        )
    filter_rows = []
    for row in filter_summary.itertuples():
        filter_rows.append(
            "<tr>"
            f"<th>{html.escape(str(row.label))}</th>"
            f"<td>{row.compounded_oos_return:+.2f}%</td>"
            f"<td>{row.median_fold_return:+.2f}%</td>"
            f"<td>{row.mean_annualized_selection_turnover_pct:.1f}%</td>"
            f"<td>{row.worst_fold_drawdown:.2f}%</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>质量因子 IC 到组合收益诊断</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f3f6fa;color:#172033;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;
line-height:1.62}} main{{max-width:1120px;margin:0 auto;padding:48px 24px 72px}}
.eyebrow{{color:#2f65a7;font-weight:700;letter-spacing:.08em}} h1{{font-size:40px;
margin:8px 0}} .sub{{color:#66758d}} .hero{{background:#172d46;color:#fff;
padding:34px 38px;border-radius:20px;margin:26px 0 22px;box-shadow:0 14px 38px #1a355322}}
.hero .tag{{display:inline-block;background:#f4d9c6;color:#6b2f1d;padding:5px 11px;
border-radius:999px;font-size:13px;font-weight:800}} .hero h2{{font-size:29px;margin:14px 0 8px}}
.hero p{{color:#d7e1ed;max-width:900px}} .metrics{{display:grid;
grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}} .metric,section,
.factor-card{{background:#fff;border:1px solid #dce4ee;border-radius:16px}}
.metric{{padding:18px}} .metric span{{display:block;color:#68788e;font-size:13px}}
.metric b{{display:block;font-size:25px;margin:5px 0}} .metric small{{color:#68788e}}
section{{padding:28px;margin:18px 0}} h2{{font-size:25px;margin:0 0 12px}}
.factor-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.factor-card{{padding:19px}} .factor-card h3{{margin:0 0 12px}} .factor-card p{{color:#65748a;font-size:13px}}
.qrow{{display:grid;grid-template-columns:28px 1fr 62px;gap:8px;align-items:center;margin:8px 0}}
.qrow b{{text-align:right;font-size:13px}} .track{{height:12px;background:#edf1f6;border-radius:8px;
overflow:hidden}} .track i{{display:block;height:100%;border-radius:8px}} .pos{{background:#35689f}}
.neg{{background:#b97826}} table{{width:100%;border-collapse:collapse;margin-top:12px}}
th,td{{padding:11px 9px;border-bottom:1px solid #dbe3ed;text-align:right;font-size:14px}}
th:first-child,td:first-child{{text-align:left}} thead th{{background:#f5f7fa;color:#58677a}}
.heat{{display:grid;grid-template-columns:110px repeat(7,1fr);gap:1px;background:#dbe3ed;
border:1px solid #dbe3ed;margin-top:12px}} .heat>*{{background:#fff;padding:12px;text-align:center}}
.heat .head{{background:#f4f6f9;font-weight:700}} .heat .good{{background:#dce9f5;color:#174c7c}}
.heat .bad{{background:#f5e6d2;color:#754512}} .callout{{border-left:4px solid #b97826;
background:#fbf6ee;padding:14px 17px;margin-top:16px}} .ledger a{{color:#245e9d}}
footer{{text-align:center;color:#77869a;margin-top:28px;font-size:13px}}
@media(max-width:850px){{.metrics,.factor-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:600px){{main{{padding:26px 14px}} h1{{font-size:31px}}
.metrics,.factor-grid{{grid-template-columns:1fr}} section{{padding:20px;overflow:auto}}}}
</style></head><body><main>
<div class="eyebrow">组合研究 · 样本截至2025年5月</div>
<h1>质量因子为何没有变成组合收益</h1>
<div class="sub">月频五分位 · 未来4/8周 · 七个不重叠窗口 · 行业与规模归因 · 冻结底部20%过滤测试</div>
<div class="hero"><span class="tag">{hero_tag}</span><h2>{hero_title}</h2>
<p>{hero_body}</p></div>
<div class="metrics">
<div class="metric"><span>低应计4周Q5−Q1</span><b>{float(low_accrual_4['top_bottom_spread_pct']):+.2f}pp</b><small>高分减低分</small></div>
<div class="metric"><span>五分位单调相关</span><b>{float(low_accrual_4['monotonic_spearman']):+.2f}</b><small>1到5组 Spearman</small></div>
<div class="metric"><span>过滤器七窗胜出</span><b>{gate['positive_incremental_folds']}/{gate['measured_folds']}</b><small>相对同池低波动</small></div>
<div class="metric"><span>过滤器复合增量</span><b>{gate['compounded_incremental_return_pp']:+.2f}pp</b><small>{'通过' if gate['passed'] else '未通过'}冻结门槛</small></div>
</div>
<section><h2>五分位曲线：看IC来自“选出赢家”还是“避开输家”</h2>
<p>Q1为质量分最低20%，Q5为最高20%；收益均从信号后的下一交易日开盘开始。图示为各月份等权后再平均的未来4周收益。</p>
<div class="factor-grid">{''.join(quintile_cards)}</div></section>
<section><h2>过滤器检验：只剔除低应计最差20%</h2>
<p>其余规则完全沿用非金融同池低波动参考：月频Top 10、缓冲30、每行业最多2只、等权、下一交易日开盘和真实交易摩擦。</p>
<table><thead><tr><th>策略</th><th>七窗复合</th><th>窗口中位</th><th>年化换手</th><th>最差回撤</th></tr></thead>
<tbody>{''.join(filter_rows)}</tbody></table>
<div class="heat"><div class="head">增量</div>{''.join(f'<div class="head">窗{i}</div>' for i in range(1,8))}
<div>过滤−参考</div>{''.join(fold_cells)}</div>
<div class="callout"><strong>冻结门槛：</strong>复合与窗口中位增量&gt;0，至少5/7窗口胜出，换手不超过参考110%，最差回撤恶化不超过3个百分点。当前为<strong>{gate['positive_incremental_folds']}/{gate['measured_folds']}窗口胜出</strong>，年化换手<strong>{gate['filtered_annualized_turnover_pct']:.1f}%</strong>（上限{gate['turnover_limit_pct']:.1f}%）；判定：<strong>{'通过研究门槛，但不得直接进入正式策略' if gate['passed'] else '淘汰'}</strong>。</div>
</section>
<section><h2>行业暴露：最高分组是否只是行业下注</h2>
<p>以下列出各候选Q5相对其当月可用股票池平均超配最多的三个行业。因子已做行业中位数中性化，但极端组仍可能残留行业结构。</p>
<table><thead><tr><th>因子</th><th>Q5平均超配最多行业</th></tr></thead>
<tbody>{''.join(sector_rows)}</tbody></table></section>
<section><h2>规模暴露：使用公告时点股本估算代理</h2>
<p>规模代理＝信号日收盘价×（最新已公告年报净利润÷基本EPS）。它适合判断相对大小倾向，不等于交易所自由流通市值。</p>
<table><thead><tr><th>因子</th><th>代理覆盖</th><th>分数/规模相关</th><th>Q5平均规模分位</th><th>Q5−Q1规模分位</th></tr></thead>
<tbody>{''.join(size_rows)}</tbody></table></section>
<section><h2>PM判断与下一动作</h2>
<ul>
<li><strong>Actionability：</strong>{'仅保留为过滤器观察候选；下一步做费用、行业剔除和规模压力测试。' if gate['passed'] else 'Pass；不继续调权、不扩大剔除比例。'}</li>
<li><strong>Variant wedge：</strong>{'若存在，只是避开会计质量尾部，而不是稳定选出最高收益股票。' if gate['passed'] else '当前没有可识别的稳定可交易增量。'}</li>
<li><strong>First rejection：</strong>五分位不单调或过滤器少于5/7窗口胜出。</li>
<li><strong>What would make it investable：</strong>独立未见样本中继续稳定改善低波动参考，并通过真实资金规模与费用压力测试。</li>
<li><strong>What would kill it：</strong>过滤增量集中于单一窗口、单一行业或规模尾部。</li>
</ul></section>
<section><h2>限制与证据边界</h2><ul>
<li>规模指标是年报净利润/EPS推算的股本代理，存在送转、回购和加权EPS误差；只用于相对暴露诊断。</li>
<li>历史财报接口可能回填后续修订；公告日控制了可用时点，但不能完全恢复修订前数值。</li>
<li>本轮仍使用截至2025年5月的既有研究样本，不替代未来3–6个月模拟盘。</li>
<li>本报告是策略候选筛选，不构成个股建议、收益预测或实盘承诺。</li>
</ul></section>
<section><h2>来源账本</h2><ul class="ledger">
<li><a href="quintile_summary.csv">quintile_summary.csv</a>：三候选五分位4/8周收益。</li>
<li><a href="quintile_shape.csv">quintile_shape.csv</a>：单调性、价差与窗口稳定性。</li>
<li><a href="filter_fold_results.csv">filter_fold_results.csv</a>：过滤器与同池低波动逐窗口回测。</li>
<li><a href="sector_exposure_summary.csv">sector_exposure_summary.csv</a>：Q5行业超配。</li>
<li><a href="size_exposure.csv">size_exposure.csv</a>：规模代理覆盖与倾向。</li>
<li><a href="validation_checks.json">validation_checks.json</a>：唯一性、时点、覆盖与窗口完整性检查。</li>
</ul></section>
<footer>研究状态：{'Watchlist filter candidate' if gate['passed'] else 'Reject'} · 正式策略保持不变</footer>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def run_diagnostic(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    train_months: int = 24,
    test_months: int = 6,
    initial_cash: float = 100_000,
    report_start_year: int = 2018,
    report_end_year: int = 2024,
    statement_cache: Path = DEFAULT_STATEMENT_CACHE,
    income_cache: Path = DEFAULT_INCOME_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    folds = generate_folds(start, end, train_months, test_months)
    if not folds:
        raise ValueError("区间不足以形成走步窗口")
    report_dates = annual_report_dates(
        report_start_year, report_end_year
    )
    income = load_annual_reports(report_dates, income_cache)
    balance, cashflow = load_statement_supplements(
        report_dates, statement_cache
    )
    statements, source_audit = join_statement_sources(
        income, balance, cashflow
    )
    if not source_audit["passed"]:
        raise ValueError(f"三表审计未通过: {source_audit}")

    shared = BacktestEngine(_base_config(initial_cash))
    logger.info("加载历史价格、成分池与三表缓存")
    shared.load_price_data(start, end)
    if shared.universe_store is None:
        raise ValueError("诊断要求历史时点股票池")
    from src.price_feed import load_index_data
    index_data = load_index_data("000300", start, end)
    if index_data.empty:
        raise ValueError("沪深300数据为空")
    rebalance_dates = _generate_rebalance_dates(
        start, end, "monthly", shared.price_data
    )
    factor_frames = build_candidate_factor_frames(
        price_data=shared.price_data,
        statements=statements,
        rebalance_dates=rebalance_dates,
        universe_store=shared.universe_store,
    )
    enriched = build_enriched_candidates(
        factor_frames=factor_frames,
        price_data=shared.price_data,
        income=income,
        folds=folds,
    )
    quintile_by_date, quintile_summary, shape = (
        calculate_quintile_returns(enriched)
    )
    sector_by_date, sector_summary = calculate_sector_exposure(
        enriched
    )
    size_exposure = calculate_size_exposure(enriched)
    filter_folds, filter_summary, filter_gate = run_filter_backtest(
        factor_frames=factor_frames,
        shared=shared,
        index_data=index_data,
        folds=folds,
        initial_cash=initial_cash,
    )
    validation = validate_outputs(
        enriched=enriched,
        quintile_by_date=quintile_by_date,
        size_exposure=size_exposure,
        filter_folds=filter_folds,
        filter_gate=filter_gate,
        expected_folds=len(folds),
    )
    if validation["status"] != "passed":
        failures = [
            row["check"]
            for row in validation["checks"]
            if not row["passed"]
        ]
        raise ValueError(f"诊断输出校验失败: {failures}")

    stable_shapes = shape[shape["stable_monotonic_signal"]]
    decision = (
        "低应计仅作为底部排除过滤器进入压力测试；不作为主排名因子，"
        "不修改正式策略。"
        if filter_gate["passed"] else
        "低应计主排名与底部20%过滤均缺乏稳定组合增量；停止继续调权，"
        "正式策略保持不变。"
    )
    result = {
        "period": {
            "history_start": start,
            "research_start": folds[0].test_start,
            "end": end,
            "fold_count": len(folds),
        },
        "frozen_rules": {
            "quintiles": 5,
            "horizons_weeks": [4, 8],
            "filter": "exclude bottom 20% low-accrual score",
            "filter_gate": (
                "复合与窗口中位增量>0；至少5/7窗口胜出；"
                "换手<=参考110%；回撤恶化<=3个百分点"
            ),
        },
        "source_audit": source_audit,
        "market_cap_proxy": (
            "signal close × latest announced annual net profit/basic EPS"
        ),
        "stable_monotonic_shapes": stable_shapes.to_dict(
            orient="records"
        ),
        "filter_gate": filter_gate,
        "validation": validation,
        "verdict": {
            "status": (
                "watchlist_filter_candidate"
                if filter_gate["passed"] else "rejected"
            ),
            "decision": decision,
            "formal_strategy_changed": False,
            "research_boundary": (
                "这是既有样本上的诊断与预声明过滤器复验，不替代未来"
                "3–6个月未见数据。"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "quintile_returns_by_date.csv": quintile_by_date,
        "quintile_summary.csv": quintile_summary,
        "quintile_shape.csv": shape,
        "sector_exposure_by_date.csv": sector_by_date,
        "sector_exposure_summary.csv": sector_summary,
        "size_exposure.csv": size_exposure,
        "filter_fold_results.csv": filter_folds,
        "filter_strategy_summary.csv": filter_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            output_dir / filename,
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
    build_html_report(
        result=result,
        quintile_summary=quintile_summary,
        shape=shape,
        sector_summary=sector_summary,
        size_exposure=size_exposure,
        filter_folds=filter_folds,
        filter_summary=filter_summary,
        output_path=output_dir / "report.html",
    )
    logger.info(f"IC到组合收益诊断完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="诊断资产负债质量IC为何未转化为组合收益"
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--report-start-year", type=int, default=2018)
    parser.add_argument("--report-end-year", type=int, default=2024)
    parser.add_argument(
        "--statement-cache",
        type=Path,
        default=DEFAULT_STATEMENT_CACHE,
    )
    parser.add_argument(
        "--income-cache", type=Path, default=DEFAULT_INCOME_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_diagnostic(
        start=args.start,
        end=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        initial_cash=args.initial_cash,
        report_start_year=args.report_start_year,
        report_end_year=args.report_end_year,
        statement_cache=args.statement_cache,
        income_cache=args.income_cache,
        output_dir=args.output,
    )
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
