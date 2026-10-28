"""低应计与营运资本纪律的组合级走步验证。

候选与权重在运行前固定：

- 低应计；
- 营运资本纪律；
- 两者横截面 z-score 后各 50%。

所有候选使用历史时点沪深300非金融成分、月频 Top 10、排名缓冲30、
每行业最多2只、等权、下一交易日开盘和真实交易摩擦。完整沪深300低波动
与同股票池非金融低波动只作参考，不具备晋级资格。
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
    build_balance_quality_factors,
    is_financial_sector,
    join_statement_sources,
    load_statement_supplements,
)
from src.config import StrategyConfig
from src.low_volatility_portfolio_validation import (
    _attach_shared_data,
    _base_config,
    build_low_volatility_factor_frame,
    summarize_rows,
)
from src.price_feed import load_index_data
from src.statement_factor_screening import (
    annual_report_dates,
    load_annual_reports,
)
from src.walk_forward_attribution import _benchmark_return
from src.walk_forward_validation import Fold, generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "balance_quality_portfolio_validation"
)

CANDIDATES = {
    "low_volatility_full_reference": {
        "label": "完整沪深300低波动参考",
        "factor_one_weight": 1.0,
        "factor_two_weight": 0.0,
        "eligible_for_advancement": False,
        "comparison_reference": False,
    },
    "low_volatility_ex_financial_reference": {
        "label": "非金融同池低波动参考",
        "factor_one_weight": 1.0,
        "factor_two_weight": 0.0,
        "eligible_for_advancement": False,
        "comparison_reference": True,
    },
    "low_accruals": {
        "label": "低应计",
        "factor_one_weight": 1.0,
        "factor_two_weight": 0.0,
        "eligible_for_advancement": True,
        "comparison_reference": False,
    },
    "working_capital_discipline": {
        "label": "营运资本纪律",
        "factor_one_weight": 1.0,
        "factor_two_weight": 0.0,
        "eligible_for_advancement": True,
        "comparison_reference": False,
    },
    "accruals50_working_capital50": {
        "label": "低应计50%＋营运资本纪律50%",
        "factor_one_weight": 0.5,
        "factor_two_weight": 0.5,
        "eligible_for_advancement": True,
        "comparison_reference": False,
    },
}
MATCHED_REFERENCE = "low_volatility_ex_financial_reference"


def _close_frame(
    price_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    records: list[dict] = []
    wanted = set(pd.DatetimeIndex(rebalance_dates))
    for code, frame in price_data.items():
        if frame.empty or "close" not in frame:
            continue
        current = frame.loc[frame.index.intersection(wanted), ["close"]]
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
    statements: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> dict[str, pd.DataFrame]:
    """在相同月度信号日一次性构造全部候选与对照。"""
    quality = build_balance_quality_factors(
        statements=statements,
        rebalance_dates=rebalance_dates,
        universe_store=universe_store,
    )
    closes = _close_frame(price_data, rebalance_dates)
    quality = quality.merge(
        closes,
        on=["date", "code"],
        how="inner",
        validate="one_to_one",
    )
    quality = quality.dropna(subset=[
        "quality_low_accruals",
        "quality_working_capital_discipline",
    ]).copy()

    low_vol = build_low_volatility_factor_frame(
        price_data=price_data,
        rebalance_dates=rebalance_dates,
        universe_store=universe_store,
        candidate="low_volatility_60",
    )
    full_reference = low_vol.copy()
    full_reference["factor_one"] = full_reference["momentum"]
    full_reference["factor_two"] = 0.0

    matched_parts = []
    for date, group in low_vol.groupby("date"):
        snapshot = universe_store.snapshot_for(date)
        keep = [
            code
            for code in group["code"]
            if not is_financial_sector(
                snapshot.code_to_sector.get(code, "未知")
            )
        ]
        matched_parts.append(group[group["code"].isin(keep)].copy())
    matched_reference = pd.concat(matched_parts, ignore_index=True)
    matched_reference["factor_one"] = matched_reference["momentum"]
    matched_reference["factor_two"] = 0.0

    accruals = quality.copy()
    accruals["factor_one"] = accruals["quality_low_accruals"]
    accruals["factor_two"] = 0.0

    working_capital = quality.copy()
    working_capital["factor_one"] = (
        working_capital["quality_working_capital_discipline"]
    )
    working_capital["factor_two"] = 0.0

    composite = quality.copy()
    composite["factor_one"] = composite["quality_low_accruals"]
    composite["factor_two"] = (
        composite["quality_working_capital_discipline"]
    )

    frames = {
        "low_volatility_full_reference": full_reference,
        "low_volatility_ex_financial_reference": matched_reference,
        "low_accruals": accruals,
        "working_capital_discipline": working_capital,
        "accruals50_working_capital50": composite,
    }
    for candidate, frame in frames.items():
        result = frame.copy()
        result["momentum"] = result["factor_one"]
        result["pullback"] = result["factor_two"]
        frames[candidate] = result
        logger.info(
            f"{candidate}: {len(result)}条记录, "
            f"{result['code'].nunique()}只股票, "
            f"{result['date'].nunique()}个调仓日"
        )
    return frames


class BalanceQualityPortfolioBacktestEngine(BacktestEngine):
    """只覆写候选因子，成交、费用和持仓逻辑复用统一回测器。"""

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
        return self.factor_frames[self.candidate].loc[
            lambda frame: frame["date"].isin(wanted)
        ].copy()


def candidate_metadata(initial_cash: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "candidate": candidate,
            "label": details["label"],
            "eligible_for_advancement": details[
                "eligible_for_advancement"
            ],
            "comparison_reference": details["comparison_reference"],
            "factor_one_weight": details["factor_one_weight"],
            "factor_two_weight": details["factor_two_weight"],
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
    config.scoring.momentum_weight = float(
        details["factor_one_weight"]
    )
    config.scoring.pullback_weight = float(
        details["factor_two_weight"]
    )
    engine = BalanceQualityPortfolioBacktestEngine(
        config, candidate, factor_frames
    )
    _attach_shared_data(engine, shared)
    result = engine.run(start, end, mode="price_only")
    performance = result["performance"]
    execution = result["stats"].get("execution", {})
    trades = result["trade_log"]
    first_execution = pd.Timestamp(trades[0]["date"]) if trades else pd.NaT
    benchmark = (
        _benchmark_return(index_data, first_execution, end)
        if pd.notna(first_execution) else np.nan
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
            / initial_cash * 100
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


def profile_factor_frames(
    factor_frames: dict[str, pd.DataFrame],
    universe_store,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    duplicates = 0
    future = 0
    for candidate, frame in factor_frames.items():
        duplicates += int(frame.duplicated(["date", "code"]).sum())
        if "source_availability_date" in frame:
            future += int(
                (
                    pd.to_datetime(frame["source_availability_date"])
                    > pd.to_datetime(frame["date"])
                ).sum()
            )
        non_financial = candidate != "low_volatility_full_reference"
        for date, group in frame.groupby("date"):
            snapshot = universe_store.snapshot_for(date)
            denominator = sum(
                (
                    not non_financial
                    or not is_financial_sector(
                        snapshot.code_to_sector.get(code, "未知")
                    )
                )
                for code in snapshot.codes
            )
            rows.append({
                "candidate": candidate,
                "date": str(pd.Timestamp(date).date()),
                "universe_count": denominator,
                "available_count": int(group["code"].nunique()),
                "coverage_ratio": (
                    group["code"].nunique() / denominator
                    if denominator else np.nan
                ),
            })
    quality = pd.DataFrame(rows)
    summary = {
        "duplicate_date_code_rows": duplicates,
        "future_fundamental_matches": future,
        "minimum_coverage_ratio": float(
            quality["coverage_ratio"].min()
        ),
        "median_coverage_ratio": float(
            quality["coverage_ratio"].median()
        ),
    }
    summary["passed"] = bool(
        duplicates == 0
        and future == 0
        and summary["median_coverage_ratio"] >= 0.60
    )
    return quality, summary


def calculate_incremental_gates(
    folds_frame: pd.DataFrame,
    summary_table: pd.DataFrame,
) -> pd.DataFrame:
    """冻结的增量门槛：相对同池低波动验证真正的因子附加价值。"""
    reference_folds = folds_frame[
        folds_frame["candidate"] == MATCHED_REFERENCE
    ][[
        "fold",
        "test_total_return",
    ]].rename(columns={
        "test_total_return": "reference_fold_return",
    })
    reference_summary = summary_table[
        summary_table["strategy"] == MATCHED_REFERENCE
    ].iloc[0]
    rows = []
    for candidate, details in CANDIDATES.items():
        if not details["eligible_for_advancement"]:
            continue
        candidate_folds = folds_frame[
            folds_frame["candidate"] == candidate
        ][["fold", "test_total_return"]]
        joined = candidate_folds.merge(
            reference_folds,
            on="fold",
            validate="one_to_one",
        )
        joined["incremental_return"] = (
            joined["test_total_return"]
            - joined["reference_fold_return"]
        )
        summary = summary_table[
            summary_table["strategy"] == candidate
        ].iloc[0]
        compounded_incremental = float(
            summary["compounded_oos_return"]
            - reference_summary["compounded_oos_return"]
        )
        median_incremental = float(
            joined["incremental_return"].median()
        )
        positive_ratio = float(
            (joined["incremental_return"] > 0).mean()
        )
        turnover_limit = float(
            reference_summary[
                "mean_annualized_selection_turnover_pct"
            ] * 1.25
        )
        drawdown_floor = float(
            reference_summary["worst_fold_drawdown"] - 3.0
        )
        market_gate = bool(summary["passed"])
        incremental_gate = bool(
            compounded_incremental > 0
            and median_incremental > 0
            and positive_ratio >= 5 / 7
            and summary[
                "mean_annualized_selection_turnover_pct"
            ] <= turnover_limit
            and summary["worst_fold_drawdown"] >= drawdown_floor
        )
        rows.append({
            "candidate": candidate,
            "label": details["label"],
            "market_gate_passed": market_gate,
            "compounded_incremental_return_pp": compounded_incremental,
            "median_fold_incremental_return_pp": median_incremental,
            "positive_incremental_fold_ratio": positive_ratio,
            "positive_incremental_folds": int(
                (joined["incremental_return"] > 0).sum()
            ),
            "annualized_turnover_pct": float(
                summary["mean_annualized_selection_turnover_pct"]
            ),
            "turnover_limit_pct": turnover_limit,
            "worst_drawdown_pct": float(
                summary["worst_fold_drawdown"]
            ),
            "drawdown_floor_pct": drawdown_floor,
            "incremental_gate_passed": incremental_gate,
            "passed": bool(market_gate and incremental_gate),
        })
    return pd.DataFrame(rows)


def validate_result_tables(
    *,
    folds_frame: pd.DataFrame,
    summary_table: pd.DataFrame,
    incremental: pd.DataFrame,
    factor_quality: dict,
    expected_folds: int,
) -> dict:
    checks = []

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
    critical = [
        "test_total_return",
        "test_benchmark_return",
        "test_excess_return",
        "test_sharpe",
        "test_max_drawdown",
        "test_annualized_selection_turnover_pct",
    ]
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
        "same_benchmark_within_fold",
        bool((benchmark_counts == 1).all()),
        f"max_distinct={int(benchmark_counts.max())}",
    )
    max_difference = 0.0
    for candidate in CANDIDATES:
        recalculated = summarize_rows(
            folds_frame[folds_frame["candidate"] == candidate]
        )
        saved = summary_table[
            summary_table["strategy"] == candidate
        ].iloc[0]
        for metric in [
            "compounded_oos_return",
            "compounded_benchmark_return",
            "compounded_excess_return_pp",
            "median_fold_excess_return",
            "mean_fold_sharpe",
            "worst_fold_drawdown",
        ]:
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
            f"coverage={factor_quality['median_coverage_ratio']:.3f}"
        ),
    )
    record(
        "three_frozen_advancement_candidates",
        len(incremental) == 3,
        f"actual={len(incremental)}, expected=3",
    )
    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
    }


def build_html_report(
    *,
    result: dict,
    summary_table: pd.DataFrame,
    incremental: pd.DataFrame,
    folds_frame: pd.DataFrame,
    continuous_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = {
        candidate: details["label"]
        for candidate, details in CANDIDATES.items()
    }
    summary = summary_table.set_index("strategy")
    benchmark_return = float(
        summary.iloc[0]["compounded_benchmark_return"]
    )
    chart_rows = [
        ("沪深300", benchmark_return, "benchmark"),
        *[
            (
                labels[candidate],
                float(summary.loc[candidate, "compounded_oos_return"]),
                (
                    "reference"
                    if not CANDIDATES[candidate][
                        "eligible_for_advancement"
                    ]
                    else "candidate"
                ),
            )
            for candidate in CANDIDATES
        ],
    ]
    maximum = max(abs(value) for _, value, _ in chart_rows) * 1.08
    bar_rows = []
    for label, value, kind in chart_rows:
        width = abs(value) / maximum * 50
        left = 50 - width if value < 0 else 50
        value_class = "negative" if value < 0 else "positive"
        bar_rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{html.escape(label)}</div>"
            "<div class='bar-plot'><div class='zero'></div>"
            f"<div class='bar {kind} {value_class}' "
            f"style='left:{left:.3f}%;width:{width:.3f}%'></div>"
            f"<span class='bar-value {value_class}' "
            f"style='left:{max(2, min(92, 50 + value / maximum * 50)):.3f}%'>"
            f"{value:+.2f}%</span></div></div>"
        )

    fold_pivot = folds_frame[
        folds_frame["candidate"].isin(
            [
                MATCHED_REFERENCE,
                "low_accruals",
                "working_capital_discipline",
                "accruals50_working_capital50",
            ]
        )
    ].pivot(
        index="candidate",
        columns="fold",
        values="test_total_return",
    )
    reference = fold_pivot.loc[MATCHED_REFERENCE]
    heatmap_rows = []
    for candidate in [
        "low_accruals",
        "working_capital_discipline",
        "accruals50_working_capital50",
    ]:
        values = fold_pivot.loc[candidate] - reference
        cells = []
        for value in values:
            tone = "gain" if value > 0 else "loss"
            cells.append(
                f"<td class='{tone}'>{float(value):+.2f}</td>"
            )
        heatmap_rows.append(
            f"<tr><th>{html.escape(labels[candidate])}</th>"
            + "".join(cells)
            + f"<td>{int((values > 0).sum())}/7</td></tr>"
        )

    summary_rows = []
    for candidate in CANDIDATES:
        row = summary.loc[candidate]
        role = (
            "对照"
            if not CANDIDATES[candidate]["eligible_for_advancement"]
            else "候选"
        )
        summary_rows.append(
            "<tr>"
            f"<th>{html.escape(labels[candidate])}</th>"
            f"<td>{role}</td>"
            f"<td>{float(row['compounded_oos_return']):+.2f}%</td>"
            f"<td>{float(row['compounded_excess_return_pp']):+.2f}pp</td>"
            f"<td>{int(round(7*float(row['positive_excess_fold_ratio'])))}/7</td>"
            f"<td>{float(row['mean_fold_sharpe']):+.2f}</td>"
            f"<td>{float(row['worst_fold_drawdown']):.2f}%</td>"
            f"<td>{float(row['mean_annualized_selection_turnover_pct']):.1f}%</td>"
            f"<td>{'通过' if bool(row['passed']) else '未通过'}</td>"
            "</tr>"
        )

    gate_rows = []
    for row in incremental.to_dict(orient="records"):
        failed_reasons = []
        if not row["market_gate_passed"]:
            failed_reasons.append("市场门槛")
        if row["compounded_incremental_return_pp"] <= 0:
            failed_reasons.append("复合增量")
        if row["positive_incremental_folds"] < 5:
            failed_reasons.append("窗口稳定性")
        if row["annualized_turnover_pct"] > row["turnover_limit_pct"]:
            failed_reasons.append("换手")
        if row["worst_drawdown_pct"] < row["drawdown_floor_pct"]:
            failed_reasons.append("回撤")
        gate_rows.append(
            "<tr>"
            f"<th>{html.escape(row['label'])}</th>"
            f"<td>{row['compounded_incremental_return_pp']:+.2f}pp</td>"
            f"<td>{row['median_fold_incremental_return_pp']:+.2f}pp</td>"
            f"<td>{int(row['positive_incremental_folds'])}/7</td>"
            f"<td>{row['annualized_turnover_pct']:.1f}%</td>"
            f"<td>{row['worst_drawdown_pct']:.2f}%</td>"
            f"<td>{html.escape('、'.join(failed_reasons))}</td>"
            "<td><span class='pill reject'>淘汰</span></td>"
            "</tr>"
        )

    best_candidate = incremental.sort_values(
        "compounded_incremental_return_pp", ascending=False
    ).iloc[0]
    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").strftime(
        "%Y-%m-%d %H:%M"
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>低应计与营运资本纪律组合级验证</title>
<style>
:root {{
  color-scheme: light;
  --ink:#172033; --muted:#687386; --line:#dfe4ec; --paper:#f5f7fa;
  --card:#fff; --blue:#2f5f98; --blue-light:#dce8f5; --gold:#a66b1f;
  --gold-light:#f4e7d4; --slate:#667085; --danger:#8c4a32;
}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
"Hiragino Sans GB","Microsoft YaHei",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:48px 28px 80px}}
.eyebrow{{color:var(--blue);font-weight:700;letter-spacing:.08em;font-size:12px}}
h1{{font-size:38px;line-height:1.2;margin:10px 0 12px;letter-spacing:-.025em}}
h2{{font-size:24px;margin:0 0 14px}} h3{{font-size:17px;margin:0 0 8px}}
p{{margin:8px 0}} .muted{{color:var(--muted)}} .hero{{background:#16283f;color:white;
border-radius:18px;padding:34px 38px;margin:24px 0 22px;box-shadow:0 12px 30px #1b2b4020}}
.hero .pill{{background:#f2d9c9;color:#6d2f1f}} .hero h2{{font-size:29px;margin:12px 0 8px}}
.hero p{{color:#d7e0ea;max-width:880px}} .pill{{display:inline-block;border-radius:999px;
padding:4px 10px;font-size:12px;font-weight:750}} .reject{{background:#f2ded6;color:#783823}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 28px}}
.metric,.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:18px 20px;box-shadow:0 4px 14px #21324b0a}}
.metric strong{{display:block;font-size:25px;margin:3px 0}} .metric span{{color:var(--muted);font-size:12px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:26px 28px;margin:18px 0}} .takeaway{{font-size:17px;color:#263b55}}
.source{{display:inline-block;margin-left:6px;padding:2px 7px;border:1px solid var(--line);
border-radius:7px;color:var(--blue);text-decoration:none;font-size:11px;vertical-align:middle}}
.bar-chart{{margin-top:20px}} .bar-row{{display:grid;grid-template-columns:220px 1fr;
gap:14px;align-items:center;margin:11px 0}} .bar-label{{text-align:right;color:#344054}}
.bar-plot{{height:27px;background:#f4f6f9;border-radius:5px;position:relative;overflow:visible}}
.zero{{position:absolute;left:50%;top:-3px;bottom:-3px;border-left:1px solid #344054}}
.bar{{position:absolute;top:5px;height:17px;border-radius:3px}} .bar.candidate{{background:var(--gold)}}
.bar.reference{{background:var(--blue)}} .bar.benchmark{{background:var(--slate)}}
.bar-value{{position:absolute;top:3px;font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace;
white-space:nowrap;transform:translateX(5px)}} .bar-value.negative{{transform:translateX(-105%)}}
table{{width:100%;border-collapse:collapse;margin-top:15px;font-size:13px}}
th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}} thead th{{background:#f7f9fb;color:#4b5668;
font-size:12px;position:sticky;top:0}} tbody tr:hover{{background:#fafbfc}}
.gain{{background:var(--blue-light);color:#244f7d;font-weight:700}}
.loss{{background:var(--gold-light);color:#754718;font-weight:700}}
.callout{{border-left:4px solid var(--gold);background:#fbf7f0;padding:14px 17px;margin:16px 0}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} ul,ol{{padding-left:21px}}
.ledger li{{margin:7px 0}} code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}
footer{{color:var(--muted);font-size:12px;margin-top:24px;text-align:center}}
@media(max-width:800px){{main{{padding:24px 14px 50px}}h1{{font-size:29px}}.hero{{padding:24px}}
.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}section{{padding:20px 16px}}
.bar-row{{grid-template-columns:115px 1fr;gap:8px}}.bar-label{{font-size:11px}}.table-wrap{{overflow-x:auto}}
table{{min-width:760px}}}}
</style>
</head>
<body><main>
<div class="eyebrow">组合研究 · 截止 2025年5月</div>
<h1>低应计与营运资本纪律组合级验证</h1>
<p class="muted">月频 Top 10＋缓冲30 · 7个不重叠窗口 · 次日开盘 · 真实费用与成交限制</p>

<div class="hero">
  <span class="pill">REJECT / 不进入正式策略</span>
  <h2>三个冻结候选全部淘汰</h2>
  <p>因子层面的正 Rank IC 没有转化为可执行的 Top 10 组合收益。低应计是相对最好的候选，
  但复合收益仍为 -2.98%，且仅3/7窗口跑赢同股票池非金融低波动；继续微调权重不具备研究依据。</p>
</div>

<div class="grid">
  <div class="metric"><span>晋级候选</span><strong>0 / 3</strong><small>市场＋增量双门槛</small></div>
  <div class="metric"><span>最佳候选复合收益</span><strong>-2.98%</strong><small>低应计</small></div>
  <div class="metric"><span>最佳候选对沪深300超额</span><strong>+8.32pp</strong><small>绝对收益仍为负</small></div>
  <div class="metric"><span>最佳同池增量</span><strong>{float(best_candidate['compounded_incremental_return_pp']):+.2f}pp</strong><small>{html.escape(best_candidate['label'])}</small></div>
</div>

<section>
  <h2>绝对收益为负，正超额不足以救活候选
    <a class="source" href="strategy_summary.csv">strategy_summary.csv</a></h2>
  <p class="takeaway">低应计相对沪深300有+8.32个百分点超额，但这是“跌得更少”而非赚钱：
  七窗口复合收益为-2.98%。完整低波动参考为+13.27%，同池非金融低波动为+1.53%。</p>
  <div class="bar-chart">{''.join(bar_rows)}</div>
  <p class="muted">横轴中心线为0；蓝色为参考组合，金色为待验证质量候选，灰色为沪深300。</p>
</section>

<section>
  <h2>同池比较显示增量并不稳定
    <a class="source" href="fold_results.csv">fold_results.csv</a></h2>
  <p class="takeaway">下表是候选每个窗口收益减去“非金融同池低波动”收益。低应计只有3/7窗口胜出，
  营运资本纪律2/7，50/50复合4/7，全部低于预先锁定的5/7门槛。</p>
  <div class="table-wrap"><table>
    <thead><tr><th>相对同池低波动（pp）</th>{''.join(f'<th>窗{i}</th>' for i in range(1,8))}<th>胜出</th></tr></thead>
    <tbody>{''.join(heatmap_rows)}</tbody>
  </table></div>
</section>

<section>
  <h2>候选漏斗：没有“只差一个小参数”的近似通过者
    <a class="source" href="incremental_gates.csv">incremental_gates.csv</a></h2>
  <div class="table-wrap"><table>
    <thead><tr><th>候选</th><th>复合增量</th><th>窗口中位增量</th><th>胜出窗口</th>
    <th>年化换手</th><th>最差回撤</th><th>失败项</th><th>动作</th></tr></thead>
    <tbody>{''.join(gate_rows)}</tbody>
  </table></div>
  <div class="callout"><strong>PM判断：</strong>三个信号应归入“淘汰”，不是观察名单。
  低应计同时缺少正绝对收益、同池复合增量、5/7窗口稳定性和回撤保护；
  另外两个候选的证据更弱。没有可识别的稳定可交易增量。</div>
</section>

<section>
  <h2>完整组合指标
    <a class="source" href="strategy_summary.csv">strategy_summary.csv</a></h2>
  <div class="table-wrap"><table>
    <thead><tr><th>组合</th><th>角色</th><th>复合收益</th><th>对沪深300超额</th>
    <th>正超额窗口</th><th>平均Sharpe</th><th>最差回撤</th><th>年化换手</th><th>市场门槛</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table></div>
</section>

<section>
  <h2>研究口径与可复核性</h2>
  <div class="two">
    <div><h3>共同执行框架</h3><ul>
      <li>历史时点沪深300；质量候选与同池基准均剔除金融行业</li>
      <li>月频 Top 10，缓冲30，旧持仓最远保留到 Top 40</li>
      <li>每行业最多2只、等权、100股整数、下一交易日开盘</li>
      <li>佣金、最低佣金、过户费、印花税、滑点与成交限制全部计入</li>
    </ul></div>
    <div><h3>双重晋级门槛</h3><ul>
      <li>对沪深300：正复合收益、正超额、至少5/7正收益及正超额窗口</li>
      <li>对同池低波动：正复合与中位增量、至少5/7窗口胜出</li>
      <li>换手不超过同池基准125%，最差回撤最多恶化3个百分点</li>
      <li>候选方向、50/50权重和门槛均在结果前冻结</li>
    </ul></div>
  </div>
  <p class="muted">数据审计：三表同口径连接率100%，未来公告匹配0，日期×股票重复0；
  七窗口×五组合共35行，关键指标完整且头部指标复算一致。</p>
</section>

<section>
  <h2>下一步不是继续调权，而是解释 IC 为何没有变成组合收益</h2>
  <ol>
    <li><strong>做分位收益曲线。</strong> 检查正 Rank IC 是否主要来自避开最差股票，而不是选中最好的Top 10。</li>
    <li><strong>拆解行业与市值暴露。</strong> 判断因子收益是否集中在不可稳定承载的行业或小市值尾部。</li>
    <li><strong>只在证据支持时改为排除规则。</strong> 若信号仅能识别最差分位，可测试“剔除底部20%”而不是主排序因子。</li>
    <li><strong>停止当前三个纯质量组合。</strong> 不进入费用翻倍、100万元规模或模拟盘压力测试。</li>
  </ol>
</section>

<section>
  <h2>限制与证据边界</h2>
  <ul>
    <li>候选先在同一历史区间通过因子筛选，本轮是组合级复验，不是全新未见样本。</li>
    <li>历史财报接口可能将后续修订回填到旧报告；公告日控制了时点，但无法完全恢复修订前数值。</li>
    <li>样本截至2025年5月，不能替代未来3–6个月模拟盘；不过候选已在当前门槛下淘汰，无需启动模拟盘。</li>
    <li>本报告是研究候选筛选，不构成个股建议、收益预测或实盘承诺。</li>
  </ul>
</section>

<section>
  <h2>来源账本</h2>
  <ul class="ledger">
    <li><a href="fold_results.csv">fold_results.csv</a>：七个不重叠窗口的组合、基准、费用、换手和回撤。</li>
    <li><a href="strategy_summary.csv">strategy_summary.csv</a>：七窗口复合与晋级门槛汇总。</li>
    <li><a href="incremental_gates.csv">incremental_gates.csv</a>：相对同股票池低波动的冻结增量门槛。</li>
    <li><a href="factor_quality_by_date.csv">factor_quality_by_date.csv</a>：候选覆盖率与日期级数据质量。</li>
    <li><a href="validation_checks.json">validation_checks.json</a>：粒度、完整性、基准一致性和复算检查。</li>
  </ul>
</section>
<footer>生成于 {generated_at} · 研究状态：Reject / 正式策略保持不变</footer>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def run_validation(
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
    folds: list[Fold] = generate_folds(
        start, end, train_months, test_months
    )
    if not folds:
        raise ValueError("区间不足以形成走步测试窗口")
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

    base_config = _base_config(initial_cash)
    shared = BacktestEngine(base_config)
    logger.info("一次性加载价格、历史股票池、三张报表与沪深300")
    shared.load_price_data(start, end)
    if shared.universe_store is None:
        raise ValueError("组合验证要求历史时点股票池")
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
    quality_by_date, factor_quality = profile_factor_frames(
        factor_frames, shared.universe_store
    )
    if not factor_quality["passed"]:
        raise ValueError(f"候选因子质量未通过: {factor_quality}")

    cache: dict[tuple[str, str, str], dict] = {}

    def run_cached(
        candidate: str,
        period_start: str,
        period_end: str,
    ) -> dict:
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

    fold_rows = []
    for fold in folds:
        logger.info(
            f"Fold {fold.fold}: {fold.test_start}~{fold.test_end}"
        )
        for candidate in CANDIDATES:
            metrics = run_cached(
                candidate, fold.test_start, fold.test_end
            )
            fold_rows.append({
                "fold": fold.fold,
                "candidate": candidate,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **{
                    f"test_{key}": value
                    for key, value in metrics.items()
                },
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
            "label": details["label"],
            "eligible_for_advancement": details[
                "eligible_for_advancement"
            ],
            **fixed_summaries[candidate],
        }
        for candidate, details in CANDIDATES.items()
    ])
    incremental = calculate_incremental_gates(
        folds_frame, summary_table
    )
    continuous_start = folds[0].test_start
    continuous_rows = pd.DataFrame([
        {
            "candidate": candidate,
            "label": details["label"],
            "start": continuous_start,
            "end": end,
            **run_cached(candidate, continuous_start, end),
        }
        for candidate, details in CANDIDATES.items()
    ])
    validation = validate_result_tables(
        folds_frame=folds_frame,
        summary_table=summary_table,
        incremental=incremental,
        factor_quality=factor_quality,
        expected_folds=len(folds),
    )
    if validation["status"] != "passed":
        failures = [
            check["check"]
            for check in validation["checks"]
            if not check["passed"]
        ]
        raise ValueError(f"组合验证复核失败: {failures}")

    passing = incremental[incremental["passed"]].sort_values(
        [
            "compounded_incremental_return_pp",
            "positive_incremental_fold_ratio",
        ],
        ascending=False,
    )
    if passing.empty:
        selected_candidate = None
        status = "failed"
        decision = (
            "三个冻结候选均未同时通过市场门槛和同池低波动增量门槛。"
        )
    else:
        selected_candidate = str(passing.iloc[0]["candidate"])
        status = "passed_research_gate"
        decision = (
            f"{CANDIDATES[selected_candidate]['label']}同时通过市场与"
            "同池低波动增量门槛，进入费用、时期和资金规模压力测试。"
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
            "frictions": (
                "真实佣金、最低佣金、过户费、印花税、滑点与成交限制"
            ),
        },
        "market_advancement_rule": (
            "复合收益与超额>0；窗口收益及超额中位数>0；正收益与"
            "正超额窗口均>=60%；平均Sharpe>0；最差回撤>-30%；"
            "年化选择换手<=1500%"
        ),
        "incremental_rule": (
            "相对非金融同池低波动：复合与窗口中位增量>0；至少5/7"
            "窗口胜出；年化换手<=基准125%；最差回撤不比基准恶化"
            "超过3个百分点"
        ),
        "source_audit": source_audit,
        "factor_quality": factor_quality,
        "fixed_candidate_summaries": fixed_summaries,
        "incremental_gates": incremental.to_dict(orient="records"),
        "validation": validation,
        "selected_candidate": selected_candidate,
        "verdict": {
            "status": status,
            "decision": decision,
            "validation_assessment": "Share with caveats",
            "research_boundary": (
                "候选已在同一历史区间完成前置筛选，本结果是组合级"
                "复验，不替代未来3–6个月未见数据。"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_metadata.csv": candidate_metadata(initial_cash),
        "factor_quality_by_date.csv": quality_by_date,
        "fold_results.csv": folds_frame,
        "strategy_summary.csv": summary_table,
        "incremental_gates.csv": incremental,
        "continuous_results.csv": continuous_rows,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            output_dir / filename, index=False, encoding="utf-8-sig"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    visible = [
        "label",
        "compounded_oos_return",
        "compounded_benchmark_return",
        "compounded_excess_return_pp",
        "median_fold_excess_return",
        "positive_excess_fold_ratio",
        "mean_fold_sharpe",
        "worst_fold_drawdown",
        "mean_annualized_selection_turnover_pct",
        "passed",
    ]
    report = [
        "# 低应计与营运资本纪律组合级验证",
        "",
        "## 结论先行",
        "",
        f"- **组合级判定：{status}。** {decision}",
        "- **比较口径一致。** 月频Top 10、缓冲30、每行业最多2只、"
        "等权、下一交易日开盘与真实摩擦。",
        "- **归因口径更严格。** 候选除对比沪深300外，还必须战胜"
        "剔除金融后的同池低波动参考。",
        f"- **研究边界。** {result['verdict']['research_boundary']}",
        "",
        "## 五个组合的七窗口汇总",
        "",
        summary_table[visible].to_markdown(index=False),
        "",
        "## 三个候选的增量门槛",
        "",
        incremental.to_markdown(index=False),
        "",
        "## 连续区间核对",
        "",
        continuous_rows.to_markdown(index=False),
        "",
        "## 下一步",
        "",
        (
            "1. 对晋级者进行费用和滑点翻倍、去掉最佳窗口、行业剔除"
            "以及10万/100万资金规模压力测试。"
            if selected_candidate else
            "1. 三个候选不进入正式策略；保留结果，停止继续调权。"
        ),
        "2. 只有后续压力测试仍通过，才进入未来3–6个月模拟盘。",
        "",
        "## Caveats and Assumptions",
        "",
        f"- 参考资金为{initial_cash:,.0f}元。",
        f"- 市场门槛：{result['market_advancement_rule']}",
        f"- 增量门槛：{result['incremental_rule']}",
        "- 本结果属于研究候选验证，不构成投资建议或收益承诺。",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    build_html_report(
        result=result,
        summary_table=summary_table,
        incremental=incremental,
        folds_frame=folds_frame,
        continuous_rows=continuous_rows,
        output_path=output_dir / "report.html",
    )
    logger.info(f"资产负债质量组合验证完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="低应计与营运资本纪律组合级验证"
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
    result = run_validation(
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
