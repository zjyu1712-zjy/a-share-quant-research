"""筛选独立于估值的真实年报质量残差因子。

候选在运行前固定为四项真实年报质量指标。每个信号日先将行业中性质量
排序对行业中性1/PB排序做横截面回归，再把残差重新做行业中性化。分别在
完整历史沪深300与剔除金融行业的样本中诊断。只有两套样本均通过同一预设
门槛的候选，才允许进入组合级回测。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.config import StrategyConfig
from src.factor_screening import (
    _records,
    add_forward_returns,
    assign_folds,
    calculate_factor_correlations,
    calculate_signal_diagnostics,
    summarize_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATEMENT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "statement_factor_screening"
    / "factor_values.csv"
)
DEFAULT_VALUATION_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "fundamental_factor_screening"
    / "factor_values.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "quality_value_residual_screening"
)
QUALITY_TO_FACTOR = {
    "quality_roe_annual": "residual_roe_vs_book",
    "quality_gross_margin_annual": "residual_gross_margin_vs_book",
    "quality_net_margin_annual": "residual_net_margin_vs_book",
    "quality_cash_conversion_annual": "residual_cash_conversion_vs_book",
}
FACTOR_NAMES = list(QUALITY_TO_FACTOR.values())
UNIVERSE_VARIANTS = ["all_hs300", "ex_financials"]
VALUE_COLUMN = "value_book_to_price_mrq"


def is_financial_sector(value: object) -> bool:
    sector = str(value)
    return sector.startswith(("J66", "J67", "J68", "J69"))


def load_and_audit_sources(
    statement_path: Path = DEFAULT_STATEMENT_SOURCE,
    valuation_path: Path = DEFAULT_VALUATION_SOURCE,
) -> tuple[pd.DataFrame, dict]:
    statement = pd.read_csv(
        statement_path,
        dtype={"code": str},
        encoding="utf-8-sig",
    )
    valuation = pd.read_csv(
        valuation_path,
        dtype={"code": str},
        encoding="utf-8-sig",
    )
    for frame in [statement, valuation]:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    for column in [
        "source_notice_date",
        "source_availability_date",
        "source_report_date",
    ]:
        statement[column] = pd.to_datetime(
            statement[column], errors="coerce"
        )
    valuation["source_date"] = pd.to_datetime(
        valuation["source_date"], errors="coerce"
    )

    statement_duplicates = int(
        statement.duplicated(["date", "code"]).sum()
    )
    valuation_duplicates = int(
        valuation.duplicated(["date", "code"]).sum()
    )
    statement_columns = [
        "date",
        "code",
        "sector",
        "source_notice_date",
        "source_availability_date",
        "source_report_date",
        *QUALITY_TO_FACTOR,
    ]
    valuation_columns = [
        "date",
        "code",
        "sector",
        "source_date",
        VALUE_COLUMN,
    ]
    merged = statement[statement_columns].merge(
        valuation[valuation_columns],
        on=["date", "code"],
        how="inner",
        suffixes=("_statement", "_valuation"),
        validate="one_to_one",
    )
    sector_mismatch = int(
        (
            merged["sector_statement"].fillna("")
            != merged["sector_valuation"].fillna("")
        ).sum()
    )
    merged["sector"] = merged["sector_statement"].fillna(
        merged["sector_valuation"]
    )
    merged = merged.drop(
        columns=["sector_statement", "sector_valuation"]
    )
    future_statement = int(
        (
            merged["source_availability_date"] > merged["date"]
        ).sum()
    )
    future_valuation = int(
        (merged["source_date"] > merged["date"]).sum()
    )
    audit = {
        "statement_rows": int(len(statement)),
        "valuation_rows": int(len(valuation)),
        "joined_rows": int(len(merged)),
        "statement_key_duplicates": statement_duplicates,
        "valuation_key_duplicates": valuation_duplicates,
        "joined_key_duplicates": int(
            merged.duplicated(["date", "code"]).sum()
        ),
        "join_rate_on_statement": float(
            len(merged) / len(statement) if len(statement) else 0
        ),
        "sector_mismatches": sector_mismatch,
        "future_statement_matches": future_statement,
        "future_valuation_matches": future_valuation,
        "statement_signal_dates": int(statement["date"].nunique()),
        "valuation_signal_dates": int(valuation["date"].nunique()),
        "joined_signal_dates": int(merged["date"].nunique()),
    }
    audit["passed"] = bool(
        statement_duplicates == 0
        and valuation_duplicates == 0
        and audit["joined_key_duplicates"] == 0
        and audit["join_rate_on_statement"] >= 0.95
        and sector_mismatch == 0
        and future_statement == 0
        and future_valuation == 0
        and audit["joined_signal_dates"] >= 160
    )
    if not audit["passed"]:
        raise ValueError(f"质量与估值源数据审计失败: {audit}")
    return merged.sort_values(["date", "code"]).reset_index(drop=True), audit


def residualize_cross_section(
    frame: pd.DataFrame,
    *,
    quality_column: str,
    value_column: str = VALUE_COLUMN,
) -> tuple[pd.Series, dict]:
    """用稳健秩回归去掉质量与1/PB的同日横截面线性关系。"""
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame[[quality_column, value_column, "sector"]].dropna()
    if len(valid) < 50:
        return output, {
            "stock_count": int(len(valid)),
            "slope": np.nan,
            "pre_rank_correlation": np.nan,
            "post_rank_correlation": np.nan,
            "max_absolute_sector_median": np.nan,
        }
    quality_rank = valid[quality_column].rank(
        pct=True, method="average"
    )
    value_rank = valid[value_column].rank(
        pct=True, method="average"
    )
    design = np.column_stack([np.ones(len(valid)), value_rank.to_numpy()])
    coefficients = np.linalg.lstsq(
        design,
        quality_rank.to_numpy(),
        rcond=None,
    )[0]
    residual = pd.Series(
        quality_rank.to_numpy() - design @ coefficients,
        index=valid.index,
    )
    residual = residual - residual.groupby(valid["sector"]).transform(
        "median"
    )
    output.loc[residual.index] = residual
    post = residual.corr(value_rank, method="spearman")
    sector_medians = residual.groupby(valid["sector"]).median()
    return output, {
        "stock_count": int(len(valid)),
        "slope": float(coefficients[1]),
        "pre_rank_correlation": float(
            quality_rank.corr(value_rank, method="spearman")
        ),
        "post_rank_correlation": float(post),
        "max_absolute_sector_median": float(
            sector_medians.abs().max()
        ),
    }


def build_residual_factors(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels: list[pd.DataFrame] = []
    regression_rows: list[dict] = []
    for variant in UNIVERSE_VARIANTS:
        source = joined.copy()
        if variant == "ex_financials":
            source = source[
                ~source["sector"].map(is_financial_sector)
            ].copy()
        for date, date_frame in source.groupby("date", sort=True):
            current = date_frame.copy()
            for quality_column, factor in QUALITY_TO_FACTOR.items():
                residual, diagnostics = residualize_cross_section(
                    current,
                    quality_column=quality_column,
                )
                current[factor] = residual
                regression_rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "universe_variant": variant,
                    "factor": factor,
                    **diagnostics,
                })
            current["universe_variant"] = variant
            panels.append(current)
    return (
        pd.concat(panels, ignore_index=True),
        pd.DataFrame(regression_rows),
    )


def profile_factor_coverage(
    factors: pd.DataFrame,
    universe_store,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (variant, date), group in factors.groupby(
        ["universe_variant", "date"]
    ):
        snapshot = universe_store.snapshot_for(pd.Timestamp(date))
        if variant == "all_hs300":
            denominator = len(snapshot.codes)
        else:
            denominator = sum(
                not is_financial_sector(
                    snapshot.code_to_sector.get(code, "")
                )
                for code in snapshot.codes
            )
        for factor in FACTOR_NAMES:
            available = int(group[factor].notna().sum())
            rows.append({
                "date": str(pd.Timestamp(date).date()),
                "universe_variant": variant,
                "factor": factor,
                "universe_count": int(denominator),
                "available_count": available,
                "coverage_ratio": float(
                    available / denominator if denominator else 0
                ),
            })
    by_date = pd.DataFrame(rows)
    summary = (
        by_date.groupby(["universe_variant", "factor"])["coverage_ratio"]
        .agg(["mean", "median", "min"])
        .reset_index()
        .rename(columns={
            "mean": "mean_coverage_ratio",
            "median": "median_coverage_ratio",
            "min": "min_coverage_ratio",
        })
    )
    return by_date, summary


def screen_variants(
    aggregate: pd.DataFrame,
    by_fold: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for variant in UNIVERSE_VARIANTS:
        for factor in FACTOR_NAMES:
            four = aggregate[
                (aggregate["universe_variant"] == variant)
                & (aggregate["factor"] == factor)
                & (aggregate["horizon_weeks"] == 4)
            ]
            eight = aggregate[
                (aggregate["universe_variant"] == variant)
                & (aggregate["factor"] == factor)
                & (aggregate["horizon_weeks"] == 8)
            ]
            folds = by_fold[
                (by_fold["universe_variant"] == variant)
                & (by_fold["factor"] == factor)
                & (by_fold["horizon_weeks"] == 4)
            ]["mean_rank_ic"].dropna()
            coverage = coverage_summary[
                (coverage_summary["universe_variant"] == variant)
                & (coverage_summary["factor"] == factor)
            ]
            regression = regression_summary[
                (regression_summary["universe_variant"] == variant)
                & (regression_summary["factor"] == factor)
            ]
            if four.empty or eight.empty or coverage.empty or regression.empty:
                continue
            row4 = four.iloc[0]
            row8 = eight.iloc[0]
            median_coverage = float(
                coverage.iloc[0]["median_coverage_ratio"]
            )
            fold_ratio = float((folds > 0).mean()) if len(folds) else 0.0
            mean_post_correlation = float(
                regression.iloc[0]["mean_post_rank_correlation"]
            )
            passed = bool(
                median_coverage >= 0.60
                and abs(mean_post_correlation) <= 0.10
                and row4["mean_rank_ic"] >= 0.015
                and row4["positive_ic_ratio"] >= 0.55
                and row4["hac_t_stat"] >= 1.50
                and row4["mean_top_bottom_spread_pct"] > 0
                and fold_ratio >= 5 / 7
                and row8["mean_rank_ic"] > 0
            )
            rows.append({
                "universe_variant": variant,
                "factor": factor,
                "median_coverage_ratio": median_coverage,
                "mean_post_rank_correlation": mean_post_correlation,
                "four_week_mean_rank_ic": float(row4["mean_rank_ic"]),
                "four_week_positive_ic_ratio": float(
                    row4["positive_ic_ratio"]
                ),
                "four_week_hac_t_stat": float(row4["hac_t_stat"]),
                "four_week_mean_spread_pct": float(
                    row4["mean_top_bottom_spread_pct"]
                ),
                "positive_fold_ratio": fold_ratio,
                "eight_week_mean_rank_ic": float(row8["mean_rank_ic"]),
                "status": "promising" if passed else "rejected",
            })
    return pd.DataFrame(rows)


def build_robust_decision(screening: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_NAMES:
        current = screening[screening["factor"] == factor]
        statuses = dict(
            zip(current["universe_variant"], current["status"])
        )
        all_passed = statuses.get("all_hs300") == "promising"
        ex_financials_passed = (
            statuses.get("ex_financials") == "promising"
        )
        rows.append({
            "factor": factor,
            "all_hs300_status": statuses.get("all_hs300", "missing"),
            "ex_financials_status": statuses.get(
                "ex_financials", "missing"
            ),
            "status": (
                "promising"
                if all_passed and ex_financials_passed
                else "rejected"
            ),
        })
    return pd.DataFrame(rows)


def run_screening(
    *,
    history_start: str = "2020-01-01",
    oos_start: str = "2022-01-01",
    end: str = "2025-05-12",
    statement_source: Path = DEFAULT_STATEMENT_SOURCE,
    valuation_source: Path = DEFAULT_VALUATION_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    joined, source_audit = load_and_audit_sources(
        statement_source,
        valuation_source,
    )
    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    engine = BacktestEngine(config)
    engine.load_price_data(history_start, end)
    if engine.universe_store is None:
        raise ValueError("质量估值残差筛选要求历史时点股票池")
    signal_dates = _generate_rebalance_dates(
        oos_start, end, "weekly", engine.price_data
    )
    joined = joined[joined["date"].isin(signal_dates)].copy()
    factors, regression_by_date = build_residual_factors(joined)
    coverage_by_date, coverage_summary = profile_factor_coverage(
        factors, engine.universe_store
    )
    regression_summary = (
        regression_by_date.groupby(["universe_variant", "factor"])
        .agg(
            signal_dates=("date", "count"),
            mean_pre_rank_correlation=(
                "pre_rank_correlation", "mean"
            ),
            mean_post_rank_correlation=(
                "post_rank_correlation", "mean"
            ),
            median_post_rank_correlation=(
                "post_rank_correlation", "median"
            ),
            max_sector_median_error=(
                "max_absolute_sector_median", "max"
            ),
        )
        .reset_index()
    )

    diagnostic_frames = []
    for variant in UNIVERSE_VARIANTS:
        current = factors[
            factors["universe_variant"] == variant
        ].copy()
        diagnostics = calculate_signal_diagnostics(
            add_forward_returns(current, engine.price_data),
            factor_names=FACTOR_NAMES,
        )
        diagnostics["universe_variant"] = variant
        diagnostic_frames.append(diagnostics)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostics = assign_folds(
        diagnostics,
        start=history_start,
        end=end,
    )
    aggregate = summarize_diagnostics(
        diagnostics,
        ["universe_variant", "factor", "horizon_weeks"],
    )
    by_fold = summarize_diagnostics(
        diagnostics.dropna(subset=["fold"]),
        ["universe_variant", "fold", "factor", "horizon_weeks"],
    )
    correlation_frames = []
    correlation_summaries = []
    for variant in UNIVERSE_VARIANTS:
        current = factors[
            factors["universe_variant"] == variant
        ]
        daily, summary = calculate_factor_correlations(
            current,
            factor_names=FACTOR_NAMES,
        )
        daily["universe_variant"] = variant
        summary["universe_variant"] = variant
        correlation_frames.append(daily)
        correlation_summaries.append(summary)
    screening = screen_variants(
        aggregate,
        by_fold,
        coverage_summary,
        regression_summary,
    )
    robust = build_robust_decision(screening)
    promising = robust.loc[
        robust["status"] == "promising", "factor"
    ].tolist()
    checks = {
        **source_audit,
        "factor_key_duplicates": int(
            factors.duplicated(
                ["universe_variant", "date", "code"]
            ).sum()
        ),
        "expected_variant_factor_rows": (
            len(UNIVERSE_VARIANTS) * len(FACTOR_NAMES)
        ),
        "actual_variant_factor_rows": int(len(screening)),
        "future_statement_matches": source_audit[
            "future_statement_matches"
        ],
        "future_valuation_matches": source_audit[
            "future_valuation_matches"
        ],
    }
    checks["passed"] = bool(
        source_audit["passed"]
        and checks["factor_key_duplicates"] == 0
        and checks["actual_variant_factor_rows"]
        == checks["expected_variant_factor_rows"]
    )
    if not checks["passed"]:
        raise ValueError(f"质量估值残差结果验证失败: {checks}")
    result = {
        "period": {
            "history_start": history_start,
            "oos_start": oos_start,
            "end": end,
            "signal_dates": int(factors["date"].nunique()),
        },
        "candidate_definition": (
            "同日横截面内，将行业中性质量排序对行业中性1/PB排序回归，"
            "取残差后再次按行业中位数中性化"
        ),
        "universe_variants": {
            "all_hs300": "历史时点沪深300完整成分",
            "ex_financials": "剔除J66-J69金融行业后重新估计残差",
        },
        "screening_rule": (
            "两套股票池均须通过：中位覆盖率>=60%、与1/PB平均秩相关"
            "绝对值<=0.10、4周平均IC>=0.015、正IC周>=55%、"
            "HAC t>=1.5、Top-Bottom收益差>0、至少5/7窗口IC为正、"
            "8周平均IC>0"
        ),
        "source_audit": source_audit,
        "screening": _records(screening),
        "robust_decision": _records(robust),
        "promising_factors": promising,
        "decision": (
            "通过者进入月频Top10+缓冲30组合回测"
            if promising
            else "没有候选同时通过两套股票池，停止进入组合回测"
        ),
        "research_boundary": (
            "本轮定义在运行前冻结；不因负IC事后反转方向。历史报表仍可能"
            "含修订回填，结论属于研究筛选而非未见样本证明。"
        ),
        "validation": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "factor_values.csv": factors,
        "regression_diagnostics_by_date.csv": regression_by_date,
        "regression_diagnostics_summary.csv": regression_summary,
        "factor_quality_by_date.csv": coverage_by_date,
        "factor_quality_summary.csv": coverage_summary,
        "diagnostics_by_date.csv": diagnostics,
        "diagnostics_summary.csv": aggregate,
        "diagnostics_by_fold.csv": by_fold,
        "factor_correlations_by_date.csv": pd.concat(
            correlation_frames, ignore_index=True
        ),
        "factor_correlations_summary.csv": pd.concat(
            correlation_summaries, ignore_index=True
        ),
        "screening_decision.csv": screening,
        "robust_decision.csv": robust,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# 质量相对估值残差因子筛选",
        "",
        f"**{result['decision']}**",
        "",
        "## 稳健性决策",
        "",
        robust.to_markdown(index=False),
        "",
        "## 分股票池诊断",
        "",
        screening.to_markdown(index=False),
        "",
        "## 数据与回归质量",
        "",
        coverage_summary.to_markdown(index=False),
        "",
        regression_summary.to_markdown(index=False),
        "",
        f"- 研究边界：{result['research_boundary']}",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="质量相对估值残差筛选")
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument(
        "--statement-source",
        type=Path,
        default=DEFAULT_STATEMENT_SOURCE,
    )
    parser.add_argument(
        "--valuation-source",
        type=Path,
        default=DEFAULT_VALUATION_SOURCE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screening(
        history_start=args.history_start,
        oos_start=args.oos_start,
        end=args.end,
        statement_source=args.statement_source,
        valuation_source=args.valuation_source,
        output_dir=args.output,
    )
    print(json.dumps({
        "decision": result["decision"],
        "promising_factors": result["promising_factors"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
