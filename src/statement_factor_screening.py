"""按公告可用日构造真实年报质量与盈利因子并做研究筛选。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

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
from src.fundamental_factor_screening import (
    _winsorize,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "data" / "fundamental_cache" / "annual_reports"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "statement_factor_screening"
)
API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
FACTOR_NAMES = [
    "quality_roe_annual",
    "quality_gross_margin_annual",
    "quality_net_margin_annual",
    "quality_cash_conversion_annual",
    "growth_revenue_yoy",
    "growth_net_profit_yoy",
    "improvement_roe_yoy",
    "improvement_gross_margin_yoy",
]
RAW_TO_FACTOR = {
    f"{name}_raw": name for name in FACTOR_NAMES
}
RAW_COLUMNS = [
    "SECURITY_CODE",
    "SECURITY_NAME_ABBR",
    "SECURITY_TYPE",
    "REPORTDATE",
    "UPDATE_DATE",
    "NOTICE_DATE",
    "BASIC_EPS",
    "TOTAL_OPERATE_INCOME",
    "PARENT_NETPROFIT",
    "WEIGHTAVG_ROE",
    "YSTZ",
    "SJLTZ",
    "MGJYXJJE",
    "XSMLL",
]


def annual_report_dates(start_year: int, end_year: int) -> list[str]:
    return [f"{year}1231" for year in range(start_year, end_year + 1)]


def _request_page(
    session: requests.Session,
    report_date: str,
    page: int,
    *,
    retries: int = 5,
) -> dict:
    params = {
        "sortColumns": "UPDATE_DATE,SECURITY_CODE",
        "sortTypes": "-1,-1",
        "pageSize": "500",
        "pageNumber": str(page),
        "reportName": "RPT_LICO_FN_CPD",
        "columns": "ALL",
        "filter": (
            f"(REPORTDATE='{report_date[:4]}-"
            f"{report_date[4:6]}-{report_date[6:]}')"
        ),
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success") or not payload.get("result"):
                raise ValueError(payload.get("message", "empty result"))
            return payload
        except Exception as exc:  # noqa: BLE001 - bounded network retry
            last_error = exc
            time.sleep(min(1.5 * (attempt + 1), 6))
    raise RuntimeError(
        f"{report_date} page {page} 下载失败: {last_error}"
    )


def _normalize_report_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[RAW_COLUMNS].copy()
    frame = frame[
        (frame["SECURITY_TYPE"] == "A股")
        & frame["SECURITY_CODE"].astype(str).str.fullmatch(r"\d{6}")
    ]
    frame = frame.rename(
        columns={
            "SECURITY_CODE": "code",
            "SECURITY_NAME_ABBR": "name",
            "REPORTDATE": "report_date",
            "UPDATE_DATE": "update_date",
            "NOTICE_DATE": "notice_date",
            "BASIC_EPS": "eps",
            "TOTAL_OPERATE_INCOME": "revenue",
            "PARENT_NETPROFIT": "net_profit",
            "WEIGHTAVG_ROE": "roe_pct",
            "YSTZ": "revenue_yoy_pct",
            "SJLTZ": "net_profit_yoy_pct",
            "MGJYXJJE": "operating_cashflow_per_share",
            "XSMLL": "gross_margin_pct",
        }
    ).drop(columns=["SECURITY_TYPE"])
    for column in ["report_date", "update_date", "notice_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    numeric = [
        "eps",
        "revenue",
        "net_profit",
        "roe_pct",
        "revenue_yoy_pct",
        "net_profit_yoy_pct",
        "operating_cashflow_per_share",
        "gross_margin_pct",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return (
        frame.dropna(subset=["report_date", "update_date"])
        .sort_values(["code", "report_date", "update_date"])
        .drop_duplicates(["code", "report_date"], keep="last")
        .reset_index(drop=True)
    )


def update_annual_report_cache(
    *,
    report_dates: list[str],
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    page_root = cache_dir / "pages"
    page_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 quantitative-research"
    downloaded = 0
    reused = 0
    for report_date in report_dates:
        output = cache_dir / f"annual_report_{report_date}.csv"
        meta = output.with_suffix(".coverage.json")
        if output.exists() and meta.exists() and not refresh:
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            metadata["availability_rule"] = "NOTICE_DATE"
            metadata["availability_note"] = (
                "NOTICE_DATE对应年报最新公告日；UPDATE_DATE仅保留作源数据审计，"
                "不可解释为首次可用日"
            )
            meta.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reused += 1
            continue
        quarter_pages = page_root / report_date
        quarter_pages.mkdir(parents=True, exist_ok=True)
        first = _request_page(session, report_date, 1)
        pages = int(first["result"]["pages"])
        all_rows: list[dict] = []
        for page in range(1, pages + 1):
            page_path = quarter_pages / f"page_{page:04d}.json"
            if page_path.exists() and not refresh:
                payload = json.loads(page_path.read_text(encoding="utf-8"))
            else:
                payload = (
                    first
                    if page == 1
                    else _request_page(session, report_date, page)
                )
                page_path.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            all_rows.extend(payload["result"]["data"])
            if page == 1 or page % 5 == 0 or page == pages:
                logger.info(f"{report_date}: page {page}/{pages}")
        frame = _normalize_report_rows(all_rows)
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        meta.write_text(
            json.dumps(
                {
                    "report_date": report_date,
                    "pages": pages,
                    "raw_rows": len(all_rows),
                    "a_share_rows": len(frame),
                    "availability_rule": "NOTICE_DATE",
                    "availability_note": (
                        "NOTICE_DATE对应年报最新公告日；UPDATE_DATE仅保留作"
                        "源数据审计，不可解释为首次可用日"
                    ),
                    "source": "Eastmoney RPT_LICO_FN_CPD via AKShare schema",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        downloaded += 1
    return {
        "requested_periods": len(report_dates),
        "downloaded_periods": downloaded,
        "reused_periods": reused,
    }


def load_annual_reports(
    report_dates: list[str],
    cache_dir: Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    frames = []
    for report_date in report_dates:
        path = cache_dir / f"annual_report_{report_date}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
        for column in ["report_date", "update_date", "notice_date"]:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["code", "report_date", "update_date"])
        .drop_duplicates(["code", "report_date"], keep="last")
        .reset_index(drop=True)
    )


def profile_source_dates(reports: pd.DataFrame) -> dict:
    """Audit source date semantics before constructing point-in-time signals."""
    report_date = pd.to_datetime(reports["report_date"], errors="coerce")
    notice_date = pd.to_datetime(reports["notice_date"], errors="coerce")
    update_date = pd.to_datetime(reports["update_date"], errors="coerce")
    notice_lag = (notice_date - report_date).dt.days.dropna()
    update_lag = (update_date - notice_date).dt.days.dropna()
    return {
        "rows": int(len(reports)),
        "report_code_duplicates": int(
            reports.duplicated(["report_date", "code"]).sum()
        ),
        "missing_notice_date": int(notice_date.isna().sum()),
        "missing_update_date": int(update_date.isna().sum()),
        "notice_before_report": int((notice_date < report_date).sum()),
        "median_notice_lag_days": float(notice_lag.median()),
        "p90_notice_lag_days": float(notice_lag.quantile(0.90)),
        "median_update_after_notice_days": float(update_lag.median()),
        "share_update_after_notice_over_30_days": float(
            (update_lag > 30).mean()
        ),
        "availability_field": "NOTICE_DATE",
        "update_date_interpretation": (
            "源数据维护日期；其相对公告日中位滞后过长，不能作为首次可用日"
        ),
    }


def build_statement_factors(
    *,
    reports: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> pd.DataFrame:
    records: list[dict] = []
    by_code = {
        code: frame.sort_values("report_date")
        for code, frame in reports.groupby("code")
    }
    for date in pd.DatetimeIndex(rebalance_dates).sort_values():
        snapshot = universe_store.snapshot_for(date)
        for code in snapshot.codes:
            history = by_code.get(code)
            if history is None:
                continue
            eligible = history[history["notice_date"] <= date].sort_values(
                "report_date"
            )
            if eligible.empty:
                continue
            current = eligible.iloc[-1]
            previous = eligible.iloc[-2] if len(eligible) >= 2 else None
            revenue = float(current["revenue"])
            net_profit = float(current["net_profit"])
            eps = float(current["eps"])
            net_margin = (
                net_profit / revenue
                if np.isfinite(revenue) and revenue > 0
                else np.nan
            )
            cash_conversion = (
                float(current["operating_cashflow_per_share"]) / eps
                if np.isfinite(eps) and eps >= 0.05
                else np.nan
            )
            consecutive_previous = bool(
                previous is not None
                and (
                    int(current["report_date"].year)
                    - int(previous["report_date"].year)
                ) == 1
            )
            records.append({
                "date": pd.Timestamp(date),
                "code": code,
                "sector": snapshot.code_to_sector.get(code, "未知"),
                "source_update_date": current["update_date"],
                "source_notice_date": current["notice_date"],
                "source_availability_date": current["notice_date"],
                "source_report_date": current["report_date"],
                "prior_source_update_date": (
                    previous["update_date"] if previous is not None else pd.NaT
                ),
                "quality_roe_annual_raw": current["roe_pct"],
                "quality_gross_margin_annual_raw": current["gross_margin_pct"],
                "quality_net_margin_annual_raw": net_margin,
                "quality_cash_conversion_annual_raw": cash_conversion,
                "growth_revenue_yoy_raw": current["revenue_yoy_pct"],
                "growth_net_profit_yoy_raw": current["net_profit_yoy_pct"],
                "improvement_roe_yoy_raw": (
                    current["roe_pct"] - previous["roe_pct"]
                    if consecutive_previous else np.nan
                ),
                "improvement_gross_margin_yoy_raw": (
                    current["gross_margin_pct"]
                    - previous["gross_margin_pct"]
                    if consecutive_previous else np.nan
                ),
            })
    raw = pd.DataFrame(records)
    if raw.empty:
        return raw
    output = []
    for _, group in raw.groupby("date"):
        current = group.copy()
        for raw_name, factor in RAW_TO_FACTOR.items():
            clipped = _winsorize(current[raw_name])
            current[factor] = (
                clipped
                - clipped.groupby(current["sector"]).transform("median")
            )
        output.append(current)
    return pd.concat(output, ignore_index=True)


def profile_quality(
    factors: pd.DataFrame,
    universe_store,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for date, group in factors.groupby("date"):
        universe_count = len(universe_store.snapshot_for(date).codes)
        for factor in FACTOR_NAMES:
            rows.append({
                "date": str(pd.Timestamp(date).date()),
                "factor": factor,
                "universe_count": universe_count,
                "available_count": int(group[factor].notna().sum()),
                "coverage_ratio": float(
                    group[factor].notna().sum() / universe_count
                ),
            })
    by_date = pd.DataFrame(rows)
    future_availability = int(
        (
            pd.to_datetime(factors["source_availability_date"])
            > pd.to_datetime(factors["date"])
        ).sum()
    )
    duplicates = int(factors.duplicated(["date", "code"]).sum())
    coverage = (
        by_date.groupby("factor")["coverage_ratio"]
        .agg(["mean", "median", "min"])
        .reset_index()
    )
    coverage.columns = [
        "factor",
        "mean_coverage_ratio",
        "median_coverage_ratio",
        "min_coverage_ratio",
    ]
    return by_date, {
        "rows": len(factors),
        "signal_dates": int(factors["date"].nunique()),
        "stocks": int(factors["code"].nunique()),
        "duplicate_date_code_rows": duplicates,
        "future_availability_matches": future_availability,
        "coverage": _records(coverage),
        "passed": bool(duplicates == 0 and future_availability == 0),
    }


def calculate_low_volatility_correlations(
    factors: pd.DataFrame,
    reference_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["low_volatility_60", "sector_neutral_low_volatility_60"]
    if not reference_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    reference = pd.read_csv(
        reference_path,
        dtype={"code": str},
        parse_dates=["date"],
        encoding="utf-8-sig",
    )
    reference["code"] = reference["code"].astype(str).str.zfill(6)
    joined = factors[["date", "code", *FACTOR_NAMES]].merge(
        reference[["date", "code", *columns]],
        on=["date", "code"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for date, group in joined.groupby("date"):
        for factor in FACTOR_NAMES:
            for reference_factor in columns:
                pair = group[[factor, reference_factor]].dropna()
                correlation = (
                    pair[factor].corr(pair[reference_factor], method="spearman")
                    if len(pair) >= 50 else np.nan
                )
                rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "reference_factor": reference_factor,
                    "stock_count": len(pair),
                    "rank_correlation": correlation,
                })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.DataFrame()
    summary = (
        daily.groupby(["factor", "reference_factor"])["rank_correlation"]
        .agg(signal_dates="count", mean="mean", median="median", max="max")
        .reset_index()
    )
    return daily, summary


def screen_factors(
    aggregate: pd.DataFrame,
    by_fold: pd.DataFrame,
    quality_by_date: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_NAMES:
        row4 = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 4)
        ]
        row8 = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 8)
        ]
        folds = by_fold[
            (by_fold["factor"] == factor)
            & (by_fold["horizon_weeks"] == 4)
        ]["mean_rank_ic"].dropna()
        coverage = quality_by_date.loc[
            quality_by_date["factor"] == factor, "coverage_ratio"
        ]
        if row4.empty or row8.empty:
            continue
        four = row4.iloc[0]
        eight = row8.iloc[0]
        fold_ratio = float((folds > 0).mean()) if len(folds) else 0.0
        median_coverage = float(coverage.median()) if len(coverage) else 0.0
        passed = bool(
            median_coverage >= 0.60
            and four["mean_rank_ic"] >= 0.015
            and four["positive_ic_ratio"] >= 0.55
            and four["hac_t_stat"] >= 1.50
            and four["mean_top_bottom_spread_pct"] > 0
            and fold_ratio >= 5 / 7
            and eight["mean_rank_ic"] > 0
        )
        rows.append({
            "factor": factor,
            "median_coverage_ratio": median_coverage,
            "four_week_mean_rank_ic": float(four["mean_rank_ic"]),
            "four_week_positive_ic_ratio": float(four["positive_ic_ratio"]),
            "four_week_hac_t_stat": float(four["hac_t_stat"]),
            "four_week_mean_spread_pct": float(
                four["mean_top_bottom_spread_pct"]
            ),
            "positive_fold_ratio": fold_ratio,
            "eight_week_mean_rank_ic": float(eight["mean_rank_ic"]),
            "status": "promising" if passed else "rejected",
        })
    return pd.DataFrame(rows)


def run_screening(
    *,
    history_start: str = "2020-01-01",
    oos_start: str = "2022-01-01",
    end: str = "2025-05-12",
    report_start_year: int = 2018,
    report_end_year: int = 2024,
    cache_dir: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
    download: bool = True,
    refresh: bool = False,
) -> dict:
    dates = annual_report_dates(report_start_year, report_end_year)
    download_result = (
        update_annual_report_cache(
            report_dates=dates,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        if download else {}
    )
    reports = load_annual_reports(dates, cache_dir)
    if reports.empty:
        raise ValueError("年报缓存为空")
    source_date_audit = profile_source_dates(reports)
    if (
        source_date_audit["report_code_duplicates"]
        or source_date_audit["missing_notice_date"]
        or source_date_audit["notice_before_report"]
    ):
        raise ValueError(f"年报源日期审计失败: {source_date_audit}")
    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    engine = BacktestEngine(config)
    engine.load_price_data(history_start, end)
    if engine.universe_store is None:
        raise ValueError("要求历史时点股票池")
    signals = _generate_rebalance_dates(
        oos_start, end, "weekly", engine.price_data
    )
    factors = build_statement_factors(
        reports=reports,
        rebalance_dates=signals,
        universe_store=engine.universe_store,
    )
    quality_by_date, quality = profile_quality(
        factors,
        engine.universe_store,
    )
    diagnostics = calculate_signal_diagnostics(
        add_forward_returns(factors, engine.price_data),
        factor_names=FACTOR_NAMES,
    )
    diagnostics = assign_folds(
        diagnostics,
        start=history_start,
        end=end,
    )
    aggregate = summarize_diagnostics(
        diagnostics, ["factor", "horizon_weeks"]
    )
    by_fold = summarize_diagnostics(
        diagnostics.dropna(subset=["fold"]),
        ["fold", "factor", "horizon_weeks"],
    )
    correlations_daily, correlations_summary = calculate_factor_correlations(
        factors, factor_names=FACTOR_NAMES
    )
    ref_daily, ref_summary = calculate_low_volatility_correlations(
        factors,
        PROJECT_ROOT
        / "data"
        / "backtest_results"
        / "factor_screening"
        / "factor_values.csv",
    )
    decision = screen_factors(aggregate, by_fold, quality_by_date)
    promising = decision.loc[
        decision["status"] == "promising", "factor"
    ].tolist()
    checks = {
        "factor_key_duplicates": int(
            factors.duplicated(["date", "code"]).sum()
        ),
        "diagnostic_key_duplicates": int(
            diagnostics.duplicated(
                ["date", "factor", "horizon_weeks"]
            ).sum()
        ),
        "future_availability_matches": quality[
            "future_availability_matches"
        ],
        "factor_count": int(decision["factor"].nunique()),
        "expected_factor_count": len(FACTOR_NAMES),
    }
    checks["passed"] = bool(
        checks["factor_key_duplicates"] == 0
        and checks["diagnostic_key_duplicates"] == 0
        and checks["future_availability_matches"] == 0
        and checks["factor_count"] == checks["expected_factor_count"]
    )
    if not checks["passed"]:
        raise ValueError(f"年报因子验证失败: {checks}")
    result = {
        "period": {
            "history_start": history_start,
            "oos_start": oos_start,
            "end": end,
            "report_years": [report_start_year, report_end_year],
            "signal_dates": int(factors["date"].nunique()),
        },
        "source": (
            "东方财富RPT_LICO_FN_CPD；使用NOTICE_DATE（最新公告日期）"
            "作为可用日"
        ),
        "download": download_result,
        "source_date_audit": source_date_audit,
        "data_quality": quality,
        "screening_rule": (
            "中位覆盖率>=60%；4周平均IC>=0.015、IC为正周占比>=55%、"
            "HAC t>=1.5、Top-Bottom收益差>0、至少5/7窗口IC为正、"
            "8周平均IC>0"
        ),
        "screening": _records(decision),
        "promising_factors": promising,
        "decision": (
            "通过者进入月频Top10+缓冲30组合回测"
            if promising else "没有因子通过门槛，停止进入组合回测"
        ),
        "validation": checks,
        "research_boundary": (
            "接口只提供当前可见的历史报表快照，不能完全重建每次历史修订前"
            "的原始数值；NOTICE_DATE控制公告时点，但仍可能存在修订回填。"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "factor_values.csv": factors,
        "factor_quality_by_date.csv": quality_by_date,
        "diagnostics_by_date.csv": diagnostics,
        "diagnostics_summary.csv": aggregate,
        "diagnostics_by_fold.csv": by_fold,
        "factor_correlations_by_date.csv": correlations_daily,
        "factor_correlations_summary.csv": correlations_summary,
        "correlations_with_low_volatility_by_date.csv": ref_daily,
        "correlations_with_low_volatility_summary.csv": ref_summary,
        "screening_decision.csv": decision,
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
    (output_dir / "source_date_audit.json").write_text(
        json.dumps(source_date_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# 真实年报质量与盈利因子筛选",
        "",
        f"**{result['decision']}**",
        "",
        decision.to_markdown(index=False),
        "",
        "## 数据质量",
        "",
        pd.DataFrame(quality["coverage"]).to_markdown(index=False),
        "",
        (
            "- 原始年报记录："
            f"{source_date_audit['rows']:,}；报告日×股票重复："
            f"{source_date_audit['report_code_duplicates']}"
        ),
        (
            "- 公告相对报告期末中位滞后："
            f"{source_date_audit['median_notice_lag_days']:.0f}天；"
            "可用日字段：NOTICE_DATE"
        ),
        (
            "- 未来公告日匹配："
            f"{quality['future_availability_matches']}"
        ),
        f"- 重复日期×股票：{quality['duplicate_date_code_rows']}",
        f"- 边界：{result['research_boundary']}",
        "",
        "## 总体诊断",
        "",
        aggregate.to_markdown(index=False),
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实年报质量因子筛选")
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--report-start-year", type=int, default=2018)
    parser.add_argument("--report-end-year", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screening(
        history_start=args.history_start,
        oos_start=args.oos_start,
        end=args.end,
        report_start_year=args.report_start_year,
        report_end_year=args.report_end_year,
        cache_dir=args.cache_dir,
        output_dir=args.output,
        download=not args.no_download,
        refresh=args.refresh,
    )
    print(json.dumps({
        "decision": result["decision"],
        "promising_factors": result["promising_factors"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
