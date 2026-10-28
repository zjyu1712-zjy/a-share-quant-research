"""历史时点基本面代理因子筛选。

使用 BaoStock 按交易日提供的历史 PE/PB/PS 数据，在历史沪深300成分池中检验：

- 估值：TTM 盈利收益率、MRQ 账面市值比；
- 质量：TTM 盈利/MRQ 净资产代理、TTM 净利率代理；
- 盈利改善：上述盈利能力代理相对一年前的变化。

所有因子均先做横截面缩尾，再减去同日同行业中位数。收益从信号后的
下一交易日开盘开始计算。本模块只做候选因子诊断，不修改正式策略。
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
from src.factor_screening import (
    _records,
    add_forward_returns,
    assign_folds,
    calculate_factor_correlations,
    calculate_signal_diagnostics,
    summarize_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "data" / "fundamental_cache" / "valuation_daily"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "fundamental_factor_screening"
)

RAW_FACTOR_NAMES = [
    "earnings_yield_ttm_raw",
    "book_to_price_mrq_raw",
    "roe_ttm_proxy_raw",
    "net_margin_ttm_proxy_raw",
    "roe_change_1y_raw",
    "net_margin_change_1y_raw",
]
FACTOR_NAMES = [
    "value_earnings_yield_ttm",
    "value_book_to_price_mrq",
    "quality_roe_ttm_proxy",
    "quality_net_margin_ttm_proxy",
    "profitability_roe_change_1y",
    "profitability_margin_change_1y",
]
RAW_TO_FACTOR = dict(zip(RAW_FACTOR_NAMES, FACTOR_NAMES))
VALUATION_FIELDS = [
    "date",
    "code",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
]


def _bao_code(code: str) -> str:
    normalized = str(code).zfill(6)
    prefix = "sh" if normalized.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}.{normalized}"


def _cache_paths(cache_dir: Path, code: str) -> tuple[Path, Path]:
    data_path = cache_dir / f"valuation_{str(code).zfill(6)}.csv"
    return data_path, data_path.with_suffix(".coverage.json")


def _cache_covers(cache_dir: Path, code: str, start: str, end: str) -> bool:
    data_path, meta_path = _cache_paths(cache_dir, code)
    if not data_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (
            pd.Timestamp(meta["requested_start"]) <= pd.Timestamp(start)
            and pd.Timestamp(meta["requested_end"]) >= pd.Timestamp(end)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _query_valuation_frame(bs, code: str, start: str, end: str) -> pd.DataFrame:
    result = bs.query_history_k_data_plus(
        _bao_code(code),
        ",".join(VALUATION_FIELDS),
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="3",
    )
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(
            f"{code} BaoStock估值查询失败: "
            f"{result.error_code} {result.error_msg}"
        )
    frame = pd.DataFrame(rows, columns=result.fields or VALUATION_FIELDS)
    if frame.empty:
        return pd.DataFrame(columns=VALUATION_FIELDS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = str(code).zfill(6)
    for column in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def update_valuation_cache(
    *,
    codes: list[str],
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict:
    """下载缺失的历史估值；每只股票单独落盘，支持中断后续跑。"""
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("缺少 baostock；请先运行 setup_mac.command") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_codes = sorted({str(code).zfill(6) for code in codes})
    pending = [
        code
        for code in normalized_codes
        if refresh or not _cache_covers(cache_dir, code, start, end)
    ]
    logger.info(
        f"历史估值缓存: 共{len(normalized_codes)}只，"
        f"命中{len(normalized_codes) - len(pending)}只，待下载{len(pending)}只"
    )
    if not pending:
        return {
            "requested": len(normalized_codes),
            "downloaded": 0,
            "failed": [],
        }

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(
            f"BaoStock登录失败: {login.error_code} {login.error_msg}"
        )
    downloaded = 0
    failed: list[dict] = []
    try:
        for index, code in enumerate(pending, start=1):
            try:
                frame = _query_valuation_frame(bs, code, start, end)
                data_path, meta_path = _cache_paths(cache_dir, code)
                frame.to_csv(data_path, index=False, encoding="utf-8-sig")
                meta_path.write_text(
                    json.dumps(
                        {
                            "code": code,
                            "requested_start": start,
                            "requested_end": end,
                            "rows": int(len(frame)),
                            "data_start": (
                                str(frame["date"].min().date())
                                if not frame.empty
                                else None
                            ),
                            "data_end": (
                                str(frame["date"].max().date())
                                if not frame.empty
                                else None
                            ),
                            "source": "baostock_history_k_data_plus",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                downloaded += 1
            except Exception as exc:  # noqa: BLE001 - continue resumable batch
                failed.append({"code": code, "error": str(exc)})
                logger.warning(f"{code} 历史估值下载失败: {exc}")
            if index == 1 or index % 25 == 0 or index == len(pending):
                logger.info(
                    f"历史估值进度 {index}/{len(pending)}；"
                    f"成功{downloaded}，失败{len(failed)}"
                )
    finally:
        bs.logout()
    return {
        "requested": len(normalized_codes),
        "downloaded": downloaded,
        "failed": failed,
    }


def load_valuation_data(
    codes: list[str],
    cache_dir: Path = DEFAULT_CACHE,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for code in sorted({str(value).zfill(6) for value in codes}):
        data_path, _ = _cache_paths(cache_dir, code)
        if not data_path.exists():
            continue
        frame = pd.read_csv(data_path, encoding="utf-8-sig")
        if frame.empty:
            result[code] = frame
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        result[code] = (
            frame.dropna(subset=["date"])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .set_index("date")
        )
    return result


def _asof_values(
    frame: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    *,
    tolerance_days: int,
) -> pd.DataFrame:
    """只向后匹配，不允许使用目标日之后的数据。"""
    left = pd.DataFrame({"target_date": pd.DatetimeIndex(target_dates).sort_values()})
    if frame.empty:
        empty = left.copy()
        empty["source_date"] = pd.NaT
        for column in ["peTTM", "pbMRQ", "psTTM"]:
            empty[column] = np.nan
        return empty
    right = (
        frame.rename_axis("source_date")
        .reset_index()
        .sort_values("source_date")
    )
    return pd.merge_asof(
        left,
        right[["source_date", "peTTM", "pbMRQ", "psTTM"]],
        left_on="target_date",
        right_on="source_date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )


def _winsorize(series: pd.Series, lower: float = 0.025, upper: float = 0.975) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    if len(clean) < 20:
        return numeric
    return numeric.clip(clean.quantile(lower), clean.quantile(upper))


def build_fundamental_factors(
    *,
    valuation_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> pd.DataFrame:
    """构造行业中性基本面因子，并保留源日期用于未来函数审计。"""
    records: list[dict] = []
    signal_dates = pd.DatetimeIndex(rebalance_dates).sort_values()
    previous_targets = signal_dates - pd.DateOffset(years=1)
    for code, frame in valuation_data.items():
        if frame.empty:
            continue
        current = _asof_values(frame, signal_dates, tolerance_days=10)
        previous = _asof_values(frame, previous_targets, tolerance_days=45)
        current["code"] = str(code).zfill(6)
        current["previous_source_date"] = previous["source_date"].to_numpy()

        pe = current["peTTM"]
        pb = current["pbMRQ"]
        ps = current["psTTM"]
        previous_pe = previous["peTTM"]
        previous_pb = previous["pbMRQ"]
        previous_ps = previous["psTTM"]
        valid_pe = pe.abs() >= 0.5
        valid_previous_pe = previous_pe.abs() >= 0.5
        valid_pb = pb > 0.05
        valid_previous_pb = previous_pb > 0.05
        valid_ps = ps >= 0
        valid_previous_ps = previous_ps >= 0

        current["earnings_yield_ttm_raw"] = (1.0 / pe).where(valid_pe)
        current["book_to_price_mrq_raw"] = (1.0 / pb).where(valid_pb)
        current["roe_ttm_proxy_raw"] = (pb / pe).where(valid_pe & valid_pb)
        current["net_margin_ttm_proxy_raw"] = (ps / pe).where(
            valid_pe & valid_ps
        )
        previous_roe = (previous_pb / previous_pe).where(
            valid_previous_pe & valid_previous_pb
        )
        previous_margin = (previous_ps / previous_pe).where(
            valid_previous_pe & valid_previous_ps
        )
        current["roe_change_1y_raw"] = (
            current["roe_ttm_proxy_raw"] - previous_roe.to_numpy()
        )
        current["net_margin_change_1y_raw"] = (
            current["net_margin_ttm_proxy_raw"] - previous_margin.to_numpy()
        )
        records.extend(current.to_dict(orient="records"))

    raw = pd.DataFrame(records)
    if raw.empty:
        return raw
    raw = raw.rename(columns={"target_date": "date"})
    filtered: list[pd.DataFrame] = []
    for date, date_frame in raw.groupby("date"):
        snapshot = universe_store.snapshot_for(date)
        current = date_frame[
            date_frame["code"].isin(snapshot.codes)
        ].copy()
        current["sector"] = current["code"].map(snapshot.code_to_sector)
        for raw_name, factor_name in RAW_TO_FACTOR.items():
            clipped = _winsorize(current[raw_name])
            current[factor_name] = (
                clipped - clipped.groupby(current["sector"]).transform("median")
            )
        filtered.append(current)
    result = pd.concat(filtered, ignore_index=True)
    logger.info(
        f"基本面候选因子: {len(result)}条，"
        f"{result['code'].nunique()}只股票，"
        f"{result['date'].nunique()}个信号日"
    )
    return result


def profile_factor_quality(factors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for date, group in factors.groupby("date"):
        universe_count = int(group["code"].nunique())
        for factor in FACTOR_NAMES:
            available = int(group[factor].notna().sum())
            rows.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "universe_count": universe_count,
                    "available_count": available,
                    "coverage_ratio": (
                        available / universe_count if universe_count else 0.0
                    ),
                }
            )
    by_date = pd.DataFrame(rows)
    future_current = int(
        (
            pd.to_datetime(factors["source_date"])
            > pd.to_datetime(factors["date"])
        ).sum()
    )
    previous_cutoff = pd.to_datetime(factors["date"]) - pd.DateOffset(years=1)
    future_previous = int(
        (
            pd.to_datetime(factors["previous_source_date"])
            > previous_cutoff
        ).sum()
    )
    key_duplicates = int(factors.duplicated(["date", "code"]).sum())
    coverage = (
        by_date.groupby("factor")["coverage_ratio"]
        .agg(["mean", "median", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_coverage_ratio",
                "median": "median_coverage_ratio",
                "min": "min_coverage_ratio",
            }
        )
    )
    summary = {
        "rows": int(len(factors)),
        "signal_dates": int(factors["date"].nunique()),
        "stocks": int(factors["code"].nunique()),
        "duplicate_date_code_rows": key_duplicates,
        "future_current_matches": future_current,
        "future_previous_matches": future_previous,
        "coverage": _records(coverage),
        "passed": bool(
            key_duplicates == 0
            and future_current == 0
            and future_previous == 0
            and not coverage.empty
            and (coverage["median_coverage_ratio"] >= 0.60).all()
        ),
    }
    return by_date, summary


def screen_fundamental_factors(
    aggregate: pd.DataFrame,
    by_fold: pd.DataFrame,
    quality_by_date: pd.DataFrame,
) -> pd.DataFrame:
    """复用价格因子阶段预先固定的门槛，避免结果出来后改规则。"""
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
        coverage = quality_by_date.loc[
            quality_by_date["factor"] == factor, "coverage_ratio"
        ]
        if four_week.empty or eight_week.empty:
            continue
        row4 = four_week.iloc[0]
        row8 = eight_week.iloc[0]
        positive_fold_ratio = (
            float((fold_values > 0).mean()) if len(fold_values) else 0.0
        )
        median_coverage = float(coverage.median()) if len(coverage) else 0.0
        passed = bool(
            median_coverage >= 0.60
            and row4["mean_rank_ic"] >= 0.015
            and row4["positive_ic_ratio"] >= 0.55
            and row4["hac_t_stat"] >= 1.50
            and row4["mean_top_bottom_spread_pct"] > 0
            and positive_fold_ratio >= 5 / 7
            and row8["mean_rank_ic"] > 0
        )
        rows.append(
            {
                "factor": factor,
                "median_coverage_ratio": median_coverage,
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
            }
        )
    return pd.DataFrame(rows)


def calculate_reference_correlations(
    factors: pd.DataFrame,
    reference_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """衡量基本面候选与已验证低波动风险特征的信息重叠。"""
    columns = [
        "low_volatility_60",
        "sector_neutral_low_volatility_60",
    ]
    if not reference_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    reference = pd.read_csv(
        reference_path,
        dtype={"code": str},
        parse_dates=["date"],
        encoding="utf-8-sig",
    )
    reference["code"] = reference["code"].astype(str).str.zfill(6)
    left = factors[["date", "code", *FACTOR_NAMES]].copy()
    joined = left.merge(
        reference[["date", "code", *columns]],
        on=["date", "code"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict] = []
    for date, group in joined.groupby("date"):
        for factor in FACTOR_NAMES:
            for reference_factor in columns:
                pair = group[[factor, reference_factor]].dropna()
                correlation = (
                    pair[factor].corr(pair[reference_factor], method="spearman")
                    if len(pair) >= 50
                    else np.nan
                )
                rows.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "factor": factor,
                        "reference_factor": reference_factor,
                        "stock_count": int(len(pair)),
                        "rank_correlation": (
                            float(correlation)
                            if pd.notna(correlation)
                            else np.nan
                        ),
                    }
                )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.DataFrame()
    summary = (
        daily.groupby(["factor", "reference_factor"])["rank_correlation"]
        .agg(signal_dates="count", mean="mean", median="median", max="max")
        .reset_index()
    )
    summary["mean_absolute_correlation"] = (
        daily.assign(abs_corr=daily["rank_correlation"].abs())
        .groupby(["factor", "reference_factor"])["abs_corr"]
        .mean()
        .to_numpy()
    )
    return daily, summary


def validate_outputs(
    *,
    factors: pd.DataFrame,
    diagnostics: pd.DataFrame,
    aggregate: pd.DataFrame,
    by_fold: pd.DataFrame,
    screening: pd.DataFrame,
    quality_summary: dict,
) -> dict:
    checks = {
        "factor_key_duplicates": int(factors.duplicated(["date", "code"]).sum()),
        "diagnostic_key_duplicates": int(
            diagnostics.duplicated(["date", "factor", "horizon_weeks"]).sum()
        ),
        "future_current_matches": int(quality_summary["future_current_matches"]),
        "future_previous_matches": int(quality_summary["future_previous_matches"]),
        "factor_count": int(screening["factor"].nunique()),
        "expected_factor_count": len(FACTOR_NAMES),
        "aggregate_rows": int(len(aggregate)),
        "fold_rows": int(len(by_fold)),
        "quality_passed": bool(quality_summary["passed"]),
    }
    checks["passed"] = bool(
        checks["factor_key_duplicates"] == 0
        and checks["diagnostic_key_duplicates"] == 0
        and checks["future_current_matches"] == 0
        and checks["future_previous_matches"] == 0
        and checks["factor_count"] == checks["expected_factor_count"]
        and checks["quality_passed"]
    )
    if not checks["passed"]:
        raise ValueError(f"基本面因子验证失败: {checks}")
    return checks


def run_screening(
    *,
    history_start: str = "2020-01-01",
    oos_start: str = "2022-01-01",
    end: str = "2025-05-12",
    cache_dir: Path = DEFAULT_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
    download: bool = True,
    refresh: bool = False,
) -> dict:
    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    config.portfolio.rebalance_frequency = "weekly"
    engine = BacktestEngine(config)
    logger.info("加载行情与历史时点沪深300股票池")
    engine.load_price_data(history_start, end)
    if engine.universe_store is None:
        raise ValueError("基本面因子筛选要求历史时点股票池")
    union = engine.universe_store.union_for_range(history_start, end)
    codes = sorted(union["code"].astype(str).str.zfill(6).unique())
    download_result = {
        "requested": len(codes),
        "downloaded": 0,
        "failed": [],
    }
    if download:
        download_result = update_valuation_cache(
            codes=codes,
            start=history_start,
            end=end,
            cache_dir=cache_dir,
            refresh=refresh,
        )
    valuation_data = load_valuation_data(codes, cache_dir)
    missing_codes = sorted(set(codes) - set(valuation_data))
    if missing_codes:
        raise ValueError(
            f"历史估值缓存缺少{len(missing_codes)}只股票；"
            f"请联网重跑，示例: {missing_codes[:5]}"
        )

    rebalance_dates = _generate_rebalance_dates(
        oos_start, end, "weekly", engine.price_data
    )
    factors = build_fundamental_factors(
        valuation_data=valuation_data,
        rebalance_dates=rebalance_dates,
        universe_store=engine.universe_store,
    )
    quality_by_date, quality_summary = profile_factor_quality(factors)
    enriched = add_forward_returns(factors, engine.price_data)
    diagnostics = calculate_signal_diagnostics(
        enriched,
        factor_names=FACTOR_NAMES,
    )
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
        factors,
        factor_names=FACTOR_NAMES,
    )
    reference_correlation_daily, reference_correlation_summary = (
        calculate_reference_correlations(
            factors,
            PROJECT_ROOT
            / "data"
            / "backtest_results"
            / "factor_screening"
            / "factor_values.csv",
        )
    )
    screening = screen_fundamental_factors(
        aggregate,
        by_fold,
        quality_by_date,
    )
    checks = validate_outputs(
        factors=factors,
        diagnostics=diagnostics,
        aggregate=aggregate,
        by_fold=by_fold,
        screening=screening,
        quality_summary=quality_summary,
    )
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
        "source": (
            "BaoStock历史日度peTTM/pbMRQ/psTTM；"
            "只向后匹配到信号日，不使用未来记录"
        ),
        "factor_construction": {
            "value": "1/PE、1/PB",
            "quality": "PB/PE≈TTM盈利/MRQ净资产；PS/PE≈TTM净利率",
            "profitability_improvement": "质量代理值减去一年前可得值",
            "cross_section": "2.5%/97.5%缩尾后减同行业中位数",
        },
        "screening_rule": (
            "中位覆盖率>=60%；4周平均IC>=0.015、IC为正周占比>=55%、"
            "HAC t>=1.5、Top-Bottom收益差>0、至少5/7窗口IC为正、"
            "8周平均IC>0"
        ),
        "download": download_result,
        "data_quality": quality_summary,
        "validation": checks,
        "screening": _records(screening),
        "promising_factors": promising,
        "decision": (
            "通过者进入月频Top10+缓冲30组合回测"
            if promising
            else "没有因子通过门槛，停止进入组合回测"
        ),
        "research_note": (
            "历史估值的点时性由交易日字段控制；BaoStock是否对历史财务"
            "口径做过事后重述无法从接口单独验证，因此结果为研究级证据。"
        ),
        "reference_correlation_available": bool(
            not reference_correlation_summary.empty
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "factor_values.csv": factors,
        "factor_quality_by_date.csv": quality_by_date,
        "diagnostics_by_date.csv": diagnostics,
        "diagnostics_summary.csv": aggregate,
        "diagnostics_by_fold.csv": by_fold,
        "factor_correlations_by_date.csv": correlation_daily,
        "factor_correlations_summary.csv": correlation_summary,
        "correlations_with_low_volatility_by_date.csv": (
            reference_correlation_daily
        ),
        "correlations_with_low_volatility_summary.csv": (
            reference_correlation_summary
        ),
        "screening_decision.csv": screening,
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
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# 估值、质量与盈利改善因子筛选报告",
        "",
        f"- 样本区间：{oos_start} 至 {end}",
        f"- 周度信号日：{factors['date'].nunique()}个",
        "- 股票池：历史时点沪深300成分",
        "- 收益口径：信号后下一交易日开盘至未来开盘",
        "- 横截面处理：缩尾并减同行业中位数",
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
        "## 数据质量",
        "",
        f"- 日期×股票重复记录：{checks['factor_key_duplicates']}",
        f"- 使用信号日之后估值记录：{checks['future_current_matches']}",
        f"- 一年前比较使用目标日之后记录：{checks['future_previous_matches']}",
        f"- 数据质量门槛：{'通过' if quality_summary['passed'] else '未通过'}",
        "",
        pd.DataFrame(quality_summary["coverage"]).to_markdown(index=False),
        "",
        "## 各期限总体诊断",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## 因子相关性",
        "",
        correlation_summary.to_markdown(index=False),
        "",
        "## 与低波动风险特征的信息重叠",
        "",
        (
            reference_correlation_summary.to_markdown(index=False)
            if not reference_correlation_summary.empty
            else "未找到上一阶段低波动因子明细，未计算。"
        ),
        "",
        result["research_note"],
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    logger.info(f"基本面因子筛选完成: {output_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="估值、质量与盈利改善因子筛选"
    )
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="只使用已有估值缓存",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="忽略覆盖记录并重新下载",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screening(
        history_start=args.history_start,
        oos_start=args.oos_start,
        end=args.end,
        cache_dir=args.cache,
        output_dir=args.output,
        download=not args.no_download,
        refresh=args.refresh,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
