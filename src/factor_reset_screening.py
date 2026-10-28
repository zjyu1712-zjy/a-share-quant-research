"""策略重建：第二代独立价格因子筛选。

冻结候选定义后，只在研究截止日以前的历史时点沪深300成分股中检验：

- 低市场 Beta；
- 低特质波动率；
- 低单日极端收益（彩票偏好规避）；
- 12-1 月动量；
- 距离52周高点（跳过最近一个月）；
- 一个月短期反转。

本模块只决定候选因子是否值得进入组合级验证，不修改正式策略。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.config import StrategyConfig
from src.factor_screening import _hac_t_stat, _records
from src.price_feed import load_index_data
from src.walk_forward_attribution import _open_forward_return


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "factor_reset_screening"
)
RESEARCH_END = "2025-05-12"
HORIZONS = {4: 20, 8: 40, 12: 60}
FACTOR_SPECS = {
    "low_beta_120": {
        "label": "低Beta（120日）",
        "rationale": "规避对市场方向依赖过强的股票，寻找更稳健的横截面收益。",
        "first_rejection": "若仅复制低波动暴露，或在上涨市场IC转负，则不具独立价值。",
    },
    "low_idiosyncratic_volatility_120": {
        "label": "低特质波动（120日）",
        "rationale": "规避无法由市场解释的高噪声波动和投机性定价。",
        "first_rejection": "若与旧低波动因子高度相关，则不作为新增因子。",
    },
    "low_max_daily_return_20": {
        "label": "低极端单日收益（20日）",
        "rationale": "规避近期依赖单日暴涨、具有彩票偏好特征的股票。",
        "first_rejection": "若排名轮换过快或分组收益不单调，则难以低成本执行。",
    },
    "momentum_12_1": {
        "label": "12-1月动量",
        "rationale": "检验更长周期、跳过最近一个月后的趋势延续。",
        "first_rejection": "若完整年度或下跌市场方向不稳定，则不进入组合。",
    },
    "high_52week_skip_1m": {
        "label": "52周高点接近度（跳过1月）",
        "rationale": "用价格锚定检验中期趋势，而不直接重复短周期涨幅。",
        "first_rejection": "若与12-1月动量高度重合且无更稳定IC，则淘汰。",
    },
    "short_term_reversal_20": {
        "label": "一个月短期反转",
        "rationale": "检验短期过度反应后的均值回归。",
        "first_rejection": "若月度排名换手过高，理论收益可能无法覆盖交易摩擦。",
    },
}
FACTOR_NAMES = list(FACTOR_SPECS)
REFERENCE_FACTOR = "low_volatility_60_reference"
FULL_GATE_YEARS = [2021, 2022, 2023, 2024]


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol(
    *,
    history_start: str,
    research_start: str,
    research_end: str,
) -> dict:
    """返回在读取结果前固定的研究协议。"""
    protocol = {
        "protocol_version": "factor-reset-v1",
        "frozen_on": "2026-07-31",
        "history_start": history_start,
        "research_start": research_start,
        "research_end": research_end,
        "future_holdout_start": "2026-08-03",
        "universe": "point-in-time CSI 300 constituents",
        "signal_frequency": "monthly",
        "execution_return": "next-session open to future open",
        "candidate_directions": {
            name: FACTOR_SPECS[name]["label"] for name in FACTOR_NAMES
        },
        "primary_horizon_weeks": 4,
        "secondary_horizons_weeks": [8, 12],
        "multiple_testing": "one-sided HAC p-values with Benjamini-Hochberg FDR",
        "gates": {
            "median_coverage_ratio": 0.95,
            "four_week_mean_rank_ic": 0.020,
            "four_week_median_rank_ic": 0.0,
            "four_week_positive_ic_ratio": 0.55,
            "four_week_hac_t_stat": 2.50,
            "four_week_raw_p_value": 0.01,
            "four_week_fdr_q_value": 0.10,
            "annualized_gross_top_bottom_spread_pct": 4.0,
            "aggregate_quintile_monotonic_corr": 0.80,
            "correct_adjacent_quintile_steps": 3,
            "annualized_icir": 0.50,
            "positive_full_year_ratio": 0.75,
            "leave_one_full_year_out_min_mean_ic": 0.012,
            "positive_market_regimes": ["up", "down"],
            "eight_week_mean_rank_ic": 0.0,
            "max_abs_correlation_to_low_volatility": 0.75,
            "max_monthly_top_quintile_turnover": 0.70,
        },
        "quarantine": (
            "2025-05-13至2026-07-30的既有冻结审计结果已经被观察，"
            "不得再次称为未见样本；本轮不读取该区间价格。"
        ),
        "advancement": (
            "通过全部门槛仅表示可进入月频Top10+缓冲30组合级验证，"
            "不构成买入建议或实盘授权。"
        ),
    }
    protocol["protocol_hash"] = _canonical_hash(protocol)
    return protocol


def audit_price_provenance(
    *,
    codes: list[str],
    cache_dir: Path = PROJECT_ROOT / "data" / "price_cache",
) -> dict:
    """要求每只研究股票都具备可追溯的前复权缓存元数据。"""
    accepted_sources = {
        "akshare_stock_zh_a_hist_tx",
        "akshare_stock_zh_a_hist_eastmoney",
        "akshare_stock_zh_a_daily_sina",
    }
    failures: list[dict] = []
    sources: dict[str, int] = {}
    for code in sorted({str(value).zfill(6) for value in codes}):
        meta_path = cache_dir / f"stock_{code}.coverage.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append({"code": code, "reason": "missing_or_invalid_metadata"})
            continue
        source = str(meta.get("source", ""))
        reasons = []
        if int(meta.get("schema_version", 0)) < 2:
            reasons.append("legacy_schema")
        if source not in accepted_sources:
            reasons.append("unknown_source")
        if meta.get("adjust") != "qfq":
            reasons.append("not_qfq")
        if reasons:
            failures.append({"code": code, "reason": ",".join(reasons)})
        else:
            sources[source] = sources.get(source, 0) + 1
    return {
        "requested_codes": len(set(codes)),
        "passed_codes": len(set(codes)) - len(failures),
        "source_counts": sources,
        "failures": failures,
        "passed": not failures,
    }


def _winsorize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    if len(clean) < 20:
        return numeric
    return numeric.clip(clean.quantile(0.025), clean.quantile(0.975))


def _at(series: pd.Series, date: pd.Timestamp) -> float:
    value = series.get(date, np.nan)
    return float(value) if pd.notna(value) else np.nan


def build_reset_factors(
    *,
    price_data: dict[str, pd.DataFrame],
    index_data: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> pd.DataFrame:
    """只用信号日及以前的数据构造预先声明的候选因子。"""
    benchmark_return = (
        index_data["close"].sort_index().astype(float).pct_change()
    )
    records: list[dict] = []
    for code, frame in price_data.items():
        if frame.empty or "close" not in frame:
            continue
        close = frame["close"].sort_index().astype(float)
        stock_return = close.pct_change()
        market_return = benchmark_return.reindex(stock_return.index)
        market_variance = market_return.rolling(120, min_periods=100).var()
        beta = (
            stock_return.rolling(120, min_periods=100).cov(market_return)
            / market_variance.replace(0, np.nan)
        )
        residual_return = stock_return - beta * market_return
        idiosyncratic_volatility = (
            residual_return.rolling(120, min_periods=100).std()
            * np.sqrt(252)
        )
        shifted_close = close.shift(20)
        raw = {
            "low_beta_120": -beta,
            "low_idiosyncratic_volatility_120": -idiosyncratic_volatility,
            "low_max_daily_return_20": -stock_return.rolling(
                20, min_periods=20
            ).max(),
            "momentum_12_1": shifted_close / close.shift(252) - 1,
            "high_52week_skip_1m": (
                shifted_close
                / shifted_close.rolling(252, min_periods=252).max()
            ),
            "short_term_reversal_20": -(close / close.shift(20) - 1),
            REFERENCE_FACTOR: -(
                stock_return.rolling(60, min_periods=60).std() * np.sqrt(252)
            ),
        }
        for date in rebalance_dates:
            timestamp = pd.Timestamp(date)
            if timestamp not in close.index:
                continue
            records.append({
                "date": timestamp,
                "code": str(code).zfill(6),
                **{name: _at(values, timestamp) for name, values in raw.items()},
            })

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    normalized: list[pd.DataFrame] = []
    factor_columns = FACTOR_NAMES + [REFERENCE_FACTOR]
    for date, date_frame in frame.groupby("date"):
        snapshot = universe_store.snapshot_for(date)
        current = date_frame[
            date_frame["code"].isin(snapshot.codes)
        ].copy()
        current["sector"] = current["code"].map(snapshot.code_to_sector)
        for factor in factor_columns:
            winsorized = _winsorize(current[factor])
            current[factor] = (
                winsorized
                - winsorized.groupby(current["sector"]).transform("median")
            )
        normalized.append(current)
    result = pd.concat(normalized, ignore_index=True)
    logger.info(
        f"第二代因子: {len(result)}条，{result['code'].nunique()}只股票，"
        f"{result['date'].nunique()}个信号日"
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


def _quintile_statistics(pair: pd.DataFrame, factor: str, return_col: str) -> tuple:
    percentile = pair[factor].rank(pct=True, method="average")
    quintile = np.ceil(percentile * 5).clip(1, 5).astype(int)
    means = pair.assign(quintile=quintile).groupby("quintile")[return_col].mean()
    top = float(means.get(5, np.nan))
    bottom = float(means.get(1, np.nan))
    if len(means) == 5 and means.nunique() > 1:
        monotonic = float(
            pd.Series(means.to_numpy()).corr(
                pd.Series(np.arange(1, 6)), method="spearman"
            )
        )
    else:
        monotonic = np.nan
    quintile_values = [
        float(means.get(quintile_number, np.nan))
        for quintile_number in range(1, 6)
    ]
    return top, bottom, top - bottom, monotonic, quintile_values


def calculate_diagnostics(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for date, date_frame in enriched.groupby("date"):
        for factor in FACTOR_NAMES:
            for weeks in HORIZONS:
                return_col = f"forward_{weeks}w_pct"
                pair = date_frame[[factor, return_col]].dropna()
                measurable = (
                    len(pair) >= 80
                    and pair[factor].nunique() >= 10
                    and pair[return_col].nunique() >= 10
                )
                if measurable:
                    ic = pair[factor].corr(pair[return_col], method="spearman")
                    (
                        top,
                        bottom,
                        spread,
                        monotonic,
                        quintile_values,
                    ) = _quintile_statistics(
                        pair, factor, return_col
                    )
                else:
                    ic = top = bottom = spread = monotonic = np.nan
                    quintile_values = [np.nan] * 5
                rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "horizon_weeks": weeks,
                    "stock_count": int(len(pair)),
                    "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                    "top_quintile_return_pct": top,
                    "bottom_quintile_return_pct": bottom,
                    "top_bottom_spread_pct": spread,
                    "quintile_monotonic_corr": monotonic,
                    **{
                        f"quintile_{number}_return_pct": value
                        for number, value in enumerate(
                            quintile_values, start=1
                        )
                    },
                })
    return pd.DataFrame(rows)


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
        monotonic = group["quintile_monotonic_corr"].dropna()
        mean_quintiles = [
            float(group[f"quintile_{number}_return_pct"].mean())
            for number in range(1, 6)
        ]
        if all(pd.notna(mean_quintiles)) and len(set(mean_quintiles)) > 1:
            aggregate_monotonic = float(
                pd.Series(mean_quintiles).corr(
                    pd.Series(np.arange(1, 6)), method="spearman"
                )
            )
            correct_steps = int(
                sum(
                    right > left
                    for left, right in zip(
                        mean_quintiles[:-1], mean_quintiles[1:]
                    )
                )
            )
        else:
            aggregate_monotonic = np.nan
            correct_steps = 0
        horizon = int(group["horizon_weeks"].iloc[0])
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
        row = dict(zip(group_columns, keys))
        row.update({
            "signal_dates": int(len(ic)),
            "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
            "median_rank_ic": float(ic.median()) if len(ic) else np.nan,
            "rank_ic_std": ic_std,
            "annualized_icir": (
                float(ic.mean() / ic_std * np.sqrt(12))
                if pd.notna(ic_std) and ic_std > 0
                else 0.0
            ),
            "positive_ic_ratio": float((ic > 0).mean()) if len(ic) else np.nan,
            "hac_t_stat": _hac_t_stat(ic, max_lag=max(horizon // 4 - 1, 0)),
            "mean_top_bottom_spread_pct": (
                float(spread.mean()) if len(spread) else np.nan
            ),
            "positive_spread_ratio": (
                float((spread > 0).mean()) if len(spread) else np.nan
            ),
            "mean_quintile_monotonic_corr": (
                float(monotonic.mean()) if len(monotonic) else np.nan
            ),
            "aggregate_quintile_monotonic_corr": aggregate_monotonic,
            "correct_adjacent_quintile_steps": correct_steps,
            **{
                f"mean_quintile_{number}_return_pct": value
                for number, value in enumerate(mean_quintiles, start=1)
            },
        })
        rows.append(row)
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR校正，保持输入索引。"""
    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(float)
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def calculate_correlations(
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = FACTOR_NAMES + [REFERENCE_FACTOR]
    rows: list[dict] = []
    for date, date_frame in factors.groupby("date"):
        correlation = date_frame[columns].corr(method="spearman")
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1:]:
                value = correlation.loc[left, right]
                rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor_left": left,
                    "factor_right": right,
                    "rank_correlation": (
                        float(value) if pd.notna(value) else np.nan
                    ),
                })
    daily = pd.DataFrame(rows)
    summary = (
        daily.assign(abs_correlation=daily["rank_correlation"].abs())
        .groupby(["factor_left", "factor_right"])
        .agg(
            signal_dates=("rank_correlation", "count"),
            mean_rank_correlation=("rank_correlation", "mean"),
            mean_absolute_correlation=("abs_correlation", "mean"),
        )
        .reset_index()
    )
    return daily, summary


def calculate_top_quintile_turnover(factors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for factor in FACTOR_NAMES:
        previous: set[str] | None = None
        turnovers: list[float] = []
        for date, date_frame in factors.groupby("date", sort=True):
            pair = date_frame[["code", factor]].dropna()
            percentile = pair[factor].rank(pct=True, method="average")
            current = set(pair.loc[percentile >= 0.80, "code"])
            if previous:
                turnover = 1.0 - len(current & previous) / len(previous)
                turnovers.append(turnover)
                rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "top_quintile_turnover": turnover,
                })
            previous = current
        if not turnovers:
            rows.append({
                "date": None,
                "factor": factor,
                "top_quintile_turnover": np.nan,
            })
    return pd.DataFrame(rows)


def _reference_correlation(
    correlations: pd.DataFrame,
    factor: str,
) -> float:
    pair = correlations[
        (
            (correlations["factor_left"] == factor)
            & (correlations["factor_right"] == REFERENCE_FACTOR)
        )
        | (
            (correlations["factor_right"] == factor)
            & (correlations["factor_left"] == REFERENCE_FACTOR)
        )
    ]
    if pair.empty:
        return np.nan
    return float(pair["mean_absolute_correlation"].iloc[0])


def screen_candidates(
    *,
    aggregate: pd.DataFrame,
    by_year: pd.DataFrame,
    by_regime: pd.DataFrame,
    coverage: pd.DataFrame,
    correlations: pd.DataFrame,
    turnover: pd.DataFrame,
) -> pd.DataFrame:
    primary = aggregate[aggregate["horizon_weeks"] == 4].copy()
    primary["one_sided_p_value"] = norm.sf(primary["hac_t_stat"])
    primary["fdr_q_value"] = benjamini_hochberg(
        primary["one_sided_p_value"]
    )
    rows: list[dict] = []
    for factor in FACTOR_NAMES:
        row4 = primary[primary["factor"] == factor].iloc[0]
        row8 = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 8)
        ].iloc[0]
        years = by_year[
            (by_year["factor"] == factor)
            & (by_year["horizon_weeks"] == 4)
            & (by_year["year"].isin(FULL_GATE_YEARS))
        ]
        regimes = by_regime[
            (by_regime["factor"] == factor)
            & (by_regime["horizon_weeks"] == 4)
            & (by_regime["regime"].isin(["up", "down"]))
        ]
        positive_year_ratio = float(
            (years["mean_rank_ic"] > 0).mean()
        ) if len(years) == len(FULL_GATE_YEARS) else 0.0
        leave_one_year_means = []
        if len(years) == len(FULL_GATE_YEARS):
            for omitted_year in FULL_GATE_YEARS:
                remaining = years[years["year"] != omitted_year]
                weights = remaining["signal_dates"].astype(float)
                leave_one_year_means.append(
                    float(
                        np.average(
                            remaining["mean_rank_ic"].astype(float),
                            weights=weights,
                        )
                    )
                )
        leave_one_year_min_ic = (
            min(leave_one_year_means) if leave_one_year_means else np.nan
        )
        regime_map = dict(zip(regimes["regime"], regimes["mean_rank_ic"]))
        both_regimes_positive = bool(
            set(regime_map) == {"up", "down"}
            and all(float(value) > 0 for value in regime_map.values())
        )
        factor_coverage = float(
            coverage.loc[coverage["factor"] == factor, "median_coverage_ratio"].iloc[0]
        )
        reference_corr = _reference_correlation(correlations, factor)
        factor_turnover = float(
            turnover.loc[
                turnover["factor"] == factor, "top_quintile_turnover"
            ].mean()
        )
        annualized_gross_spread = float(
            (
                (1 + float(row4["mean_top_bottom_spread_pct"]) / 100) ** 12
                - 1
            )
            * 100
        )
        gates = {
            "coverage": factor_coverage >= 0.95,
            "mean_ic": float(row4["mean_rank_ic"]) >= 0.020,
            "median_ic": float(row4["median_rank_ic"]) > 0,
            "positive_ic": float(row4["positive_ic_ratio"]) >= 0.55,
            "hac_t": float(row4["hac_t_stat"]) >= 2.50,
            "raw_p": float(row4["one_sided_p_value"]) <= 0.01,
            "fdr": float(row4["fdr_q_value"]) <= 0.10,
            "gross_spread": annualized_gross_spread >= 4.0,
            "monotonicity": float(
                row4["aggregate_quintile_monotonic_corr"]
            ) >= 0.80,
            "adjacent_steps": int(
                row4["correct_adjacent_quintile_steps"]
            ) >= 3,
            "icir": float(row4["annualized_icir"]) >= 0.50,
            "full_years": positive_year_ratio >= 0.75,
            "leave_one_year_out": (
                pd.notna(leave_one_year_min_ic)
                and leave_one_year_min_ic >= 0.012
            ),
            "market_regimes": both_regimes_positive,
            "eight_week_ic": float(row8["mean_rank_ic"]) > 0,
            "independence": (
                pd.notna(reference_corr) and reference_corr <= 0.75
            ),
            "turnover": factor_turnover <= 0.70,
        }
        failed = [name for name, passed in gates.items() if not passed]
        rows.append({
            "factor": factor,
            "label": FACTOR_SPECS[factor]["label"],
            "median_coverage_ratio": factor_coverage,
            "four_week_mean_rank_ic": float(row4["mean_rank_ic"]),
            "four_week_median_rank_ic": float(row4["median_rank_ic"]),
            "four_week_positive_ic_ratio": float(row4["positive_ic_ratio"]),
            "four_week_hac_t_stat": float(row4["hac_t_stat"]),
            "four_week_raw_p_value": float(row4["one_sided_p_value"]),
            "four_week_fdr_q_value": float(row4["fdr_q_value"]),
            "four_week_mean_spread_pct": float(
                row4["mean_top_bottom_spread_pct"]
            ),
            "annualized_gross_spread_pct": annualized_gross_spread,
            "four_week_aggregate_monotonic_corr": float(
                row4["aggregate_quintile_monotonic_corr"]
            ),
            "four_week_correct_adjacent_steps": int(
                row4["correct_adjacent_quintile_steps"]
            ),
            "annualized_icir": float(row4["annualized_icir"]),
            "positive_full_year_ratio": positive_year_ratio,
            "leave_one_full_year_out_min_mean_ic": leave_one_year_min_ic,
            "up_regime_mean_ic": float(regime_map.get("up", np.nan)),
            "down_regime_mean_ic": float(regime_map.get("down", np.nan)),
            "eight_week_mean_rank_ic": float(row8["mean_rank_ic"]),
            "abs_correlation_to_low_volatility": reference_corr,
            "mean_monthly_top_quintile_turnover": factor_turnover,
            "failed_gates": ",".join(failed),
            "status": "advance" if not failed else "reject",
        })
    return pd.DataFrame(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def build_html_report(
    *,
    protocol: dict,
    screening: pd.DataFrame,
    aggregate: pd.DataFrame,
    by_year: pd.DataFrame,
    by_regime: pd.DataFrame,
    output_path: Path,
) -> None:
    advanced = screening[screening["status"] == "advance"]
    headline = (
        f"{len(advanced)}个因子通过全部准入门槛"
        if len(advanced)
        else "没有因子通过全部准入门槛"
    )
    decision = (
        "通过者仅进入组合级验证，不进入实盘。"
        if len(advanced)
        else "停止本批候选，不进行组合回测，也不修改正式策略。"
    )
    board_rows = []
    for row in screening.itertuples(index=False):
        status_label = "进入组合验证" if row.status == "advance" else "淘汰"
        failed = row.failed_gates.replace(",", "、") or "无"
        board_rows.append(
            f"""
            <tr>
              <td><strong>{row.label}</strong><br><span class="muted">{row.factor}</span></td>
              <td class="num">{_fmt(row.four_week_mean_rank_ic)}</td>
              <td class="num">{_fmt(row.four_week_hac_t_stat, 2)}</td>
              <td class="num">{_fmt(row.four_week_fdr_q_value, 3)}</td>
              <td class="num">{_fmt(row.four_week_mean_spread_pct, 2)}%</td>
              <td class="num">{_fmt(row.up_regime_mean_ic)} / {_fmt(row.down_regime_mean_ic)}</td>
              <td class="num">{_fmt(row.abs_correlation_to_low_volatility, 2)}</td>
              <td class="num">{_fmt(row.mean_monthly_top_quintile_turnover * 100, 1)}%</td>
              <td><span class="pill {row.status}">{status_label}</span><br><span class="small">未过：{failed}</span></td>
            </tr>
            """
        )
    cards = []
    for row in screening.itertuples(index=False):
        cards.append(
            f"""
            <article class="idea-card">
              <div class="card-head"><h3>{row.label}</h3><span class="pill {row.status}">{
                  "验证候选" if row.status == "advance" else "停止"
              }</span></div>
              <p><b>经济假设：</b>{FACTOR_SPECS[row.factor]["rationale"]}</p>
              <p><b>第一否决条件：</b>{FACTOR_SPECS[row.factor]["first_rejection"]}</p>
              <p><b>本轮结论：</b>{
                  "全部门槛通过；下一流程为真实成本组合级验证。"
                  if row.status == "advance"
                  else "未通过：" + row.failed_gates.replace(",", "、") + "。"
              }</p>
            </article>
            """
        )
    primary = aggregate[aggregate["horizon_weeks"] == 4].copy()
    year_primary = by_year[by_year["horizon_weeks"] == 4].copy()
    year_pivot = year_primary.pivot(
        index="factor", columns="year", values="mean_rank_ic"
    ).reindex(FACTOR_NAMES)
    year_header = "".join(f"<th>{int(year)}</th>" for year in year_pivot.columns)
    year_rows = []
    for factor, values in year_pivot.iterrows():
        cells = "".join(
            f'<td class="num {"positive" if value > 0 else "negative"}">{_fmt(value)}</td>'
            if pd.notna(value) else '<td class="num">—</td>'
            for value in values
        )
        year_rows.append(
            f"<tr><td>{FACTOR_SPECS[factor]['label']}</td>{cells}</tr>"
        )
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>策略重建｜独立因子准入审计</title>
<style>
:root{{--ink:#17202a;--muted:#68737d;--line:#dfe5e8;--paper:#f5f7f8;--navy:#17324d;
--green:#18794e;--red:#b42318;--amber:#9a6700;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:40px 28px 72px}}
.hero{{background:linear-gradient(135deg,#10283f,#244d68);color:white;padding:34px 38px;border-radius:18px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.76}}
h1{{font-size:34px;line-height:1.18;margin:8px 0 14px}} h2{{font-size:23px;margin:36px 0 14px}}
h3{{margin:0;font-size:17px}} .hero p{{max-width:880px;margin:0;color:#dce7ee}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 0}}
.tile{{background:rgba(255,255,255,.1);padding:14px;border-radius:12px}}
.tile b{{display:block;font-size:23px}} .tile span{{font-size:12px;color:#dce7ee}}
.callout{{background:white;border-left:5px solid var(--red);padding:18px 22px;margin:20px 0;border-radius:10px;
box-shadow:0 4px 16px rgba(20,39,55,.06)}} .callout b{{color:var(--red)}}
.panel{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;margin:14px 0;
box-shadow:0 3px 12px rgba(20,39,55,.04)}} .table-wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;min-width:1050px;font-size:13px}}
th{{text-align:left;background:#eef2f4;color:#41515d;padding:10px;border-bottom:1px solid var(--line)}}
td{{padding:10px;border-bottom:1px solid #edf0f2;vertical-align:top}} .num{{font-variant-numeric:tabular-nums;text-align:right}}
.muted,.small{{color:var(--muted)}} .small{{font-size:11px}} .pill{{display:inline-block;border-radius:999px;
padding:3px 9px;font-size:11px;font-weight:700}} .pill.advance{{background:#dcfce7;color:var(--green)}}
.pill.reject{{background:#fee4e2;color:var(--red)}} .card-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
.idea-card{{background:white;border:1px solid var(--line);border-radius:14px;padding:18px}}
.card-head{{display:flex;justify-content:space-between;gap:12px;align-items:center}}
.idea-card p{{margin:10px 0 0}} .positive{{color:var(--green)}} .negative{{color:var(--red)}}
.foot{{color:var(--muted);font-size:12px;margin-top:28px;border-top:1px solid var(--line);padding-top:16px}}
@media(max-width:820px){{main{{padding:18px 12px}}.hero{{padding:24px}}.tiles,.card-grid{{grid-template-columns:1fr 1fr}}
h1{{font-size:28px}}}} @media(max-width:520px){{.tiles,.card-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<section class="hero">
  <div class="eyebrow">Quant Research Reset · 2026-07-31</div>
  <h1>{headline}</h1>
  <p>这是因子研究优先级审计，不是股票推荐。候选方向在结果计算前冻结，只使用
  {protocol["history_start"]}至{protocol["research_end"]}数据；已看过的后续区间被隔离。</p>
  <div class="tiles">
    <div class="tile"><b>{len(screening)}</b><span>预声明候选</span></div>
    <div class="tile"><b>{len(advanced)}</b><span>通过全部门槛</span></div>
    <div class="tile"><b>{len(primary)}</b><span>完成主期限检验</span></div>
    <div class="tile"><b>2026-08-03</b><span>新未来样本起点</span></div>
  </div>
</section>
<div class="callout"><b>决策：</b>{decision} 2025-05-13之后的数据已被观察，不能再次当作真正未见样本。</div>

<h2>候选漏斗与硬门槛</h2>
<section class="panel table-wrap"><table>
<thead><tr><th>候选因子</th><th>4周IC</th><th>HAC t</th><th>FDR q</th><th>Top-Bottom</th>
<th>上涨/下跌IC</th><th>与低波动相关</th><th>月度Top组换手</th><th>结论</th></tr></thead>
<tbody>{"".join(board_rows)}</tbody></table></section>

<h2>完整年度稳定性（4周Rank IC）</h2>
<section class="panel table-wrap"><table style="min-width:720px">
<thead><tr><th>因子</th>{year_header}</tr></thead><tbody>{"".join(year_rows)}</tbody></table></section>

<h2>候选研究卡</h2>
<section class="card-grid">{"".join(cards)}</section>

<h2>研究纪律</h2>
<section class="panel">
<p><b>因子准入：</b>覆盖率、Rank IC、正IC比例、HAC显著性、FDR、多空分组差、
五分位单调性、完整年度、上涨/下跌市场、8周期限、与旧低波动独立性及换手必须全部通过。</p>
<p><b>证据边界：</b>行情来自本地缓存的前复权OHLCV，股票池使用历史时点沪深300快照；
行业中性化采用信号日股票池行业中位数。该证据可用于研究取舍，不足以证明未来盈利。</p>
<p><b>协议哈希：</b><code>{protocol["protocol_hash"]}</code></p>
</section>
<div class="foot">支持明细：screening_decision.csv、diagnostics_by_date.csv、
diagnostics_by_year.csv、diagnostics_by_regime.csv、factor_correlations_summary.csv、
top_quintile_turnover.csv、validation_checks.json。</div>
</main></body></html>"""
    output_path.write_text(report, encoding="utf-8")


def run_screening(
    *,
    history_start: str = "2020-01-01",
    research_start: str = "2021-01-01",
    research_end: str = RESEARCH_END,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    if pd.Timestamp(research_end) > pd.Timestamp(RESEARCH_END):
        raise ValueError(
            f"策略重建研究不得读取已观察的后续区间；research_end<={RESEARCH_END}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(
        history_start=history_start,
        research_start=research_start,
        research_end=research_end,
    )
    (output_dir / "research_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    engine = BacktestEngine(config)
    engine.load_price_data(history_start, research_end)
    if engine.universe_store is None:
        raise ValueError("策略重建要求历史时点股票池")
    index_data = load_index_data("000300", history_start, research_end)
    provenance = audit_price_provenance(codes=list(engine.price_data))
    if not provenance["passed"]:
        raise ValueError(
            "价格缓存来源/复权审计失败；拒绝运行长期价格因子。"
            f"失败股票数={len(provenance['failures'])}"
        )
    rebalance_dates = _generate_rebalance_dates(
        research_start, research_end, "monthly", engine.price_data
    )
    factors = build_reset_factors(
        price_data=engine.price_data,
        index_data=index_data,
        rebalance_dates=rebalance_dates,
        universe_store=engine.universe_store,
    )
    enriched = add_forward_returns(factors, engine.price_data)
    diagnostics = calculate_diagnostics(enriched)
    diagnostics["year"] = pd.to_datetime(diagnostics["date"]).dt.year
    market_trend = (
        index_data["close"].astype(float).pct_change(60)
        .reindex(pd.to_datetime(diagnostics["date"]).unique())
    )
    regime_map = {
        str(pd.Timestamp(date).date()): (
            "up" if pd.notna(value) and value >= 0 else "down"
        )
        for date, value in market_trend.items()
    }
    diagnostics["regime"] = diagnostics["date"].map(regime_map)

    aggregate = summarize_diagnostics(
        diagnostics, ["factor", "horizon_weeks"]
    )
    by_year = summarize_diagnostics(
        diagnostics, ["year", "factor", "horizon_weeks"]
    )
    by_regime = summarize_diagnostics(
        diagnostics, ["regime", "factor", "horizon_weeks"]
    )
    coverage_rows = []
    for factor in FACTOR_NAMES:
        ratios = factors.groupby("date")[factor].apply(
            lambda values: float(values.notna().mean())
        )
        coverage_rows.append({
            "factor": factor,
            "mean_coverage_ratio": float(ratios.mean()),
            "median_coverage_ratio": float(ratios.median()),
            "minimum_coverage_ratio": float(ratios.min()),
        })
    coverage = pd.DataFrame(coverage_rows)
    correlation_daily, correlation_summary = calculate_correlations(factors)
    turnover = calculate_top_quintile_turnover(factors)
    screening = screen_candidates(
        aggregate=aggregate,
        by_year=by_year,
        by_regime=by_regime,
        coverage=coverage,
        correlations=correlation_summary,
        turnover=turnover,
    )
    advanced = screening.loc[
        screening["status"] == "advance", "factor"
    ].tolist()

    max_price_date = max(
        (frame.index.max() for frame in engine.price_data.values() if not frame.empty),
        default=pd.NaT,
    )
    checks = [
        {
            "check": "candidate_set_matches_frozen_protocol",
            "passed": set(screening["factor"]) == set(FACTOR_NAMES),
            "detail": f"actual={len(screening)}, expected={len(FACTOR_NAMES)}",
        },
        {
            "check": "no_post_cutoff_price_rows",
            "passed": pd.Timestamp(max_price_date) <= pd.Timestamp(research_end),
            "detail": f"max_price_date={pd.Timestamp(max_price_date).date()}",
        },
        {
            "check": "unique_factor_grain",
            "passed": not factors.duplicated(["date", "code"]).any(),
            "detail": f"duplicates={int(factors.duplicated(['date', 'code']).sum())}",
        },
        {
            "check": "all_primary_diagnostics_present",
            "passed": (
                set(
                    aggregate.loc[
                        aggregate["horizon_weeks"] == 4, "factor"
                    ]
                )
                == set(FACTOR_NAMES)
            ),
            "detail": f"primary_factors={int((aggregate['horizon_weeks'] == 4).sum())}",
        },
        {
            "check": "reference_correlations_present",
            "passed": screening[
                "abs_correlation_to_low_volatility"
            ].notna().all(),
            "detail": (
                f"non_null={int(screening['abs_correlation_to_low_volatility'].notna().sum())}"
            ),
        },
        {
            "check": "protocol_hash_recomputes",
            "passed": protocol["protocol_hash"] == _canonical_hash(
                {k: v for k, v in protocol.items() if k != "protocol_hash"}
            ),
            "detail": protocol["protocol_hash"],
        },
    ]
    for item in checks:
        item["passed"] = bool(item["passed"])
    validation = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "checks": checks,
    }
    if validation["status"] != "passed":
        raise AssertionError(f"因子重建输出验证失败: {checks}")

    summary = {
        "period": {
            "history_start": history_start,
            "research_start": research_start,
            "research_end": research_end,
            "signal_dates": int(factors["date"].nunique()),
        },
        "protocol_hash": protocol["protocol_hash"],
        "data_audit": {
            "loaded_stocks": len(engine.price_data),
            "factor_stocks": int(factors["code"].nunique()),
            "factor_rows": int(len(factors)),
            "max_price_date": str(pd.Timestamp(max_price_date).date()),
            "index_max_date": str(pd.Timestamp(index_data.index.max()).date()),
            "price_provenance": provenance,
        },
        "screening": _records(screening),
        "advanced_factors": advanced,
        "decision": (
            "通过者进入月频Top10+缓冲30组合级验证；正式策略保持不变"
            if advanced
            else "没有候选通过全部门槛；停止本批组合回测，正式策略保持不变"
        ),
        "future_holdout_start": protocol["future_holdout_start"],
        "validation": validation,
    }

    outputs = {
        "factor_values.csv": factors,
        "diagnostics_by_date.csv": diagnostics,
        "diagnostics_summary.csv": aggregate,
        "diagnostics_by_year.csv": by_year,
        "diagnostics_by_regime.csv": by_regime,
        "factor_coverage.csv": coverage,
        "factor_correlations_by_date.csv": correlation_daily,
        "factor_correlations_summary.csv": correlation_summary,
        "top_quintile_turnover.csv": turnover,
        "screening_decision.csv": screening,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    (output_dir / "validation_checks.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_html_report(
        protocol=protocol,
        screening=screening,
        aggregate=aggregate,
        by_year=by_year,
        by_regime=by_regime,
        output_path=output_dir / "report.html",
    )
    logger.info(f"策略重建因子筛选完成: {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="策略重建独立因子筛选")
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--research-start", default="2021-01-01")
    parser.add_argument("--research-end", default=RESEARCH_END)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screening(
        history_start=args.history_start,
        research_start=args.research_start,
        research_end=args.research_end,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
