"""Independent factor admission for the v2 SSE Composite research pool."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.factor_reset_screening import benjamini_hochberg
from src.factor_screening import _hac_t_stat
from src.tradability import TradabilityRules, screen_tradable_codes
from src.universe import PointInTimeUniverse
from src.v2_factor_data import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SNAPSHOT_DIR,
    FACTOR_NAMES,
    audit_factor_cache,
    build_factor_protocol,
    discover_universe_union,
    ensure_factor_protocol,
)
from src.v2_monthly_strategy import build_v2_config


HORIZONS = {4: 20, 8: 40, 12: 60}


def calculate_price_factors(
    close: pd.Series, signal_date: pd.Timestamp
) -> dict[str, float]:
    """Calculate frozen price factors using no observations after signal."""
    series = pd.to_numeric(close, errors="coerce").dropna().copy()
    series.index = pd.DatetimeIndex(series.index).normalize()
    history = series[series.index <= pd.Timestamp(signal_date).normalize()]
    returns = history.pct_change().dropna()
    low_volatility = (
        -float(returns.tail(60).std(ddof=1) * np.sqrt(252))
        if len(returns) >= 60 else np.nan
    )
    momentum = (
        float(history.iloc[-21] / history.iloc[-253] - 1)
        if len(history) >= 253
        and history.iloc[-21] > 0
        and history.iloc[-253] > 0
        else np.nan
    )
    return {
        "low_volatility_60": low_volatility,
        "momentum_12_1": momentum,
    }


def latest_positive_pb_asof(
    valuation: pd.DataFrame, signal_date: pd.Timestamp
) -> float:
    """Return the latest known PB, only if that latest value is positive."""
    if valuation.empty or not {"date", "pbMRQ"}.issubset(valuation.columns):
        return np.nan
    local = valuation[["date", "pbMRQ"]].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce")
    local["pbMRQ"] = pd.to_numeric(local["pbMRQ"], errors="coerce")
    local = local[
        local["date"] <= pd.Timestamp(signal_date).normalize()
    ].dropna(subset=["date", "pbMRQ"]).sort_values("date")
    if local.empty:
        return np.nan
    value = float(local.iloc[-1]["pbMRQ"])
    return value if value > 0 else np.nan


def sector_neutralize_book_to_price(frame: pd.DataFrame) -> pd.Series:
    """Winsorize a date cross-section and remove its sector medians."""
    values = pd.to_numeric(frame["book_to_price"], errors="coerce")
    valid = values.dropna()
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.empty:
        return result
    clipped = values.clip(valid.quantile(0.025), valid.quantile(0.975))
    sector_medians = clipped.groupby(frame["sector"]).transform("median")
    result.loc[clipped.notna()] = (
        clipped[clipped.notna()] - sector_medians[clipped.notna()]
    )
    return result


def _indexed_price(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    if "date" in local.columns:
        local["date"] = pd.to_datetime(local["date"], errors="coerce")
        local = local.dropna(subset=["date"]).set_index("date")
    local.index = pd.DatetimeIndex(local.index).normalize()
    return local.sort_index()


def build_factor_frame(
    *,
    price_data: dict[str, pd.DataFrame],
    valuation_data: dict[str, pd.DataFrame],
    signal_dates: pd.DatetimeIndex,
    universe_store: Any,
    tradability_rules: TradabilityRules,
) -> pd.DataFrame:
    """Build all frozen factors after point-in-time tradability filtering."""
    indexed_prices = {
        str(code).zfill(6): _indexed_price(frame)
        for code, frame in price_data.items()
    }
    records: list[dict[str, Any]] = []
    for signal in pd.DatetimeIndex(signal_dates).sort_values():
        signal = pd.Timestamp(signal).normalize()
        snapshot = universe_store.snapshot_for(signal)
        eligible, _details, _audit = screen_tradable_codes(
            snapshot.members,
            indexed_prices,
            signal,
            tradability_rules,
        )
        pool_size = len(eligible)
        current_rows: list[dict[str, Any]] = []
        members = snapshot.members.set_index("code")
        for code in sorted(eligible):
            price = indexed_prices.get(code, pd.DataFrame())
            factors = calculate_price_factors(price["close"], signal)
            pb = latest_positive_pb_asof(
                valuation_data.get(code, pd.DataFrame()), signal
            )
            sector = str(members.loc[code, "sector"])
            current_rows.append({
                "date": signal,
                "code": code,
                "sector": sector,
                "tradable_pool_size": pool_size,
                "low_volatility_60": factors["low_volatility_60"],
                "momentum_12_1": factors["momentum_12_1"],
                "book_to_price": 1.0 / pb if pd.notna(pb) else np.nan,
            })
        if not current_rows:
            continue
        current = pd.DataFrame(current_rows)
        current["sector_neutral_book_to_price_mrq"] = (
            sector_neutralize_book_to_price(current)
        )
        records.extend(current.to_dict("records"))
    columns = [
        "date", "code", "sector", "tradable_pool_size",
        "low_volatility_60", "sector_neutral_book_to_price_mrq",
        "momentum_12_1",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)[columns].sort_values(
        ["date", "code"]
    ).reset_index(drop=True)


def _forward_open_return(
    frame: pd.DataFrame, signal_date: pd.Timestamp, periods: int
) -> float:
    if frame.empty or "open" not in frame.columns:
        return np.nan
    local = _indexed_price(frame)
    entry = int(
        local.index.searchsorted(
            pd.Timestamp(signal_date).normalize(), side="right"
        )
    )
    exit_position = entry + int(periods)
    if entry >= len(local) or exit_position >= len(local):
        return np.nan
    entry_price = pd.to_numeric(
        pd.Series([local.iloc[entry]["open"]]), errors="coerce"
    ).iloc[0]
    exit_price = pd.to_numeric(
        pd.Series([local.iloc[exit_position]["open"]]), errors="coerce"
    ).iloc[0]
    if pd.isna(entry_price) or pd.isna(exit_price):
        return np.nan
    if float(entry_price) <= 0 or float(exit_price) <= 0:
        return np.nan
    return float(exit_price / entry_price - 1)


def add_forward_open_returns(
    factors: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = factors.copy()
    for weeks, sessions in HORIZONS.items():
        result[f"forward_{weeks}w"] = [
            _forward_open_return(
                price_data.get(str(code).zfill(6), pd.DataFrame()),
                pd.Timestamp(date),
                sessions,
            )
            for code, date in zip(result["code"], result["date"])
        ]
    return result


def _quintiles(pair: pd.DataFrame, factor: str) -> pd.Series:
    ranked = pair[factor].rank(method="first")
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def calculate_monthly_diagnostics(
    enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate monthly IC, quintile returns, coverage, and turnover."""
    monthly_rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []

    for date, date_frame in enriched.groupby("date", sort=True):
        pool_size = int(pd.to_numeric(
            date_frame["tradable_pool_size"], errors="coerce"
        ).max())
        for factor in FACTOR_NAMES:
            factor_count = int(date_frame[factor].notna().sum())
            coverage = factor_count / pool_size if pool_size > 0 else 0.0
            for weeks in HORIZONS:
                return_column = f"forward_{weeks}w"
                pair = date_frame[["code", factor, return_column]].dropna()
                stock_count = int(len(pair))
                measurable = (
                    stock_count >= 100
                    and pair[factor].nunique() > 4
                    and pair[return_column].nunique() > 4
                )
                if measurable:
                    rank_ic = pair[factor].corr(
                        pair[return_column], method="spearman"
                    )
                    assigned = pair.assign(
                        quintile=_quintiles(pair, factor).to_numpy()
                    )
                    means = assigned.groupby("quintile")[return_column].mean()
                else:
                    rank_ic = np.nan
                    means = pd.Series(dtype=float)
                top = float(means.get(5, np.nan))
                bottom = float(means.get(1, np.nan))
                monthly_rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "horizon_weeks": int(weeks),
                    "tradable_pool_size": pool_size,
                    "factor_count": factor_count,
                    "forward_pair_count": stock_count,
                    "stock_count": stock_count,
                    "coverage_ratio": float(coverage),
                    "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                    "top_quintile_return": top,
                    "bottom_quintile_return": bottom,
                    "top_bottom_spread": (
                        top - bottom
                        if pd.notna(top) and pd.notna(bottom) else np.nan
                    ),
                })
                for quintile in range(1, 6):
                    quintile_rows.append({
                        "date": str(pd.Timestamp(date).date()),
                        "factor": factor,
                        "horizon_weeks": int(weeks),
                        "quintile": quintile,
                        "mean_forward_return": float(
                            means.get(quintile, np.nan)
                        ),
                    })

    for factor in FACTOR_NAMES:
        previous: set[str] | None = None
        for date, date_frame in enriched.groupby("date", sort=True):
            pair = date_frame[["code", factor]].dropna()
            if len(pair) < 5:
                current: set[str] = set()
            else:
                percentile = pair[factor].rank(pct=True, method="average")
                current = set(
                    pair.loc[percentile >= 0.80, "code"].astype(str)
                )
            if previous is not None and previous:
                turnover_rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor": factor,
                    "top_quintile_turnover": float(
                        1.0 - len(current & previous) / len(previous)
                    ),
                })
            previous = current

    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(quintile_rows),
        pd.DataFrame(
            turnover_rows,
            columns=["date", "factor", "top_quintile_turnover"],
        ),
    )


def summarize_factor_diagnostics(
    monthly: pd.DataFrame,
    quintiles: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    protocol: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Summarize the frozen diagnostics without looking at factor outcomes."""
    diagnostics = monthly.copy()
    diagnostics["date"] = pd.to_datetime(diagnostics["date"])
    quintile_data = quintiles.copy()
    quintile_data["date"] = pd.to_datetime(quintile_data["date"])
    turnover_data = turnover.copy()
    if not turnover_data.empty:
        turnover_data["date"] = pd.to_datetime(turnover_data["date"])

    aggregate_rows: list[dict[str, Any]] = []
    for (factor, horizon), group in diagnostics.groupby(
        ["factor", "horizon_weeks"], sort=False
    ):
        ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spread = pd.to_numeric(
            group["top_bottom_spread"], errors="coerce"
        ).dropna()
        coverage = pd.to_numeric(
            group["coverage_ratio"], errors="coerce"
        ).dropna()
        ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
        q_group = quintile_data[
            (quintile_data["factor"] == factor)
            & (quintile_data["horizon_weeks"] == horizon)
        ]
        q_means = (
            q_group.groupby("quintile")["mean_forward_return"]
            .mean()
            .reindex(range(1, 6))
        )
        if q_means.notna().all() and q_means.nunique() > 1:
            monotonic = float(pd.Series(
                q_means.to_numpy()
            ).corr(pd.Series(range(1, 6)), method="spearman"))
            correct_steps = int(sum(
                right > left
                for left, right in zip(q_means.iloc[:-1], q_means.iloc[1:])
            ))
        else:
            monotonic = np.nan
            correct_steps = 0
        factor_turnover = turnover_data[
            turnover_data.get("factor", pd.Series(dtype=str)) == factor
        ] if not turnover_data.empty else pd.DataFrame()
        mean_turnover = (
            float(pd.to_numeric(
                factor_turnover["top_quintile_turnover"], errors="coerce"
            ).mean())
            if not factor_turnover.empty else np.nan
        )
        mean_spread = float(spread.mean()) if len(spread) else np.nan
        annualized_spread = (
            float(((1 + mean_spread) ** 12 - 1) * 100)
            if pd.notna(mean_spread) and mean_spread > -1 else np.nan
        )
        aggregate_rows.append({
            "factor": factor,
            "horizon_weeks": int(horizon),
            "signal_dates": int(len(ic)),
            "mean_rank_ic": float(ic.mean()) if len(ic) else np.nan,
            "median_rank_ic": float(ic.median()) if len(ic) else np.nan,
            "positive_ic_ratio": float((ic > 0).mean()) if len(ic) else 0.0,
            "hac_t_stat": _hac_t_stat(
                ic, max_lag=max(int(horizon) // 4 - 1, 0)
            ),
            "annualized_icir": (
                float(ic.mean() / ic_std * np.sqrt(12))
                if pd.notna(ic_std) and ic_std > 0 else 0.0
            ),
            "mean_top_bottom_spread": mean_spread,
            "annualized_gross_spread_pct": annualized_spread,
            "aggregate_quintile_monotonic_corr": monotonic,
            "correct_adjacent_quintile_steps": correct_steps,
            "median_coverage_ratio": (
                float(coverage.median()) if len(coverage) else 0.0
            ),
            "mean_top_quintile_turnover": mean_turnover,
        })
    aggregate = pd.DataFrame(aggregate_rows)

    by_year = (
        diagnostics.assign(year=diagnostics["date"].dt.year)
        .groupby(["factor", "horizon_weeks", "year"], as_index=False)
        .agg(
            signal_dates=("rank_ic", "count"),
            mean_rank_ic=("rank_ic", "mean"),
            median_coverage_ratio=("coverage_ratio", "median"),
        )
    )

    fold_frames: list[pd.DataFrame] = []
    for definition in protocol["fixed_windows"]:
        mask = (
            (diagnostics["date"] >= pd.Timestamp(definition["start"]))
            & (diagnostics["date"] <= pd.Timestamp(definition["end"]))
        )
        current = diagnostics.loc[mask].copy()
        if current.empty:
            continue
        current["fold"] = int(definition["id"])
        fold_frames.append(current)
    folded = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames else pd.DataFrame(columns=list(diagnostics.columns) + ["fold"])
    )
    by_fold = (
        folded.groupby(
            ["factor", "horizon_weeks", "fold"], as_index=False
        ).agg(
            signal_dates=("rank_ic", "count"),
            mean_rank_ic=("rank_ic", "mean"),
            median_coverage_ratio=("coverage_ratio", "median"),
        )
        if not folded.empty
        else pd.DataFrame(columns=[
            "factor", "horizon_weeks", "fold", "signal_dates",
            "mean_rank_ic", "median_coverage_ratio",
        ])
    )
    return {
        "aggregate": aggregate,
        "by_year": by_year,
        "by_fold": by_fold,
    }


def evaluate_factor_gates(
    summaries: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    """Apply every frozen factor-admission gate exactly once."""
    aggregate = summaries["aggregate"].copy()
    by_year = summaries["by_year"].copy()
    by_fold = summaries["by_fold"].copy()
    primary = aggregate[aggregate["horizon_weeks"] == 4].copy()
    primary["one_sided_p_value"] = norm.sf(primary["hac_t_stat"])
    primary["fdr_q_value"] = benjamini_hochberg(
        primary["one_sided_p_value"]
    )
    gates_config = protocol["gates"]
    rows: list[dict[str, Any]] = []
    for factor in FACTOR_NAMES:
        primary_match = primary[primary["factor"] == factor]
        eight_match = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 8)
        ]
        if primary_match.empty or eight_match.empty:
            rows.append({
                "factor": factor,
                "failed_gates": "missing_required_summary",
                "status": "reject",
            })
            continue
        row4 = primary_match.iloc[0]
        row8 = eight_match.iloc[0]
        years = by_year[
            (by_year["factor"] == factor)
            & (by_year["horizon_weeks"] == 4)
            & by_year["year"].isin([2021, 2022, 2023, 2024])
        ]
        folds = by_fold[
            (by_fold["factor"] == factor)
            & (by_fold["horizon_weeks"] == 4)
            & by_fold["fold"].isin(range(1, 8))
        ]
        positive_years = int((years["mean_rank_ic"] > 0).sum())
        positive_folds = int((folds["mean_rank_ic"] > 0).sum())
        full_year_coverage = bool(
            len(years) == 4
            and (years["median_coverage_ratio"] >=
                 gates_config["minimum_full_year_median_coverage_ratio"]).all()
        )
        checks = {
            "median_coverage": float(row4["median_coverage_ratio"])
            >= gates_config["median_coverage_ratio"],
            "full_year_coverage": full_year_coverage,
            "mean_ic": float(row4["mean_rank_ic"])
            >= gates_config["four_week_mean_rank_ic"],
            "median_ic": float(row4["median_rank_ic"]) > 0,
            "positive_ic_ratio": float(row4["positive_ic_ratio"])
            >= gates_config["four_week_positive_ic_ratio"],
            "hac_t": float(row4["hac_t_stat"])
            >= gates_config["four_week_hac_t_stat"],
            "raw_p": float(row4["one_sided_p_value"])
            <= gates_config["four_week_one_sided_p_value"],
            "fdr": float(row4["fdr_q_value"])
            <= gates_config["four_week_fdr_q_value"],
            "gross_spread": float(row4["annualized_gross_spread_pct"])
            >= gates_config["annualized_gross_top_bottom_spread_pct"],
            "quintile_monotonicity": float(
                row4["aggregate_quintile_monotonic_corr"]
            ) >= gates_config["aggregate_quintile_monotonic_corr"],
            "adjacent_steps": int(row4["correct_adjacent_quintile_steps"])
            >= gates_config["correct_adjacent_quintile_steps"],
            "icir": float(row4["annualized_icir"])
            >= gates_config["annualized_icir"],
            "positive_full_years": (
                len(years) == 4
                and positive_years >= gates_config["positive_full_years_of_four"]
            ),
            "positive_fixed_windows": (
                len(folds) == 7
                and positive_folds
                >= gates_config["positive_fixed_windows_of_seven"]
            ),
            "eight_week_ic": float(row8["mean_rank_ic"]) > 0,
            "turnover": float(row4["mean_top_quintile_turnover"])
            <= gates_config["max_monthly_top_quintile_turnover"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        output = {
            "factor": factor,
            "four_week_mean_rank_ic": float(row4["mean_rank_ic"]),
            "four_week_median_rank_ic": float(row4["median_rank_ic"]),
            "four_week_positive_ic_ratio": float(row4["positive_ic_ratio"]),
            "four_week_hac_t_stat": float(row4["hac_t_stat"]),
            "four_week_one_sided_p_value": float(
                row4["one_sided_p_value"]
            ),
            "four_week_fdr_q_value": float(row4["fdr_q_value"]),
            "positive_full_years": positive_years,
            "positive_fixed_windows": positive_folds,
            "failed_gates": ",".join(failed),
            "status": "advance" if not failed else "reject",
        }
        output.update({f"gate_{key}": bool(value) for key, value in checks.items()})
        rows.append(output)
    return pd.DataFrame(rows)


def calculate_factor_correlations(factors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, current in factors.groupby("date"):
        correlations = current[FACTOR_NAMES].corr(method="spearman")
        for index, left in enumerate(FACTOR_NAMES):
            for right in FACTOR_NAMES[index + 1:]:
                value = correlations.loc[left, right]
                rows.append({
                    "date": str(pd.Timestamp(date).date()),
                    "factor_left": left,
                    "factor_right": right,
                    "rank_correlation": (
                        float(value) if pd.notna(value) else np.nan
                    ),
                })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(columns=[
            "factor_left", "factor_right", "signal_dates",
            "mean_rank_correlation", "mean_absolute_correlation",
        ])
    return (
        daily.assign(abs_correlation=daily["rank_correlation"].abs())
        .groupby(["factor_left", "factor_right"], as_index=False)
        .agg(
            signal_dates=("rank_correlation", "count"),
            mean_rank_correlation=("rank_correlation", "mean"),
            mean_absolute_correlation=("abs_correlation", "mean"),
        )
    )


def _safe_output_dir(output_dir: Path) -> Path:
    output = Path(output_dir).resolve()
    v1 = (
        Path(__file__).resolve().parent.parent / "data" / "paper_trading"
    ).resolve()
    if output == v1 or v1 in output.parents:
        raise ValueError("v2因子研究禁止写入v1正式模拟盘目录")
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _render_report(
    *,
    summary: dict[str, Any],
    decisions: pd.DataFrame | None = None,
) -> str:
    status_labels = {
        "data_audit_failed": "数据审计未通过",
        "factor_failed": "三个因子均未通过",
        "factor_passed": "至少一个因子通过独立准入",
    }
    rows = ""
    if decisions is not None and not decisions.empty:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(row.factor))}</td>"
            f"<td>{'进入组合验证' if row.status == 'advance' else '淘汰'}</td>"
            f"<td>{html.escape(str(row.failed_gates) or '无')}</td>"
            "</tr>"
            for row in decisions.itertuples(index=False)
        )
    else:
        rows = (
            "<tr><td colspan='3'>数据未达到筛选条件，未计算因子结论。</td></tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>v2上证综指独立因子筛选</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:980px;margin:32px auto;padding:0 20px;color:#1f2937}}table{{border-collapse:collapse;width:100%;table-layout:fixed}}td,th{{border:1px solid #d1d5db;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th:nth-child(1){{width:29%}}th:nth-child(2){{width:18%}}th:nth-child(3){{width:53%}}.warn{{background:#fff7ed;padding:14px;border-left:4px solid #f97316}}</style>
</head><body><h1>v2 上证综指独立因子筛选</h1>
<p class="warn">本报告只决定因子能否进入组合回测，不构成买入建议、模拟盘或实盘授权，也不修改v1。</p>
<p><b>状态：</b>{html.escape(status_labels.get(summary['status'], summary['status']))}</p>
<p><b>股票池并集：</b>{summary.get('universe_union_codes', 0)}只；<b>历史快照：</b>{summary.get('snapshot_count', 0)}期。</p>
<table><thead><tr><th>因子</th><th>结论</th><th>未通过门槛</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def _load_valid_cache(
    audit: pd.DataFrame,
    cache_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prices: dict[str, pd.DataFrame] = {}
    valuations: dict[str, pd.DataFrame] = {}
    for code in audit.loc[audit["status"] == "complete", "code"]:
        normalized = str(code).zfill(6)
        price_path = Path(cache_root) / "prices" / f"price_{normalized}.csv"
        valuation_path = (
            Path(cache_root) / "valuation" / f"valuation_{normalized}.csv"
        )
        price = pd.read_csv(price_path, parse_dates=["date"])
        valuation = pd.read_csv(valuation_path, parse_dates=["date"])
        prices[normalized] = price.set_index("date").sort_index()
        valuations[normalized] = valuation.sort_values("date")
    return prices, valuations


def _tradability_rules() -> TradabilityRules:
    config = build_v2_config()
    return TradabilityRules(
        min_average_amount=50_000_000.0,
        amount_lookback=20,
        top_n=config.portfolio.top_n,
        initial_cash=config.execution.initial_cash,
        min_lot=config.execution.min_lot,
        commission=config.execution.commission,
        min_commission=config.execution.min_commission,
        transfer_fee_rate=config.execution.transfer_fee_rate,
        slippage_bps=config.execution.slippage_bps,
    )


def run_v2_factor_screening(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    cache_root: Path,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the fail-closed end-to-end factor admission workflow."""
    output = _safe_output_dir(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _protocol_path, protocol_hash = ensure_factor_protocol(output)
    protocol = build_factor_protocol()
    snapshot_dir = Path(snapshot_dir)
    cache_root = Path(cache_root)
    try:
        codes = discover_universe_union(snapshot_dir)
    except (ValueError, OSError) as error:
        summary = {
            "strategy_id": "v2_monthly_top10_dd10",
            "stage_id": "v2_independent_factor_screening",
            "deployment_authorized": False,
            "protocol_sha256": protocol_hash,
            "research_start": start,
            "research_end": end,
            "status": "data_audit_failed",
            "error": str(error),
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "data_audit_summary.json", {
            "infrastructure_ready": False,
            "ready_for_screening": False,
            "error": str(error),
        })
        (output / "report.html").write_text(
            _render_report(summary=summary), encoding="utf-8"
        )
        return summary
    snapshot_paths = sorted(snapshot_dir.glob("*.csv"))
    data_audit, cache_summary = audit_factor_cache(
        codes=codes,
        start=protocol["cache_window"]["start"],
        end=protocol["cache_window"]["end"],
        price_dir=cache_root / "prices",
        valuation_dir=cache_root / "valuation",
        audit_dir=cache_root / "audit",
    )
    data_audit.to_csv(
        output / "data_audit.csv", index=False, encoding="utf-8-sig"
    )
    infrastructure_ready = bool(
        len(codes) == protocol["universe"]["expected_union_codes"]
        and len(snapshot_paths) == protocol["universe"]["expected_snapshots"]
        and cache_summary["ready_for_screening"]
    )
    audit_summary = {
        **cache_summary,
        "actual_union_codes": len(codes),
        "expected_union_codes": protocol["universe"]["expected_union_codes"],
        "snapshot_count": len(snapshot_paths),
        "expected_snapshots": protocol["universe"]["expected_snapshots"],
        "infrastructure_ready": infrastructure_ready,
    }
    _write_json(output / "data_audit_summary.json", audit_summary)
    base_summary: dict[str, Any] = {
        "strategy_id": "v2_monthly_top10_dd10",
        "stage_id": "v2_independent_factor_screening",
        "deployment_authorized": False,
        "protocol_sha256": protocol_hash,
        "research_start": start,
        "research_end": end,
        "universe_union_codes": len(codes),
        "snapshot_count": len(snapshot_paths),
    }
    if not infrastructure_ready:
        summary = {**base_summary, "status": "data_audit_failed"}
        _write_json(output / "summary.json", summary)
        (output / "report.html").write_text(
            _render_report(summary=summary), encoding="utf-8"
        )
        return summary

    price_data, valuation_data = _load_valid_cache(data_audit, cache_root)
    signal_dates = pd.DatetimeIndex([
        pd.Timestamp(path.stem)
        for path in snapshot_paths
        if pd.Timestamp(start) <= pd.Timestamp(path.stem) <= pd.Timestamp(end)
    ])
    universe_store = PointInTimeUniverse(snapshot_dir)
    factors = build_factor_frame(
        price_data=price_data,
        valuation_data=valuation_data,
        signal_dates=signal_dates,
        universe_store=universe_store,
        tradability_rules=_tradability_rules(),
    )
    pools = (
        factors.groupby("date")["tradable_pool_size"].max()
        if not factors.empty else pd.Series(dtype=float)
    )
    pool_ready = bool(
        len(pools) == len(signal_dates)
        and len(pools) > 0
        and (pools >= protocol["universe"]["minimum_cross_section"]).all()
    )
    if not pool_ready:
        audit_summary["tradable_pool_ready"] = False
        audit_summary["valid_signal_dates"] = int(len(pools))
        audit_summary["expected_signal_dates"] = int(len(signal_dates))
        audit_summary["minimum_tradable_pool"] = (
            int(pools.min()) if len(pools) else 0
        )
        _write_json(output / "data_audit_summary.json", audit_summary)
        summary = {**base_summary, "status": "data_audit_failed"}
        _write_json(output / "summary.json", summary)
        (output / "report.html").write_text(
            _render_report(summary=summary), encoding="utf-8"
        )
        return summary

    enriched = add_forward_open_returns(factors, price_data)
    monthly, quintiles, turnover = calculate_monthly_diagnostics(enriched)
    summaries = summarize_factor_diagnostics(
        monthly, quintiles, turnover, protocol=protocol
    )
    decisions = evaluate_factor_gates(summaries, protocol)
    correlations = calculate_factor_correlations(factors)
    factors.to_csv(output / "factor_values.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(
        output / "monthly_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    quintiles.to_csv(
        output / "quintile_returns.csv", index=False, encoding="utf-8-sig"
    )
    turnover.to_csv(output / "turnover.csv", index=False, encoding="utf-8-sig")
    summaries["aggregate"].to_csv(
        output / "aggregate_summary.csv", index=False, encoding="utf-8-sig"
    )
    summaries["by_year"].to_csv(
        output / "year_summary.csv", index=False, encoding="utf-8-sig"
    )
    summaries["by_fold"].to_csv(
        output / "fold_summary.csv", index=False, encoding="utf-8-sig"
    )
    correlations.to_csv(
        output / "factor_correlations.csv", index=False, encoding="utf-8-sig"
    )
    decisions.to_csv(
        output / "screening_decision.csv", index=False, encoding="utf-8-sig"
    )
    advanced = decisions.loc[
        decisions["status"] == "advance", "factor"
    ].tolist()
    summary = {
        **base_summary,
        "status": "factor_passed" if advanced else "factor_failed",
        "signal_dates": int(factors["date"].nunique()),
        "factor_rows": int(len(factors)),
        "advanced_factors": advanced,
    }
    _write_json(output / "summary.json", summary)
    (output / "report.html").write_text(
        _render_report(summary=summary, decisions=decisions), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="运行v2上证综指三个独立因子的冻结准入筛选"
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_v2_factor_screening(
        start=args.start,
        end=args.end,
        cache_root=Path(args.cache_root),
        snapshot_dir=Path(args.snapshot_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"v2独立因子筛选状态：{summary['status']}")
    if summary.get("advanced_factors"):
        print("通过因子：" + "、".join(summary["advanced_factors"]))
    print(f"报告：{Path(args.output_dir) / 'report.html'}")
    return 2 if summary["status"] == "data_audit_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
