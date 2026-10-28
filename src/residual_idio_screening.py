"""低特质波动相对旧低波动的残差增量检验。

这是 factor_reset_screening 之后单独预登记的跟进实验。它不改变窗口，
也不重新选择方向；只检验低特质波动剔除低波动与行业暴露后是否仍有
独立解释力。
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

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig
from src.factor_reset_screening import (
    FULL_GATE_YEARS,
    HORIZONS,
    _quintile_statistics,
    add_forward_returns,
    audit_price_provenance,
    summarize_diagnostics,
)
from src.price_feed import load_index_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "factor_reset_screening"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "residual_idio_screening"
)
FACTOR = "residual_low_idiosyncratic_volatility_vs_lowvol"
RESEARCH_END = "2025-05-12"


def _hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_protocol() -> dict:
    protocol = {
        "protocol_version": "residual-idio-v1",
        "frozen_on": "2026-07-31",
        "parent_protocol_hash": (
            "381f9251684e4984b72b07258d223336c6211286f6560cc48dca12a962c8b1db"
        ),
        "history_start": "2020-01-01",
        "research_start": "2021-01-01",
        "research_end": RESEARCH_END,
        "future_holdout_start": "2026-08-03",
        "factor": FACTOR,
        "formula": (
            "每个信号日将行业中性低特质波动秩，对行业中性60日低波动秩"
            "及行业哑变量做截面OLS，取残差；正值表示相对旧低波动仍更低的"
            "特质波动。"
        ),
        "direction": "higher_is_better",
        "no_parameter_search": True,
        "primary_horizon_weeks": 4,
        "gates": {
            "coverage": 0.95,
            "mean_rank_ic": 0.020,
            "median_rank_ic": 0.0,
            "positive_ic_ratio": 0.55,
            "hac_t_stat": 2.50,
            "one_sided_p_value": 0.01,
            "annualized_icir": 0.50,
            "annualized_gross_spread_pct": 4.0,
            "quintile_monotonic_corr": 0.80,
            "correct_adjacent_steps": 3,
            "positive_full_year_ratio": 0.75,
            "leave_one_year_out_min_ic": 0.012,
            "up_and_down_regime_ic_positive": True,
            "eight_week_ic_positive": True,
            "abs_rank_correlation_to_lowvol": 0.70,
            "monthly_top_quintile_turnover": 0.70,
        },
        "decision_boundary": (
            "全部通过才可进入组合级验证；失败后不得调窗口、翻方向或降低门槛。"
        ),
    }
    protocol["protocol_hash"] = _hash(protocol)
    return protocol


def build_residual_factor(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按日期对低波动秩与行业哑变量残差化。"""
    rows: list[pd.DataFrame] = []
    audits: list[dict] = []
    for date, date_frame in base.groupby("date"):
        current = date_frame[
            [
                "date",
                "code",
                "sector",
                "low_idiosyncratic_volatility_120",
                "low_volatility_60_reference",
            ]
        ].dropna().copy()
        y = current["low_idiosyncratic_volatility_120"].rank(pct=True)
        lowvol_rank = current["low_volatility_60_reference"].rank(pct=True)
        sector_dummies = pd.get_dummies(
            current["sector"].fillna("未知"),
            prefix="sector",
            drop_first=True,
            dtype=float,
        )
        design = pd.concat(
            [
                pd.Series(1.0, index=current.index, name="intercept"),
                lowvol_rank.rename("lowvol_rank"),
                sector_dummies,
            ],
            axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(
            design.to_numpy(float), y.to_numpy(float), rcond=None
        )
        fitted = design.to_numpy(float) @ coefficients
        current[FACTOR] = y.to_numpy(float) - fitted
        rows.append(current[["date", "code", "sector", FACTOR]])
        audits.append({
            "date": str(pd.Timestamp(date).date()),
            "stock_count": int(len(current)),
            "r_squared": float(
                1
                - np.square(y.to_numpy(float) - fitted).sum()
                / np.square(y.to_numpy(float) - y.mean()).sum()
            ),
            "pearson_corr_with_lowvol_rank": float(
                np.corrcoef(current[FACTOR], lowvol_rank)[0, 1]
            ),
            "spearman_corr_with_lowvol": float(
                current[FACTOR].corr(lowvol_rank, method="spearman")
            ),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def calculate_diagnostics(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for date, date_frame in enriched.groupby("date"):
        for weeks in HORIZONS:
            return_col = f"forward_{weeks}w_pct"
            pair = date_frame[[FACTOR, return_col]].dropna()
            measurable = (
                len(pair) >= 80
                and pair[FACTOR].nunique() >= 10
                and pair[return_col].nunique() >= 10
            )
            if measurable:
                ic = pair[FACTOR].corr(pair[return_col], method="spearman")
                top, bottom, spread, monotonic, quintiles = (
                    _quintile_statistics(pair, FACTOR, return_col)
                )
            else:
                ic = top = bottom = spread = monotonic = np.nan
                quintiles = [np.nan] * 5
            rows.append({
                "date": str(pd.Timestamp(date).date()),
                "factor": FACTOR,
                "horizon_weeks": weeks,
                "stock_count": int(len(pair)),
                "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                "top_quintile_return_pct": top,
                "bottom_quintile_return_pct": bottom,
                "top_bottom_spread_pct": spread,
                "quintile_monotonic_corr": monotonic,
                **{
                    f"quintile_{number}_return_pct": value
                    for number, value in enumerate(quintiles, start=1)
                },
            })
    return pd.DataFrame(rows)


def calculate_turnover(factor_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous: set[str] | None = None
    for date, frame in factor_values.groupby("date", sort=True):
        percentile = frame[FACTOR].rank(pct=True, method="average")
        current = set(frame.loc[percentile >= 0.80, "code"])
        if previous:
            rows.append({
                "date": str(pd.Timestamp(date).date()),
                "top_quintile_turnover": (
                    1 - len(current & previous) / len(previous)
                ),
            })
        previous = current
    return pd.DataFrame(rows)


def evaluate(
    *,
    aggregate: pd.DataFrame,
    by_year: pd.DataFrame,
    by_regime: pd.DataFrame,
    factor_values: pd.DataFrame,
    audit: pd.DataFrame,
    turnover: pd.DataFrame,
) -> dict:
    row4 = aggregate[aggregate["horizon_weeks"] == 4].iloc[0]
    row8 = aggregate[aggregate["horizon_weeks"] == 8].iloc[0]
    years = by_year[
        (by_year["horizon_weeks"] == 4)
        & by_year["year"].isin(FULL_GATE_YEARS)
    ]
    positive_year_ratio = float(
        (years["mean_rank_ic"] > 0).mean()
    ) if len(years) == 4 else 0.0
    leave_one_out = []
    if len(years) == 4:
        for omitted in FULL_GATE_YEARS:
            remaining = years[years["year"] != omitted]
            leave_one_out.append(
                float(
                    np.average(
                        remaining["mean_rank_ic"],
                        weights=remaining["signal_dates"],
                    )
                )
            )
    min_leave_one_out = min(leave_one_out) if leave_one_out else np.nan
    regimes = by_regime[by_regime["horizon_weeks"] == 4]
    regime_map = dict(zip(regimes["regime"], regimes["mean_rank_ic"]))
    coverage = float(
        factor_values.groupby("date")[FACTOR]
        .apply(lambda values: values.notna().mean())
        .median()
    )
    p_value = float(norm.sf(float(row4["hac_t_stat"])))
    annualized_spread = float(
        (
            (1 + float(row4["mean_top_bottom_spread_pct"]) / 100) ** 12
            - 1
        )
        * 100
    )
    mean_abs_lowvol_corr = float(
        audit["spearman_corr_with_lowvol"].abs().mean()
    )
    mean_turnover = float(turnover["top_quintile_turnover"].mean())
    gates = {
        "coverage": coverage >= 0.95,
        "mean_ic": float(row4["mean_rank_ic"]) >= 0.020,
        "median_ic": float(row4["median_rank_ic"]) > 0,
        "positive_ic": float(row4["positive_ic_ratio"]) >= 0.55,
        "hac_t": float(row4["hac_t_stat"]) >= 2.50,
        "raw_p": p_value <= 0.01,
        "icir": float(row4["annualized_icir"]) >= 0.50,
        "gross_spread": annualized_spread >= 4.0,
        "monotonicity": float(
            row4["aggregate_quintile_monotonic_corr"]
        ) >= 0.80,
        "adjacent_steps": int(
            row4["correct_adjacent_quintile_steps"]
        ) >= 3,
        "full_years": positive_year_ratio >= 0.75,
        "leave_one_year_out": (
            pd.notna(min_leave_one_out) and min_leave_one_out >= 0.012
        ),
        "market_regimes": (
            set(regime_map) == {"up", "down"}
            and all(float(value) > 0 for value in regime_map.values())
        ),
        "eight_week_ic": float(row8["mean_rank_ic"]) > 0,
        "independence": mean_abs_lowvol_corr <= 0.70,
        "turnover": mean_turnover <= 0.70,
    }
    return {
        "factor": FACTOR,
        "coverage": coverage,
        "four_week_mean_rank_ic": float(row4["mean_rank_ic"]),
        "four_week_median_rank_ic": float(row4["median_rank_ic"]),
        "four_week_positive_ic_ratio": float(row4["positive_ic_ratio"]),
        "four_week_hac_t_stat": float(row4["hac_t_stat"]),
        "four_week_one_sided_p_value": p_value,
        "annualized_icir": float(row4["annualized_icir"]),
        "annualized_gross_spread_pct": annualized_spread,
        "aggregate_quintile_monotonic_corr": float(
            row4["aggregate_quintile_monotonic_corr"]
        ),
        "correct_adjacent_quintile_steps": int(
            row4["correct_adjacent_quintile_steps"]
        ),
        "positive_full_year_ratio": positive_year_ratio,
        "leave_one_year_out_min_ic": min_leave_one_out,
        "up_regime_mean_ic": float(regime_map.get("up", np.nan)),
        "down_regime_mean_ic": float(regime_map.get("down", np.nan)),
        "eight_week_mean_rank_ic": float(row8["mean_rank_ic"]),
        "mean_abs_rank_correlation_to_lowvol": mean_abs_lowvol_corr,
        "mean_monthly_top_quintile_turnover": mean_turnover,
        "mean_residualization_r_squared": float(audit["r_squared"].mean()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "status": "advance" if all(gates.values()) else "reject",
    }


def build_report(protocol: dict, result: dict, output: Path) -> None:
    passed = result["status"] == "advance"
    headline = (
        "低特质波动的独立部分，通过历史准入门槛"
        if passed
        else "低特质波动的独立部分，仍不足以晋级"
    )
    verdict = (
        "通过历史准入门槛，仅进入组合级验证"
        if passed
        else "未通过准入门槛，停止该因子"
    )
    failed = "、".join(result["failed_gates"]) or "无"
    gate_line = (
        '<span style="color:#16805a;font-weight:700">未通过：无</span>'
        if passed
        else f'<span class="bad">未通过：{failed}</span>'
    )
    metrics = [
        ("4周 Rank IC", f"{result['four_week_mean_rank_ic']:.3f}", "≥ 0.020"),
        ("HAC t", f"{result['four_week_hac_t_stat']:.2f}", "≥ 2.50"),
        ("单侧 p", f"{result['four_week_one_sided_p_value']:.4f}", "≤ 0.010"),
        ("年化多空毛差", f"{result['annualized_gross_spread_pct']:.2f}%", "≥ 4%"),
        ("五分位单调性", f"{result['aggregate_quintile_monotonic_corr']:.2f}", "≥ 0.80"),
        ("与旧低波动相关", f"{result['mean_abs_rank_correlation_to_lowvol']:.2f}", "≤ 0.70"),
        ("月度Top组换手", f"{result['mean_monthly_top_quintile_turnover']:.1%}", "≤ 70%"),
        ("正年度比例", f"{result['positive_full_year_ratio']:.0%}", "≥ 75%"),
    ]
    rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td><td>{gate}</td></tr>"
        for label, value, gate in metrics
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>低特质波动残差增量检验</title><style>
body{{margin:0;background:#f4f6f7;color:#17212b;font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}
main{{max-width:980px;margin:auto;padding:42px 24px 70px}}header{{padding:32px;border-radius:18px;background:#16324a;color:white}}
h1{{font-size:32px;margin:6px 0 12px}}.eyebrow{{font-size:12px;letter-spacing:.12em;opacity:.75}}
.verdict{{margin:20px 0;padding:18px 22px;background:white;border-left:5px solid {"#16805a" if passed else "#b42318"};border-radius:10px}}
section{{background:white;border:1px solid #dde4e8;border-radius:14px;padding:22px;margin:16px 0}}
h2{{font-size:21px;margin:0 0 12px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:11px;border-bottom:1px solid #e8edef;text-align:left}}
th{{background:#eef3f5}}code{{word-break:break-all}}.muted{{color:#69747e}}.bad{{color:#b42318;font-weight:700}}
@media(max-width:600px){{main{{padding:16px 10px}}header{{padding:24px}}h1{{font-size:26px}}}}
</style></head><body><main><header><div class="eyebrow">QUANT RESEARCH · FOLLOW-UP BATCH</div>
<h1>{headline}</h1>
<p>固定120日窗口与正方向，仅剔除旧60日低波动及行业暴露；没有重新调参。</p></header>
<div class="verdict"><b>{verdict}</b><br>{gate_line}</div>
<section><h2>核心结果</h2><table><thead><tr><th>指标</th><th>结果</th><th>硬门槛</th></tr></thead>
<tbody>{rows}</tbody></table></section>
<section><h2>研究解释</h2>
<p>残差化平均解释度为 <b>{result['mean_residualization_r_squared']:.1%}</b>，
说明原始低特质波动相当一部分确实与旧低波动重合。剔除后必须重新满足全部
IC、年度、市场状态、分组形态和换手门槛，不能继承原因子的显著性。</p>
<p class="muted">该结果属于历史研究取舍，不是个股推荐或实盘授权。真正的新未见样本从
2026-08-03开始积累。</p></section>
<section><h2>冻结记录</h2><p><code>{protocol['protocol_hash']}</code></p>
<p>{protocol['formula']}</p></section>
</main></body></html>"""
    output.write_text(html, encoding="utf-8")


def run(
    *,
    base_output: Path = BASE_OUTPUT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    protocol = build_protocol()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base = pd.read_csv(
        base_output / "factor_values.csv",
        dtype={"code": str},
        parse_dates=["date"],
    )
    base["code"] = base["code"].str.zfill(6)
    factors, residual_audit = build_residual_factor(base)

    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    engine = BacktestEngine(config)
    engine.load_price_data("2020-01-01", RESEARCH_END)
    provenance = audit_price_provenance(codes=list(engine.price_data))
    if not provenance["passed"]:
        raise ValueError("价格缓存来源/复权审计失败")
    enriched = add_forward_returns(factors, engine.price_data)
    diagnostics = calculate_diagnostics(enriched)
    diagnostics["year"] = pd.to_datetime(diagnostics["date"]).dt.year
    index_data = load_index_data("000300", "2020-01-01", RESEARCH_END)
    trend = index_data["close"].astype(float).pct_change(252)
    diagnostics["regime"] = diagnostics["date"].map(
        {
            str(pd.Timestamp(date).date()): (
                "up" if pd.notna(value) and value >= 0 else "down"
            )
            for date, value in trend.items()
        }
    )
    aggregate = summarize_diagnostics(
        diagnostics, ["factor", "horizon_weeks"]
    )
    by_year = summarize_diagnostics(
        diagnostics, ["year", "factor", "horizon_weeks"]
    )
    by_regime = summarize_diagnostics(
        diagnostics, ["regime", "factor", "horizon_weeks"]
    )
    turnover = calculate_turnover(factors)
    decision = evaluate(
        aggregate=aggregate,
        by_year=by_year,
        by_regime=by_regime,
        factor_values=factors,
        audit=residual_audit,
        turnover=turnover,
    )
    validation = {
        "passed": bool(
            not factors.duplicated(["date", "code"]).any()
            and factors["date"].max() <= pd.Timestamp(RESEARCH_END)
            and abs(
                residual_audit["pearson_corr_with_lowvol_rank"]
            ).max() < 1e-10
            and protocol["protocol_hash"]
            == _hash({k: v for k, v in protocol.items() if k != "protocol_hash"})
        ),
        "factor_rows": int(len(factors)),
        "signal_dates": int(factors["date"].nunique()),
        "max_abs_pearson_corr_with_lowvol_rank": float(
            residual_audit["pearson_corr_with_lowvol_rank"].abs().max()
        ),
    }
    if not validation["passed"]:
        raise AssertionError(f"残差因子验证失败: {validation}")
    summary = {
        "protocol_hash": protocol["protocol_hash"],
        "data_audit": {
            "price_provenance": provenance,
            "research_end": RESEARCH_END,
            "post_cutoff_data_used": False,
        },
        "decision": decision,
        "validation": validation,
        "formal_strategy_changed": False,
    }
    for name, frame in {
        "factor_values.csv": factors,
        "residualization_audit.csv": residual_audit,
        "diagnostics_by_date.csv": diagnostics,
        "diagnostics_summary.csv": aggregate,
        "diagnostics_by_year.csv": by_year,
        "diagnostics_by_regime.csv": by_regime,
        "top_quintile_turnover.csv": turnover,
    }.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_report(protocol, decision, output_dir / "report.html")
    logger.info(f"低特质波动残差筛选完成: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="低特质波动残差增量检验")
    parser.add_argument("--base-output", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(base_output=args.base_output, output_dir=args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
