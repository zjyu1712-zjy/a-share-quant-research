"""Explain why the admitted v2 low-volatility factor failed in a portfolio.

The analysis is an order-dependent, additive ablation chain.  It is research
only and must never write to the v1 paper-trading directory.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.indicators import select_with_rank_buffer
from src.low_volatility_portfolio_validation import _attach_shared_data
from src.price_feed import load_index_data
from src.v2_factor_data import DEFAULT_CACHE_ROOT, DEFAULT_SNAPSHOT_DIR
from src.v2_factor_portfolio_validation import (
    DEFAULT_FACTOR_OUTPUT,
    V1_PAPER_DIR,
    V2FactorPortfolioBacktestEngine,
    _benchmark_return,
    _close_frame,
    _load_price_cache,
    _shared_engine,
    _validate_source,
    build_candidate_factor_frames,
)
from src.v2_monthly_strategy import build_v2_config
from src.walk_forward_validation import Fold, generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORTFOLIO_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "v2_factor_portfolio_validation"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "v2_lowvol_failure_attribution"
)

VARIANTS: dict[str, dict[str, Any]] = {
    "raw_lowvol": {
        "label": "原始低波动Top 10（无约束、无费用）",
        "component": "raw_factor_portfolio",
        "sector_enabled": False,
        "rank_buffer": 0,
        "drawdown_overlay": False,
        "frictionless": True,
    },
    "add_sector_constraint": {
        "label": "加入行业限制",
        "component": "sector_constraint",
        "sector_enabled": True,
        "rank_buffer": 0,
        "drawdown_overlay": False,
        "frictionless": True,
    },
    "add_rank_buffer": {
        "label": "再加入排名缓冲30",
        "component": "rank_buffer",
        "sector_enabled": True,
        "rank_buffer": 30,
        "drawdown_overlay": False,
        "frictionless": True,
    },
    "add_drawdown_overlay": {
        "label": "再加入回撤仓位控制",
        "component": "drawdown_overlay",
        "sector_enabled": True,
        "rank_buffer": 30,
        "drawdown_overlay": True,
        "frictionless": True,
    },
    "full_framework": {
        "label": "最后加入真实交易费用",
        "component": "trading_frictions",
        "sector_enabled": True,
        "rank_buffer": 30,
        "drawdown_overlay": True,
        "frictionless": False,
    },
}
CAUSE_LABELS = {
    "factor_translation": "因子到集中组合的转化",
    "sector_constraint": "行业限制",
    "rank_buffer": "排名缓冲",
    "drawdown_overlay": "回撤仓位控制",
    "trading_frictions": "交易费用",
}


def _safe_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == V1_PAPER_DIR or V1_PAPER_DIR in resolved.parents:
        raise ValueError("v2失败归因禁止写入v1模拟盘目录")
    return resolved


def ablation_metadata(initial_cash: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "variant": name,
            "sequence": sequence,
            "label": details["label"],
            "component": details["component"],
            "sector_enabled": details["sector_enabled"],
            "max_per_sector": 2 if details["sector_enabled"] else 0,
            "rank_buffer": details["rank_buffer"],
            "drawdown_overlay": details["drawdown_overlay"],
            "frictionless": details["frictionless"],
            "top_n": 10,
            "initial_cash": float(initial_cash),
        }
        for sequence, (name, details) in enumerate(VARIANTS.items(), start=1)
    ])


def build_waterfall(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return additive, order-dependent changes from raw to full strategy."""
    indexed = metrics.set_index("variant")
    rows: list[dict[str, Any]] = []
    previous: Optional[pd.Series] = None
    for variant, details in VARIANTS.items():
        if variant not in indexed.index:
            raise ValueError(f"归因结果缺少变体: {variant}")
        current = indexed.loc[variant]
        rows.append({
            "variant": variant,
            "component": details["component"],
            "label": details["label"],
            "level_total_return": float(current["total_return"]),
            "level_cagr": float(current["cagr"]),
            "level_max_drawdown": float(current["max_drawdown"]),
            "delta_total_return": (
                0.0 if previous is None else
                float(current["total_return"] - previous["total_return"])
            ),
            "delta_cagr": (
                0.0 if previous is None else
                float(current["cagr"] - previous["cagr"])
            ),
            "delta_max_drawdown": (
                0.0 if previous is None else
                float(current["max_drawdown"] - previous["max_drawdown"])
            ),
        })
        previous = current
    return pd.DataFrame(rows)


def classify_failure(metrics: pd.DataFrame) -> dict[str, Any]:
    indexed = metrics.set_index("variant")
    raw = indexed.loc["raw_lowvol"]
    if float(raw["cagr"]) < 10.0 or float(raw["excess_return"]) <= 0.0:
        return {
            "primary_cause": "factor_translation",
            "primary_cause_label": CAUSE_LABELS["factor_translation"],
            "explanation": (
                "去掉行业、缓冲、回撤控制和费用后，原始低波动Top 10仍未达到"
                "年化10%且正超额收益；首要问题是因子从Rank IC到集中组合的转化。"
            ),
        }
    components = [
        ("sector_constraint", "add_sector_constraint"),
        ("rank_buffer", "add_rank_buffer"),
        ("drawdown_overlay", "add_drawdown_overlay"),
        ("trading_frictions", "full_framework"),
    ]
    previous = "raw_lowvol"
    impacts: list[tuple[str, float]] = []
    for component, variant in components:
        impacts.append((
            component,
            float(indexed.loc[variant, "cagr"] - indexed.loc[previous, "cagr"]),
        ))
        previous = variant
    primary, impact = min(impacts, key=lambda item: item[1])
    return {
        "primary_cause": primary,
        "primary_cause_label": CAUSE_LABELS[primary],
        "primary_cagr_impact_pp": impact,
        "explanation": (
            f"原始组合通过收益门槛，但{CAUSE_LABELS[primary]}造成最大的"
            "年化收益下降。"
        ),
    }


class AttributionEngine(V2FactorPortfolioBacktestEngine):
    def __init__(self, config, frames, *, overlay_enabled: bool):
        super().__init__(config, "low_volatility_60", frames)
        self.overlay_enabled = bool(overlay_enabled)

    def build_drawdown_policy(self):
        return super().build_drawdown_policy() if self.overlay_enabled else None


def _configure_variant(config, variant: str) -> None:
    details = VARIANTS[variant]
    config.scoring.momentum_weight = 1.0
    config.scoring.pullback_weight = 0.0
    config.sector_neutral.enabled = bool(details["sector_enabled"])
    config.sector_neutral.max_per_sector = 2
    config.portfolio.rank_buffer = int(details["rank_buffer"])
    if details["frictionless"]:
        config.execution.commission = 0.0
        config.execution.min_commission = 0.0
        config.execution.transfer_fee_rate = 0.0
        config.execution.stamp_duty_rate = 0.0
        config.execution.historical_stamp_duty_rate = 0.0
        config.execution.slippage_bps = 0.0


def _run_variant_period(
    *,
    base_config,
    variant: str,
    frames: dict[str, pd.DataFrame],
    shared,
    benchmark: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, float]:
    config = copy.deepcopy(base_config)
    _configure_variant(config, variant)
    engine = AttributionEngine(
        config, frames, overlay_enabled=VARIANTS[variant]["drawdown_overlay"]
    )
    _attach_shared_data(engine, shared)
    result = engine.run(start, end, mode="price_only")
    performance = result["performance"]
    execution = result["stats"].get("execution", {})
    trades = result["trade_log"]
    first_execution = pd.Timestamp(trades[0]["date"]) if trades else pd.NaT
    benchmark_return = (
        _benchmark_return(benchmark, first_execution, end)
        if pd.notna(first_execution) else np.nan
    )
    total_return = float(performance.get("total_return", 0.0))
    initial_cash = float(config.execution.initial_cash)
    fees = float(execution.get("total_fees", 0.0))
    slippage = float(execution.get("slippage_cost", 0.0))
    risk_events = result.get("risk_events", [])
    return {
        "total_return": total_return,
        "cagr": float(performance.get("cagr", 0.0)),
        "sharpe": float(performance.get("sharpe", 0.0)),
        "max_drawdown": float(performance.get("max_drawdown", 0.0)),
        "benchmark_return": float(benchmark_return),
        "excess_return": total_return - float(benchmark_return),
        "avg_turnover_pct": float(performance.get("avg_turnover_pct", 0.0)),
        "avg_deployment_pct": float(performance.get("avg_deployment_pct", 0.0)),
        "total_fees": fees,
        "slippage_cost": slippage,
        "friction_drag_pct_of_initial_cash": (
            (fees + slippage) / initial_cash * 100.0
        ),
        "risk_event_count": int(len(risk_events)),
        "warning_or_defensive_events": int(sum(
            event.get("state") in {"warning", "defensive"}
            for event in risk_events
        )),
    }


def selection_diagnostics(
    lowvol_frame: pd.DataFrame,
    universe_store,
) -> pd.DataFrame:
    previous: list[str] = []
    rows: list[dict[str, Any]] = []
    for date, group in lowvol_frame.groupby("date", sort=True):
        ranked = group.sort_values("momentum", ascending=False).copy()
        sectors = universe_store.snapshot_for(date).code_to_sector
        raw = ranked.head(10)["code"].astype(str).tolist()
        selected = select_with_rank_buffer(
            ranked=ranked,
            top_n=10,
            sectors=sectors,
            max_per_sector=2,
            previous_picks=previous,
            rank_buffer=30,
            sector_enabled=True,
        )
        raw_counts = pd.Series([sectors.get(code, "未知") for code in raw]).value_counts()
        selected_counts = pd.Series(
            [sectors.get(code, "未知") for code in selected]
        ).value_counts()
        ranks = dict(zip(ranked["code"].astype(str), range(1, len(ranked) + 1)))
        overlap = len(set(previous) & set(selected)) if previous else 0
        rows.append({
            "date": str(pd.Timestamp(date).date()),
            "selected_count": len(selected),
            "raw_unique_sectors": int(len(raw_counts)),
            "raw_max_sector_share": (
                float(raw_counts.max() / len(raw)) if raw else 0.0
            ),
            "constrained_unique_sectors": int(len(selected_counts)),
            "constrained_max_sector_share": (
                float(selected_counts.max() / len(selected)) if selected else 0.0
            ),
            "buffer_retained_outside_top10": int(sum(
                ranks.get(code, 10**9) > 10 for code in selected
            )),
            "holding_overlap_ratio": (
                float(overlap / len(previous)) if previous else np.nan
            ),
        })
        previous = list(selected)
    return pd.DataFrame(rows)


def _render_report(summary: dict, continuous: pd.DataFrame, waterfall: pd.DataFrame) -> str:
    cause = summary["failure_classification"]
    selection = summary.get("selection_summary", {})
    raw_sector_share = float(selection.get("average_raw_max_sector_share", 0.0))
    constrained_sector_share = float(
        selection.get("average_constrained_max_sector_share", 0.0)
    )
    reader_continuous = continuous.copy()
    if {"total_fees", "slippage_cost"}.issubset(reader_continuous.columns):
        reader_continuous["direct_cost"] = (
            pd.to_numeric(reader_continuous["total_fees"], errors="coerce").fillna(0)
            + pd.to_numeric(reader_continuous["slippage_cost"], errors="coerce").fillna(0)
        )
    continuous_columns = {
        "label": "版本",
        "total_return": "总收益(%)",
        "cagr": "年化收益(%)",
        "excess_return": "超额收益(百分点)",
        "max_drawdown": "最大回撤(%)",
        "avg_deployment_pct": "平均仓位(%)",
        "avg_turnover_pct": "平均换手(%)",
        "direct_cost": "直接费用(元)",
        "warning_or_defensive_events": "风险降仓次数",
    }
    visible_continuous = [
        column for column in continuous_columns
        if column in reader_continuous.columns
    ]
    reader_continuous = reader_continuous[visible_continuous].rename(
        columns=continuous_columns
    )
    waterfall_columns = {
        "label": "归因步骤",
        "level_total_return": "累计总收益(%)",
        "level_cagr": "累计年化(%)",
        "delta_total_return": "本步总收益影响(百分点)",
        "delta_cagr": "本步年化影响(百分点)",
        "level_max_drawdown": "累计最大回撤(%)",
    }
    visible_waterfall = [
        column for column in waterfall_columns if column in waterfall.columns
    ]
    reader_waterfall = waterfall[visible_waterfall].rename(
        columns=waterfall_columns
    )
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>v2低波动组合失败归因</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#1f2937}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 28px}}td,th{{border:1px solid #d1d5db;padding:7px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.lead{{background:#fff7ed;padding:15px;border-left:4px solid #f97316}}</style></head>
<body><h1>v2 低波动组合失败归因</h1>
<p class='lead'><strong>首要原因：{html.escape(cause.get('primary_cause_label', cause['primary_cause']))}</strong>。{html.escape(cause['explanation'])}</p>
<p>归因链为固定顺序，差额可加总但会受加入顺序影响。本报告仅解释历史研究失败，不授权实盘，不改变v1或v2参数。</p>
<h2>持仓结构</h2>
<ul><li>无行业限制时，第一大行业平均占比为{raw_sector_share:.1%}；加入限制后为{constrained_sector_share:.1%}。</li>
<li>行业数平均由{float(selection.get('average_raw_unique_sectors', 0.0)):.1f}个增加至{float(selection.get('average_constrained_unique_sectors', 0.0)):.1f}个。</li>
<li>缓冲机制平均保留{float(selection.get('average_buffer_retained_outside_top10', 0.0)):.1f}只当前排名10名以外的旧持仓，月度重合率为{float(selection.get('average_holding_overlap_ratio', 0.0)):.1%}。</li></ul>
<h2>连续区间变体结果</h2>{reader_continuous.to_html(index=False, border=0, float_format=lambda value: f'{value:.2f}')}
<h2>逐层收益影响</h2>{reader_waterfall.to_html(index=False, border=0, float_format=lambda value: f'{value:.2f}')}
<p><strong>路径依赖提示：</strong>回撤控制会根据前序净值改变后续仓位；因此费用加入后的收益差额可能呈正值。这不是费用创造收益，直接费用仍应以佣金、税费和滑点金额衡量。</p>
</body></html>"""


def run_attribution(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    train_months: int = 24,
    test_months: int = 6,
    initial_cash: float = 100_000,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    factor_output: Path = DEFAULT_FACTOR_OUTPUT,
    portfolio_output: Path = DEFAULT_PORTFOLIO_OUTPUT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = _safe_output_dir(output_dir)
    _validate_source(Path(factor_output))
    prior_summary_path = Path(portfolio_output) / "summary.json"
    prior_continuous_path = Path(portfolio_output) / "continuous_results.csv"
    if not prior_summary_path.exists() or not prior_continuous_path.exists():
        raise ValueError("缺少v2组合验证结果，请先运行 ./run_mac.sh v2portfolio")
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    if prior_summary.get("status") != "failed_research_gate":
        raise ValueError("本诊断只适用于未通过组合门槛的结果")

    folds: list[Fold] = generate_folds(start, end, train_months, test_months)
    if len(folds) < 7:
        raise ValueError("失败归因至少需要7个不重叠窗口")
    config = build_v2_config()
    config.execution.initial_cash = float(initial_cash)
    price_data = _load_price_cache(Path(cache_root), "2019-01-01", end)
    factors = pd.read_csv(Path(factor_output) / "factor_values.csv", dtype={"code": str})
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    closes = _close_frame(price_data, set(factors["date"]))
    frames = build_candidate_factor_frames(factors, closes)
    shared = _shared_engine(config, price_data, Path(snapshot_dir), start, end)
    benchmark = load_index_data("000001", start, end)
    if benchmark.empty:
        raise ValueError("上证综指000001数据为空")

    cache: dict[tuple[str, str, str], dict[str, float]] = {}
    def run_cached(variant: str, period_start: str, period_end: str) -> dict[str, float]:
        key = (variant, period_start, period_end)
        if key not in cache:
            cache[key] = _run_variant_period(
                base_config=config, variant=variant, frames=frames,
                shared=shared, benchmark=benchmark,
                start=period_start, end=period_end,
            )
        return cache[key]

    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        logger.info(f"低波动失败归因 Fold {fold.fold}: {fold.test_start}~{fold.test_end}")
        for variant in VARIANTS:
            metrics = run_cached(variant, fold.test_start, fold.test_end)
            fold_rows.append({
                "fold": fold.fold, "variant": variant,
                "test_start": fold.test_start, "test_end": fold.test_end,
                **{f"test_{key}": value for key, value in metrics.items()},
            })
    fold_frame = pd.DataFrame(fold_rows)
    continuous_start = folds[0].test_start
    continuous = pd.DataFrame([
        {"variant": variant, "label": details["label"],
         **run_cached(variant, continuous_start, end)}
        for variant, details in VARIANTS.items()
    ])
    waterfall = build_waterfall(continuous)
    classification = classify_failure(continuous)
    selection = selection_diagnostics(
        frames["low_volatility_60"], shared.universe_store
    )

    prior = pd.read_csv(prior_continuous_path)
    prior_low = prior[prior["candidate"] == "low_volatility_60"].iloc[0]
    full = continuous[continuous["variant"] == "full_framework"].iloc[0]
    reconcile_metrics = ["total_return", "cagr", "max_drawdown", "excess_return"]
    max_difference = max(
        abs(float(full[name]) - float(prior_low[name]))
        for name in reconcile_metrics
    )
    raw = continuous[continuous["variant"] == "raw_lowvol"].iloc[0]
    final = continuous[continuous["variant"] == "full_framework"].iloc[0]
    checks = {
        "prior_stage_failed_and_requires_attribution": prior_summary["status"] == "failed_research_gate",
        "five_variants_complete": continuous["variant"].tolist() == list(VARIANTS),
        "seven_fold_rows_per_variant": len(fold_frame) == len(VARIANTS) * len(folds),
        "full_framework_reconciles_to_prior": max_difference < 1e-9,
        "waterfall_total_return_reconciles": abs(
            float(waterfall["delta_total_return"].sum())
            - (float(final["total_return"]) - float(raw["total_return"]))
        ) < 1e-9,
        "v1_output_path_not_used": output != V1_PAPER_DIR and V1_PAPER_DIR not in output.parents,
    }
    if not all(checks.values()):
        raise ValueError(f"失败归因内部复核未通过: {checks}")

    selection_summary = {
        "signal_dates": int(len(selection)),
        "average_selected_holdings": float(selection["selected_count"].mean()),
        "average_raw_unique_sectors": float(selection["raw_unique_sectors"].mean()),
        "average_constrained_unique_sectors": float(selection["constrained_unique_sectors"].mean()),
        "average_raw_max_sector_share": float(selection["raw_max_sector_share"].mean()),
        "average_constrained_max_sector_share": float(selection["constrained_max_sector_share"].mean()),
        "average_buffer_retained_outside_top10": float(selection["buffer_retained_outside_top10"].mean()),
        "average_holding_overlap_ratio": float(selection["holding_overlap_ratio"].mean()),
    }
    result = {
        "strategy_id": "v2_monthly_top10_dd10",
        "stage_id": "v2_lowvol_failure_attribution",
        "status": "diagnosis_complete",
        "deployment_authorized": False,
        "failure_classification": classification,
        "selection_summary": selection_summary,
        "raw_portfolio": {key: float(raw[key]) for key in (
            "total_return", "cagr", "excess_return", "max_drawdown", "avg_deployment_pct"
        )},
        "full_framework": {key: float(final[key]) for key in (
            "total_return", "cagr", "excess_return", "max_drawdown", "avg_deployment_pct"
        )},
        "next_action": (
            "不放宽10%回撤门槛，也不微调当前低波动权重；先研究为何Top 10尾部选择"
            "不能保留横截面Rank IC，优先测试更宽持仓数或分位数组合。"
            if classification["primary_cause"] == "factor_translation"
            else "针对首要执行损耗做单变量稳健性验证。"
        ),
        "research_boundary": "归因结果受固定加入顺序影响，仅用于历史研究取舍。",
    }

    output.mkdir(parents=True, exist_ok=True)
    ablation_metadata(initial_cash).to_csv(output / "ablation_metadata.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(output / "fold_results.csv", index=False, encoding="utf-8-sig")
    continuous.to_csv(output / "continuous_results.csv", index=False, encoding="utf-8-sig")
    waterfall.to_csv(output / "attribution_waterfall.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(output / "selection_diagnostics.csv", index=False, encoding="utf-8-sig")
    (output / "validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(_render_report(result, continuous, waterfall), encoding="utf-8")
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行v2低波动组合失败归因")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--factor-output", default=str(DEFAULT_FACTOR_OUTPUT))
    parser.add_argument("--portfolio-output", default=str(DEFAULT_PORTFOLIO_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_attribution(
        start=args.start, end=args.end,
        train_months=args.train_months, test_months=args.test_months,
        initial_cash=args.initial_cash, cache_root=Path(args.cache_root),
        snapshot_dir=Path(args.snapshot_dir), factor_output=Path(args.factor_output),
        portfolio_output=Path(args.portfolio_output), output_dir=Path(args.output_dir),
    )
    print(f"低波动失败归因：{result['failure_classification']['primary_cause']}")
    print(result["failure_classification"]["explanation"])
    print(f"报告：{Path(args.output_dir) / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
