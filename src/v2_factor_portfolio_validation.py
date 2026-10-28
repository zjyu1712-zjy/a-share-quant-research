"""Portfolio validation for the two factors admitted by the frozen v2 screen.

This stage is research-only.  It reuses the v2 monthly execution engine and
never reads from or writes to the official v1 paper-trading directory.
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

from src.backtest_engine import BacktestEngine
from src.low_volatility_portfolio_validation import _attach_shared_data
from src.price_feed import load_index_data
from src.universe import PointInTimeUniverse
from src.v2_factor_data import DEFAULT_CACHE_ROOT, DEFAULT_SNAPSHOT_DIR
from src.v2_monthly_strategy import V2MonthlyBacktestEngine, build_v2_config
from src.v2_monthly_validation import evaluate_v2_gate
from src.walk_forward_attribution import _benchmark_return
from src.walk_forward_validation import Fold, generate_folds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTOR_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "v2_independent_factor_screening"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "backtest_results" / "v2_factor_portfolio_validation"
)
V1_PAPER_DIR = (PROJECT_ROOT / "data" / "paper_trading").resolve()

PASSED_FACTORS = [
    "low_volatility_60",
    "sector_neutral_book_to_price_mrq",
]
CANDIDATES: dict[str, dict[str, Any]] = {
    "low_volatility_60": {
        "label": "60日低波动",
        "low_volatility_weight": 1.0,
        "book_to_price_weight": 0.0,
    },
    "sector_neutral_book_to_price_mrq": {
        "label": "行业中性1/PB",
        "low_volatility_weight": 0.0,
        "book_to_price_weight": 1.0,
    },
    "lowvol50_value50": {
        "label": "低波动50% + 行业中性1/PB 50%",
        "low_volatility_weight": 0.5,
        "book_to_price_weight": 0.5,
    },
}


def _safe_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == V1_PAPER_DIR or V1_PAPER_DIR in resolved.parents:
        raise ValueError("v2组合研究禁止写入v1模拟盘目录")
    return resolved


def candidate_metadata(initial_cash: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "candidate": name,
            "label": details["label"],
            "low_volatility_weight": details["low_volatility_weight"],
            "book_to_price_weight": details["book_to_price_weight"],
            "normalization": "cross_sectional_zscore_by_rebalance_date",
            "rebalance_frequency": "monthly",
            "top_n": 10,
            "rank_buffer": 30,
            "eligible_old_rank": 40,
            "max_per_sector": 2,
            "weighting": "equal_weight",
            "benchmark": "SSE Composite / 000001",
            "max_drawdown_limit_pct": 10.0,
            "reference_initial_cash": float(initial_cash),
        }
        for name, details in CANDIDATES.items()
    ])


def build_candidate_factor_frames(
    factors: pd.DataFrame,
    closes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Map the frozen admitted factors to the unified engine's two inputs."""
    required = {"date", "code", *PASSED_FACTORS}
    missing = required - set(factors.columns)
    if missing:
        raise ValueError(f"冻结因子表缺少字段: {sorted(missing)}")
    local = factors[list(required)].copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce").dt.normalize()
    local["code"] = local["code"].astype(str).str.zfill(6)
    price = closes[["date", "code", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
    price["code"] = price["code"].astype(str).str.zfill(6)
    base = local.merge(
        price, on=["date", "code"], how="inner", validate="one_to_one"
    ).dropna(subset=["close"])

    frames: dict[str, pd.DataFrame] = {}
    low = base[["date", "code", "close", "low_volatility_60"]].copy()
    low["momentum"] = low["low_volatility_60"]
    low["pullback"] = 0.0
    frames["low_volatility_60"] = low

    value = base[
        ["date", "code", "close", "sector_neutral_book_to_price_mrq"]
    ].copy()
    value["momentum"] = value["sector_neutral_book_to_price_mrq"]
    value["pullback"] = 0.0
    frames["sector_neutral_book_to_price_mrq"] = value

    blend = base.copy()
    blend["momentum"] = blend["low_volatility_60"]
    blend["pullback"] = blend["sector_neutral_book_to_price_mrq"]
    frames["lowvol50_value50"] = blend
    return frames


def evaluate_candidate_gate(folds: pd.DataFrame, continuous: dict) -> dict:
    """Apply the already frozen v2 10% drawdown and return gates."""
    return evaluate_v2_gate(folds, continuous)


class V2FactorPortfolioBacktestEngine(V2MonthlyBacktestEngine):
    def __init__(self, config, candidate: str, frames: dict[str, pd.DataFrame]):
        super().__init__(config)
        self.candidate = candidate
        self.frames = frames

    def build_price_factor_frame(self, rebalance_dates, adjusted_factors):
        del adjusted_factors
        wanted = set(pd.DatetimeIndex(rebalance_dates).normalize())
        frame = self.frames[self.candidate]
        return frame[frame["date"].isin(wanted)].copy()


def _load_price_cache(cache_root: Path, start: str, end: str) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for path in sorted((Path(cache_root) / "prices").glob("price_*.csv")):
        code = path.stem.removeprefix("price_").zfill(6)
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = frame[
            (frame["date"] >= start_ts) & (frame["date"] <= end_ts)
        ].copy()
        if frame.empty:
            continue
        frame["code"] = code
        prices[code] = frame.set_index("date").sort_index()
    if not prices:
        raise ValueError("v2价格缓存为空，请先运行 ./run_mac.sh v2factordata")
    return prices


def _close_frame(
    price_data: dict[str, pd.DataFrame], signal_dates: set[pd.Timestamp]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for code, frame in price_data.items():
        if "close" not in frame.columns:
            continue
        dates = frame.index.intersection(pd.DatetimeIndex(signal_dates))
        if len(dates):
            current = frame.loc[dates, ["close"]].reset_index()
            current.columns = ["date", "close"]
            current["code"] = code
            rows.append(current[["date", "code", "close"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["date", "code", "close"]
    )


def effective_universe_start(
    requested_start: str,
    available_dates: Iterable[pd.Timestamp],
) -> str:
    """Clamp metadata loading to the first real point-in-time snapshot."""
    dates = sorted(pd.Timestamp(value).normalize() for value in available_dates)
    if not dates:
        raise ValueError("历史时点股票池没有可用快照")
    effective = max(pd.Timestamp(requested_start).normalize(), dates[0])
    return str(effective.date())


def _shared_engine(
    config,
    price_data: dict[str, pd.DataFrame],
    snapshot_dir: Path,
    start: str,
    end: str,
) -> BacktestEngine:
    shared = BacktestEngine(config)
    shared.price_data = price_data
    shared.universe_store = PointInTimeUniverse(snapshot_dir)
    metadata_start = effective_universe_start(
        start, shared.universe_store.available_dates
    )
    members = shared.universe_store.union_for_range(metadata_start, end)
    shared.code_to_name = dict(zip(members["code"], members["name"]))
    shared.code_to_sector = dict(zip(members["code"], members["sector"]))
    shared.name_to_code = {name: code for code, name in shared.code_to_name.items()}
    return shared


def _run_period(
    *,
    base_config,
    candidate: str,
    frames: dict[str, pd.DataFrame],
    shared: BacktestEngine,
    benchmark: pd.DataFrame,
    start: str,
    end: str,
) -> dict[str, float]:
    config = copy.deepcopy(base_config)
    details = CANDIDATES[candidate]
    if candidate == "lowvol50_value50":
        config.scoring.momentum_weight = float(details["low_volatility_weight"])
        config.scoring.pullback_weight = float(details["book_to_price_weight"])
    else:
        config.scoring.momentum_weight = 1.0
        config.scoring.pullback_weight = 0.0
    engine = V2FactorPortfolioBacktestEngine(config, candidate, frames)
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
    return {
        "total_return": total_return,
        "cagr": float(performance.get("cagr", 0.0)),
        "sharpe": float(performance.get("sharpe", 0.0)),
        "max_drawdown": float(performance.get("max_drawdown", 0.0)),
        "benchmark_return": float(benchmark_return),
        "excess_return": total_return - float(benchmark_return),
        "avg_turnover_pct": float(performance.get("avg_turnover_pct", 0.0)),
        "total_fees": fees,
        "slippage_cost": slippage,
        "friction_drag_pct_of_initial_cash": (
            (fees + slippage) / initial_cash * 100.0
        ),
        "avg_deployment_pct": float(performance.get("avg_deployment_pct", 0.0)),
    }


def _compound(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((np.prod(1 + values / 100.0) - 1) * 100.0) if len(values) else 0.0


def _summarize(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "folds": int(len(rows)),
        "compounded_oos_return": _compound(rows["test_total_return"]),
        "compounded_benchmark_return": _compound(rows["test_benchmark_return"]),
        "median_fold_excess_return": float(rows["test_excess_return"].median()),
        "profitable_folds": int((rows["test_total_return"] > 0).sum()),
        "outperforming_folds": int((rows["test_excess_return"] > 0).sum()),
        "worst_fold_drawdown": float(rows["test_max_drawdown"].min()),
        "mean_fold_sharpe": float(rows["test_sharpe"].mean()),
        "total_fees": float(rows["test_total_fees"].sum()),
        "total_slippage_cost": float(rows["test_slippage_cost"].sum()),
    }


def _validate_source(factor_output: Path) -> dict[str, Any]:
    summary_path = factor_output / "summary.json"
    values_path = factor_output / "factor_values.csv"
    if not summary_path.exists() or not values_path.exists():
        raise ValueError("缺少v2独立因子筛选结果，请先运行 ./run_mac.sh v2factors")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "factor_passed":
        raise ValueError("v2独立因子尚未通过准入")
    if summary.get("advanced_factors") != PASSED_FACTORS:
        raise ValueError("通过因子与冻结组合设计不一致，拒绝继续")
    return summary


def _render_report(summary: dict, table: pd.DataFrame) -> str:
    selected = summary.get("selected_candidate") or "无"
    status = summary["status"]
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>v2通过因子组合验证</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#1f2937}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d1d5db;padding:7px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.warn{{background:#fff7ed;padding:14px;border-left:4px solid #f97316}}</style></head>
<body><h1>v2 通过因子组合验证</h1><p class='warn'>研究状态：{html.escape(status)}；晋级候选：{html.escape(selected)}。本报告不授权实盘，也不覆盖v1模拟盘。</p>
<p>统一口径：月频Top 10、排名缓冲30、每行业最多2只、下一交易日开盘、真实交易费用与成交限制；主基准为上证综指000001；严格最大回撤门槛10%。</p>
{table.to_html(index=False, border=0)}
</body></html>"""


def run_validation(
    *,
    start: str = "2020-01-01",
    end: str = "2025-05-12",
    train_months: int = 24,
    test_months: int = 6,
    initial_cash: float = 100_000,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    factor_output: Path = DEFAULT_FACTOR_OUTPUT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = _safe_output_dir(output_dir)
    source_summary = _validate_source(Path(factor_output))
    folds: list[Fold] = generate_folds(start, end, train_months, test_months)
    if len(folds) < 7:
        raise ValueError("组合验证至少需要7个不重叠样本外窗口")

    config = build_v2_config()
    config.execution.initial_cash = float(initial_cash)
    price_data = _load_price_cache(Path(cache_root), "2019-01-01", end)
    factors = pd.read_csv(Path(factor_output) / "factor_values.csv", dtype={"code": str})
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    if factors.duplicated(["date", "code"]).any():
        raise ValueError("冻结因子表存在重复date-code记录")
    closes = _close_frame(price_data, set(factors["date"]))
    frames = build_candidate_factor_frames(factors, closes)
    for candidate, frame in frames.items():
        if frame["date"].nunique() != factors["date"].nunique():
            raise ValueError(f"{candidate}信号日期不完整")

    shared = _shared_engine(config, price_data, Path(snapshot_dir), start, end)
    logger.info("加载上证综指000001作为唯一主基准")
    benchmark = load_index_data("000001", start, end)
    if benchmark.empty:
        raise ValueError("上证综指000001数据为空")

    cache: dict[tuple[str, str, str], dict[str, float]] = {}
    def run_cached(candidate: str, period_start: str, period_end: str) -> dict[str, float]:
        key = (candidate, period_start, period_end)
        if key not in cache:
            cache[key] = _run_period(
                base_config=config, candidate=candidate, frames=frames,
                shared=shared, benchmark=benchmark,
                start=period_start, end=period_end,
            )
        return cache[key]

    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        logger.info(f"组合验证 Fold {fold.fold}: {fold.test_start}~{fold.test_end}")
        for candidate in CANDIDATES:
            metrics = run_cached(candidate, fold.test_start, fold.test_end)
            fold_rows.append({
                "fold": fold.fold, "candidate": candidate,
                "test_start": fold.test_start, "test_end": fold.test_end,
                **{f"test_{key}": value for key, value in metrics.items()},
            })
    fold_frame = pd.DataFrame(fold_rows)
    benchmark_counts = fold_frame.groupby("fold")["test_benchmark_return"].nunique(dropna=False)
    if not (benchmark_counts == 1).all():
        raise ValueError("同一窗口候选使用了不同基准收益")

    continuous_start = folds[0].test_start
    continuous_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gates: dict[str, dict] = {}
    for candidate, details in CANDIDATES.items():
        continuous = run_cached(candidate, continuous_start, end)
        gate = evaluate_candidate_gate(
            fold_frame[fold_frame["candidate"] == candidate], continuous
        )
        gates[candidate] = gate
        continuous_rows.append({"candidate": candidate, **continuous})
        summary_rows.append({
            "strategy": candidate, "label": details["label"],
            **_summarize(fold_frame[fold_frame["candidate"] == candidate]),
            "continuous_cagr": continuous["cagr"],
            "continuous_excess_return": continuous["excess_return"],
            "continuous_max_drawdown": continuous["max_drawdown"],
            "strict_gate_passed": gate["passed"],
        })
    strategy_summary = pd.DataFrame(summary_rows)
    passing = strategy_summary[strategy_summary["strict_gate_passed"]].sort_values(
        ["continuous_excess_return", "continuous_cagr", "mean_fold_sharpe"],
        ascending=False,
    )
    selected = None if passing.empty else str(passing.iloc[0]["strategy"])
    status = "passed_research_gate" if selected else "failed_research_gate"
    validation = {
        "status": "passed",
        "checks": {
            "source_factor_screen_passed": source_summary["status"] == "factor_passed",
            "exactly_three_frozen_candidates": list(frames) == list(CANDIDATES),
            "seven_or_more_non_overlapping_folds": len(folds) >= 7,
            "same_sse_benchmark_within_each_fold": bool((benchmark_counts == 1).all()),
            "v1_output_path_not_used": V1_PAPER_DIR not in output.parents and output != V1_PAPER_DIR,
        },
        "candidate_gates": gates,
    }
    if not all(validation["checks"].values()):
        raise ValueError("组合验证内部复核未通过")
    result = {
        "strategy_id": "v2_monthly_top10_dd10",
        "stage_id": "v2_factor_portfolio_validation",
        "status": status,
        "deployment_authorized": False,
        "selected_candidate": selected,
        "benchmark": "SSE Composite / 000001",
        "period": {"start": start, "continuous_start": continuous_start, "end": end, "folds": len(folds)},
        "decision": (
            f"{CANDIDATES[selected]['label']}通过严格组合门槛，下一步只做压力测试。"
            if selected else "三个候选均未同时满足年化10%、超额收益和严格10%回撤门槛，不晋级。"
        ),
        "research_boundary": "同一历史样本上的组合复验，不是未来未见数据，不构成收益承诺。",
    }

    output.mkdir(parents=True, exist_ok=True)
    candidate_metadata(initial_cash).to_csv(output / "candidate_metadata.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(output / "fold_results.csv", index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(output / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(continuous_rows).to_csv(output / "continuous_results.csv", index=False, encoding="utf-8-sig")
    (output / "validation_checks.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(_render_report(result, strategy_summary), encoding="utf-8")
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="验证v2已通过因子的三个固定组合")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--factor-output", default=str(DEFAULT_FACTOR_OUTPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_validation(
        start=args.start, end=args.end,
        train_months=args.train_months, test_months=args.test_months,
        initial_cash=args.initial_cash, cache_root=Path(args.cache_root),
        snapshot_dir=Path(args.snapshot_dir), factor_output=Path(args.factor_output),
        output_dir=Path(args.output_dir),
    )
    print(f"v2组合验证状态：{result['status']}")
    print(f"晋级候选：{result['selected_candidate'] or '无'}")
    print(f"报告：{Path(args.output_dir) / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
