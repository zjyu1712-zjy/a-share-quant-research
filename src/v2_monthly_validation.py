"""Infrastructure audit and strict research gates for v2."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.sse_composite_universe import DEFAULT_SNAPSHOT_DIR
from src.v2_monthly_strategy import DEFAULT_OUTPUT, ensure_v2_protocol


def evaluate_v2_gate(folds: pd.DataFrame, continuous: dict) -> dict:
    """Evaluate pre-declared advancement gates without changing parameters."""
    required = {
        "test_total_return", "test_excess_return", "test_max_drawdown"
    }
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(f"走步结果缺少字段: {sorted(missing)}")
    if len(folds) < 7:
        raise ValueError("v2晋级判定至少需要7个样本外窗口")
    returns = pd.to_numeric(folds["test_total_return"], errors="coerce")
    excess = pd.to_numeric(folds["test_excess_return"], errors="coerce")
    drawdowns = pd.to_numeric(folds["test_max_drawdown"], errors="coerce")
    if returns.isna().any() or excess.isna().any() or drawdowns.isna().any():
        raise ValueError("走步结果含无法解析的数值")

    checks = {
        "continuous_cagr_gte_10": float(continuous.get("cagr", float("-inf"))) >= 10.0,
        "continuous_excess_gt_0": float(continuous.get("excess_return", float("-inf"))) > 0.0,
        "continuous_max_drawdown_gte_minus_10": float(
            continuous.get("max_drawdown", float("-inf"))
        ) >= -10.0,
        "every_fold_max_drawdown_gte_minus_10": bool((drawdowns >= -10.0).all()),
        "at_least_5_of_7_profitable": int((returns > 0).sum()) >= 5,
        "at_least_5_of_7_outperform": int((excess > 0).sum()) >= 5,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "folds": int(len(folds)),
        "profitable_folds": int((returns > 0).sum()),
        "outperforming_folds": int((excess > 0).sum()),
        "worst_fold_drawdown": float(drawdowns.min()),
        "continuous": {
            "cagr": float(continuous.get("cagr", 0.0)),
            "excess_return": float(continuous.get("excess_return", 0.0)),
            "max_drawdown": float(continuous.get("max_drawdown", 0.0)),
        },
    }


def _snapshot_audits(
    snapshot_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audits: list[dict] = []
    exclusions: list[dict] = []
    if snapshot_dir.exists():
        for path in sorted(snapshot_dir.glob("*.csv")):
            try:
                effective = pd.Timestamp(path.stem).normalize()
            except ValueError:
                continue
            if not start <= effective <= end:
                continue
            sidecar = path.with_suffix(".audit.json")
            if sidecar.exists():
                row = json.loads(sidecar.read_text(encoding="utf-8"))
            else:
                members = pd.read_csv(path, dtype={"code": str})
                row = {
                    "effective_date": str(effective.date()),
                    "final_members": int(len(members)),
                    "audit_sidecar_missing": True,
                }
            row["snapshot_path"] = str(path)
            audits.append(row)
            for key, value in row.items():
                if key.startswith("excluded_"):
                    exclusions.append({
                        "effective_date": str(effective.date()),
                        "reason": key.removeprefix("excluded_"),
                        "count": int(value),
                    })
    audit_frame = pd.DataFrame(audits)
    exclusion_frame = pd.DataFrame(
        exclusions, columns=["effective_date", "reason", "count"]
    )
    return audit_frame, exclusion_frame


def _render_report(summary: dict, checks: dict) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{'通过' if value else '未通过'}</td></tr>"
        for key, value in checks.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>v2上证综指研究基础设施审计</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;max-width:960px;margin:32px auto;padding:0 20px;color:#1f2937}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d1d5db;padding:8px;text-align:left}}.warn{{background:#fff7ed;padding:14px;border-left:4px solid #f97316}}</style>
</head><body><h1>v2 上证综指可交易A股池审计</h1>
<p class="warn">当前状态：仅完成研究基础设施，不生成买入清单，不授权实盘，也不替换v1模拟盘。</p>
<p>审计区间：{summary['start']} 至 {summary['end']}；历史快照：{summary['snapshot_count']}期。</p>
<p>应有月份：{summary['expected_snapshot_months']}期；缺失月份：{html.escape(', '.join(summary['missing_snapshot_months']) or '无')}。</p>
<p>主基准：{html.escape(summary['primary_benchmark'])}</p>
<table><thead><tr><th>检查</th><th>结果</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def run_infrastructure_audit(
    *,
    start: str,
    end: str,
    output_dir: Path = DEFAULT_OUTPUT,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> dict:
    """Audit local v2 universe artifacts and write a fixed report package."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError(f"起始日期晚于结束日期: {start} > {end}")
    output_dir = Path(output_dir)
    snapshot_dir = Path(snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path, protocol_hash = ensure_v2_protocol(output_dir)
    audits, exclusions = _snapshot_audits(snapshot_dir, start_ts, end_ts)
    audits.to_csv(
        output_dir / "universe_audit_by_date.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exclusions.to_csv(
        output_dir / "universe_exclusions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sidecars_complete = bool(
        not audits.empty
        and "audit_sidecar_missing" not in audits.columns
    )
    member_counts = (
        pd.to_numeric(audits.get("final_members"), errors="coerce")
        if not audits.empty and "final_members" in audits.columns
        else pd.Series(dtype=float)
    )
    expected_periods = list(pd.period_range(start_ts, end_ts, freq="M"))
    if audits.empty or "effective_date" not in audits.columns:
        available_periods: set[pd.Period] = set()
    else:
        available_periods = set(
            pd.to_datetime(audits["effective_date"], errors="coerce")
            .dropna()
            .dt.to_period("M")
        )
    missing_periods = [
        str(period) for period in expected_periods
        if period not in available_periods
    ]
    checks = {
        "protocol_frozen": protocol_path.exists() and len(protocol_hash) == 64,
        "has_point_in_time_snapshots": not audits.empty,
        "every_requested_month_has_snapshot": not missing_periods,
        "all_snapshots_have_audit_sidecars": sidecars_complete,
        "all_snapshots_have_at_least_100_members": bool(
            not member_counts.empty and (member_counts >= 100).all()
        ),
        "v1_output_path_is_not_used": "data/paper_trading" not in str(output_dir),
    }
    summary = {
        "strategy_id": "v2_monthly_top10_dd10",
        "research_status": "infrastructure_only",
        "deployment_authorized": False,
        "primary_benchmark": "SSE Composite / 000001",
        "secondary_benchmark": "CSI 300 / 000300",
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "snapshot_count": int(len(audits)),
        "expected_snapshot_months": int(len(expected_periods)),
        "missing_snapshot_months": missing_periods,
        "first_snapshot": (
            str(audits["effective_date"].min()) if not audits.empty else None
        ),
        "last_snapshot": (
            str(audits["effective_date"].max()) if not audits.empty else None
        ),
        "minimum_members": (
            int(member_counts.min()) if not member_counts.empty else 0
        ),
        "maximum_members": (
            int(member_counts.max()) if not member_counts.empty else 0
        ),
        "protocol_sha256": protocol_hash,
        "infrastructure_checks_passed": bool(all(checks.values())),
        "next_stage": "independent_factor_rank_ic_screening",
    }
    (output_dir / "validation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(
        _render_report(summary, checks), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行v2股票池基础设施审计")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_infrastructure_audit(
        start=args.start,
        end=args.end,
        output_dir=Path(args.output_dir),
        snapshot_dir=Path(args.snapshot_dir),
    )
    print(
        f"v2基础审计完成：{summary['snapshot_count']}期快照；"
        f"检查={'通过' if summary['infrastructure_checks_passed'] else '未通过'}"
    )
    print(f"报告：{Path(args.output_dir) / 'report.html'}")
    return 0 if summary["infrastructure_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
