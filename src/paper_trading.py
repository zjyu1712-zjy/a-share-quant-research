"""Append-only forward paper-trading tracker.

The tracker deliberately separates four states:

1. a frozen research-only protocol;
2. close-of-period signals and intended target weights;
3. next-session simulated fills with realistic execution frictions;
4. daily close NAV and CSI 300 benchmark observations.

The JSONL event ledger is the audit source of truth. CSV/JSON/HTML files are
derived views and may be regenerated without changing the historical events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.config import StrategyConfig
from src.execution_engine import _portfolio_value, _rebalance_at_open
from src.price_feed import load_index_data, load_pool_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "paper_trading"
DRY_RUN_OUTPUT = PROJECT_ROOT / "data" / "paper_trading_dry_run"
DEFAULT_PENDING = PROJECT_ROOT / "data" / "pending_portfolio_state.json"
SCORING_START = "2026-08-03"
BENCHMARK_CODE = "000300"
STRATEGY_ID = "current_weekly_top5"
SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"无法序列化: {type(value)!r}")


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _sha256(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
    )


def _serialize_positions(positions: dict[str, dict]) -> dict[str, dict]:
    return json.loads(
        json.dumps(positions, ensure_ascii=False, default=_json_default)
    )


def build_protocol(
    *,
    initial_cash: float = 100_000,
    dry_run: bool = False,
) -> dict:
    """Return the immutable paper-account protocol and its content hash."""
    config = StrategyConfig.from_env()
    core = {
        "schema_version": SCHEMA_VERSION,
        "account_mode": "synthetic_dry_run" if dry_run else "official_forward",
        "strategy_id": STRATEGY_ID,
        "strategy_label": "当前正式周频 Top 5（仅作前向模拟记录）",
        "research_status": "historical_gates_failed",
        "deployment_authorized": False,
        "scoring_start": SCORING_START,
        "initial_cash": round(float(initial_cash), 2),
        "benchmark": {
            "name": "沪深300",
            "code": BENCHMARK_CODE,
            "anchor": "scoring_start_first_session_open",
            "valuation": "daily_close",
        },
        "signal_rule": {
            "frequency": "weekly",
            "timestamp": "period_end_close_after_market",
            "market_data_must_reach_signal_date": True,
            "target_count": 5,
            "rank_buffer": 0,
            "max_per_sector": 1,
            "factor": "equal_weight_price_momentum_plus_pullback",
            "effective_momentum_sessions": 8,
            "effective_pullback_sessions": 18,
            "sentiment_weight": 0.0,
            "market_timing": False,
            "concentration_filter": {
                "lookback_weeks": 4,
                "maximum_selections": 2,
                "mode": "exclude",
            },
        },
        "execution_rule": {
            "entry": "next_actual_trading_session_open",
            "weighting": "equal_weight",
            "minimum_lot": config.execution.min_lot,
            "commission": config.execution.commission,
            "minimum_commission": config.execution.min_commission,
            "transfer_fee_rate": config.execution.transfer_fee_rate,
            "stamp_duty_rate": config.execution.stamp_duty_rate,
            "historical_stamp_duty_rate": (
                config.execution.historical_stamp_duty_rate
            ),
            "stamp_duty_cutover_date": (
                config.execution.stamp_duty_cutover_date
            ),
            "slippage_bps": config.execution.slippage_bps,
            "maximum_volume_participation": (
                config.execution.max_volume_participation
            ),
            "volume_lookback": config.execution.volume_lookback,
            "price_limit_enforcement": (
                config.execution.enforce_price_limits
            ),
        },
        "data_integrity": {
            "stock_adjustment": "qfq",
            "ledger": "append_only_jsonl",
            "duplicate_policy": "idempotent_same_payload_reject_conflict",
            "prelaunch_observations_scored": False,
            "parameter_changes_after_launch": "new_protocol_required",
        },
        "decision_boundary": (
            "该账户仅检验冻结策略在真正未见数据中的运行表现；历史候选均未"
            "通过部署门槛，因此模拟盈利也不自动构成实盘授权或收益保证。"
        ),
    }
    return {**core, "protocol_sha256": _sha256(core)}


def _protocol_path(output_dir: Path) -> Path:
    return output_dir / "protocol.json"


def _ledger_path(output_dir: Path) -> Path:
    return output_dir / "events.jsonl"


def load_protocol(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    path = _protocol_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"模拟盘尚未初始化: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(output_dir: Path = DEFAULT_OUTPUT) -> list[dict]:
    path = _ledger_path(output_dir)
    if not path.exists():
        return []
    events: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"事件账本第{line_number}行损坏") from exc
    return events


def _event_identity(event: dict) -> str:
    return str(event["event_id"])


def _append_events(output_dir: Path, new_events: list[dict]) -> list[dict]:
    """Append only new events; same id is idempotent, same natural key conflicts."""
    if not new_events:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = load_events(output_dir)
    by_id = {_event_identity(event): event for event in existing}
    by_key = {
        str(event.get("natural_key")): event
        for event in existing
        if event.get("natural_key")
    }
    appended: list[dict] = []
    for raw_event in new_events:
        payload = {
            key: value
            for key, value in raw_event.items()
            if key not in {"event_id", "recorded_at"}
        }
        event_id = _sha256(payload)
        event = {
            **payload,
            "event_id": event_id,
            "recorded_at": raw_event.get("recorded_at", _now()),
        }
        if event_id in by_id:
            continue
        natural_key = str(event.get("natural_key", ""))
        if natural_key and natural_key in by_key:
            raise ValueError(
                f"事件冲突，已存在不同内容: {natural_key}"
            )
        appended.append(event)
        by_id[event_id] = event
        if natural_key:
            by_key[natural_key] = event
    if appended:
        with open(_ledger_path(output_dir), "a", encoding="utf-8") as handle:
            for event in appended:
                handle.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        default=_json_default,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    return appended


def initialize_tracker(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    initial_cash: float = 100_000,
    dry_run: bool = False,
) -> dict:
    """Initialize the official or synthetic account without scoring observations."""
    output_dir = Path(output_dir)
    protocol = build_protocol(initial_cash=initial_cash, dry_run=dry_run)
    protocol_path = _protocol_path(output_dir)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError(
                "已有模拟盘协议与当前配置不同；不得覆盖，需使用新目录"
            )
        protocol = existing
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(protocol_path, protocol)
    _append_events(
        output_dir,
        [{
            "event_type": "initialize",
            "natural_key": "initialize",
            "protocol_sha256": protocol["protocol_sha256"],
            "initial_cash": protocol["initial_cash"],
            "scoring_start": protocol["scoring_start"],
            "scored": False,
        }],
    )
    refresh_views(output_dir)
    return protocol


def tracker_initialized(output_dir: Path = DEFAULT_OUTPUT) -> bool:
    return _protocol_path(Path(output_dir)).exists()


def _normalize_targets(plan: dict) -> tuple[list[str], dict[str, float]]:
    positions = plan.get("positions", [])
    targets = [str(row["code"]).zfill(6) for row in positions]
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("待执行计划的股票代码为空或重复")
    supplied = {
        str(code).zfill(6): float(weight)
        for code, weight in (plan.get("target_weights") or {}).items()
        if str(code).zfill(6) in targets and float(weight) > 0
    }
    if set(supplied) == set(targets):
        total = sum(supplied.values())
        weights = {code: value / total for code, value in supplied.items()}
    else:
        weights = {code: 1.0 / len(targets) for code in targets}
    return targets, weights


def record_signal(
    plan: dict,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    allow_prelaunch: bool = True,
) -> dict:
    """Record one close-of-period signal without assuming any fill."""
    output_dir = Path(output_dir)
    protocol = load_protocol(output_dir)
    signal_date = str(pd.Timestamp(plan["signal_date"]).date())
    market_data_as_of = str(
        pd.Timestamp(plan.get("market_data_as_of", signal_date)).date()
    )
    if market_data_as_of != signal_date:
        raise ValueError(
            "行情尚未覆盖信号日，拒绝记录正式信号: "
            f"signal={signal_date}, market_data_as_of={market_data_as_of}"
        )
    expected_execution = str(
        pd.Timestamp(plan["expected_execution_date"]).date()
    )
    if (
        not allow_prelaunch
        and pd.Timestamp(expected_execution)
        < pd.Timestamp(protocol["scoring_start"])
    ):
        raise ValueError("预计成交日在正式计分起点之前")
    if (
        protocol["account_mode"] == "official_forward"
        and pd.Timestamp(expected_execution)
        < pd.Timestamp(protocol["scoring_start"])
    ):
        raise ValueError("正式账户不得写入计分起点之前的成交计划")

    targets, target_weights = _normalize_targets(plan)
    metadata = {
        str(row["code"]).zfill(6): {
            "name": str(row.get("name", row["code"])),
            "sector": str(row.get("sector", "?")),
            "signal_price": float(
                row.get("signal_price", row.get("entry_price", 0.0))
            ),
            "estimated_shares": int(row.get("shares", 0)),
        }
        for row in plan.get("positions", [])
    }
    event = {
        "event_type": "signal",
        "natural_key": f"signal:{signal_date}",
        "protocol_sha256": protocol["protocol_sha256"],
        "signal_date": signal_date,
        "market_data_as_of": market_data_as_of,
        "expected_execution_date": expected_execution,
        "targets": targets,
        "target_weights": target_weights,
        "security_metadata": metadata,
        "backup_pool": [
            str(code).zfill(6) for code in plan.get("backup_pool", [])
        ],
        "score_details": plan.get("score_details", {}),
        "scored": (
            pd.Timestamp(expected_execution)
            >= pd.Timestamp(protocol["scoring_start"])
            and protocol["account_mode"] == "official_forward"
        ),
    }
    _append_events(output_dir, [event])
    refresh_views(output_dir)
    return next(
        item
        for item in load_events(output_dir)
        if item.get("natural_key") == event["natural_key"]
    )


def capture_pending_plan(
    pending_path: Path = DEFAULT_PENDING,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict:
    path = Path(pending_path)
    if not path.exists():
        raise FileNotFoundError(f"待执行计划不存在: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "pending":
        raise ValueError("只接受 status=pending 的待执行计划")
    return record_signal(plan, output_dir)


def _latest_rebalance_state(
    events: list[dict],
    protocol: dict,
) -> tuple[float, dict[str, dict]]:
    rebalances = [
        event for event in events if event.get("event_type") == "rebalance"
    ]
    if not rebalances:
        return float(protocol["initial_cash"]), {}
    latest = max(rebalances, key=lambda event: event["execution_date"])
    return float(latest["cash_after"]), {
        code: dict(position)
        for code, position in latest["positions_after"].items()
    }


def _load_market_data(
    codes: list[str],
    *,
    start: str,
    end: str,
    price_data: Optional[dict[str, pd.DataFrame]] = None,
    index_data: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    stocks = price_data if price_data is not None else load_pool_data(
        codes, start, end
    )
    benchmark = index_data if index_data is not None else load_index_data(
        BENCHMARK_CODE, start, end
    )
    if benchmark.empty:
        raise ValueError("沪深300行情为空，不能确认交易日或计算基准")
    benchmark = benchmark.sort_index()
    return stocks, benchmark


def execute_due_signals(
    *,
    as_of: str,
    output_dir: Path = DEFAULT_OUTPUT,
    price_data: Optional[dict[str, pd.DataFrame]] = None,
    index_data: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """Execute every recorded, unfilled signal on its next actual index session."""
    output_dir = Path(output_dir)
    protocol = load_protocol(output_dir)
    events = load_events(output_dir)
    signals = sorted(
        (
            event for event in events
            if event.get("event_type") == "signal"
        ),
        key=lambda event: event["signal_date"],
    )
    executed_signals = {
        event["signal_event_id"]
        for event in events
        if event.get("event_type") == "rebalance"
    }
    due = [signal for signal in signals if signal["event_id"] not in executed_signals]
    if not due:
        return []

    cash, positions = _latest_rebalance_state(events, protocol)
    all_codes = sorted(
        set(positions)
        | {
            code
            for signal in due
            for code in signal["targets"]
        }
    )
    history_start = str(
        (pd.Timestamp(min(signal["signal_date"] for signal in due))
         - pd.Timedelta(days=90)).date()
    )
    stocks, benchmark = _load_market_data(
        all_codes,
        start=history_start,
        end=as_of,
        price_data=price_data,
        index_data=index_data,
    )
    available_sessions = pd.DatetimeIndex(
        benchmark.index[benchmark.index <= pd.Timestamp(as_of)]
    )
    rule = protocol["execution_rule"]
    appended: list[dict] = []

    for signal in due:
        future = available_sessions[
            available_sessions > pd.Timestamp(signal["signal_date"])
        ]
        if len(future) == 0:
            continue
        execution_date = pd.Timestamp(future[0])
        if (
            protocol["account_mode"] == "official_forward"
            and execution_date < pd.Timestamp(protocol["scoring_start"])
        ):
            raise ValueError("实际成交日在正式计分起点之前")
        cash, positions, log = _rebalance_at_open(
            date=execution_date,
            signal_date=pd.Timestamp(signal["signal_date"]),
            target=list(signal["targets"]),
            target_weights=dict(signal["target_weights"]),
            cash=cash,
            positions=positions,
            price_data=stocks,
            commission=float(rule["commission"]),
            min_lot=int(rule["minimum_lot"]),
            min_commission=float(rule["minimum_commission"]),
            transfer_fee_rate=float(rule["transfer_fee_rate"]),
            stamp_duty_rate=float(rule["stamp_duty_rate"]),
            historical_stamp_duty_rate=float(
                rule["historical_stamp_duty_rate"]
            ),
            stamp_duty_cutover_date=str(
                rule["stamp_duty_cutover_date"]
            ),
            slippage_bps=float(rule["slippage_bps"]),
            max_volume_participation=float(
                rule["maximum_volume_participation"]
            ),
            volume_lookback=int(rule["volume_lookback"]),
            enforce_price_limits=bool(rule["price_limit_enforcement"]),
        )
        event = {
            "event_type": "rebalance",
            "natural_key": f"rebalance:{signal['event_id']}",
            "protocol_sha256": protocol["protocol_sha256"],
            "signal_event_id": signal["event_id"],
            "signal_date": signal["signal_date"],
            "execution_date": str(execution_date.date()),
            "targets": signal["targets"],
            "target_weights": signal["target_weights"],
            "buys": log["buys"],
            "sells": log["sells"],
            "constraints": log["constraints"],
            "skipped": log["skipped"],
            "cash_after": round(float(cash), 2),
            "equity_at_open": round(float(log["equity"]), 2),
            "positions_after": _serialize_positions(positions),
            "scored": (
                protocol["account_mode"] == "official_forward"
                and execution_date >= pd.Timestamp(protocol["scoring_start"])
            ),
        }
        appended.extend(_append_events(output_dir, [event]))
    if appended:
        refresh_views(output_dir)
    return appended


def _metadata_by_code(events: list[dict]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for event in events:
        if event.get("event_type") == "signal":
            metadata.update(event.get("security_metadata", {}))
    return metadata


def _state_for_date(
    events: list[dict],
    protocol: dict,
    date: pd.Timestamp,
) -> tuple[float, dict[str, dict]]:
    eligible = [
        event
        for event in events
        if (
            event.get("event_type") == "rebalance"
            and pd.Timestamp(event["execution_date"]) <= date
        )
    ]
    if not eligible:
        return float(protocol["initial_cash"]), {}
    latest = max(eligible, key=lambda event: event["execution_date"])
    return float(latest["cash_after"]), {
        code: dict(position)
        for code, position in latest["positions_after"].items()
    }


def mark_to_market(
    *,
    as_of: str,
    output_dir: Path = DEFAULT_OUTPUT,
    price_data: Optional[dict[str, pd.DataFrame]] = None,
    index_data: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """Append all missing daily close observations through available index data."""
    output_dir = Path(output_dir)
    protocol = load_protocol(output_dir)
    events = load_events(output_dir)
    rebalances = [
        event for event in events if event.get("event_type") == "rebalance"
    ]
    all_codes = sorted({
        code
        for event in rebalances
        for code in event.get("positions_after", {})
    })
    start = protocol["scoring_start"]
    stocks, benchmark = _load_market_data(
        all_codes,
        start=str((pd.Timestamp(start) - pd.Timedelta(days=90)).date()),
        end=as_of,
        price_data=price_data,
        index_data=index_data,
    )
    sessions = pd.DatetimeIndex(
        benchmark.index[
            (benchmark.index >= pd.Timestamp(start))
            & (benchmark.index <= pd.Timestamp(as_of))
        ]
    )
    if len(sessions) == 0:
        return []
    first_session = pd.Timestamp(sessions[0])
    first_row = benchmark.loc[first_session]
    if isinstance(first_row, pd.DataFrame):
        first_row = first_row.iloc[-1]
    benchmark_anchor = float(first_row["open"])
    if benchmark_anchor <= 0:
        raise ValueError("沪深300计分起点开盘价无效")

    existing_dates = {
        event["date"]
        for event in events
        if event.get("event_type") == "mark"
    }
    metadata = _metadata_by_code(events)
    prior_marks = sorted(
        (
            event for event in events
            if event.get("event_type") == "mark"
        ),
        key=lambda event: event["date"],
    )
    prior_equity = (
        float(prior_marks[-1]["equity"])
        if prior_marks else float(protocol["initial_cash"])
    )
    high_water = max(
        [float(protocol["initial_cash"])]
        + [float(event["equity"]) for event in prior_marks]
    )
    new_events: list[dict] = []
    for date in sessions:
        date_str = str(pd.Timestamp(date).date())
        if date_str in existing_dates:
            continue
        cash, positions = _state_for_date(events, protocol, pd.Timestamp(date))
        equity, stock_value = _portfolio_value(
            cash,
            positions,
            stocks,
            pd.Timestamp(date),
            field="close",
        )
        benchmark_row = benchmark.loc[date]
        if isinstance(benchmark_row, pd.DataFrame):
            benchmark_row = benchmark_row.iloc[-1]
        benchmark_value = (
            float(protocol["initial_cash"])
            * float(benchmark_row["close"])
            / benchmark_anchor
        )
        high_water = max(high_water, float(equity))
        positions_snapshot: list[dict] = []
        for code, position in sorted(positions.items()):
            frame = stocks.get(code)
            eligible = (
                frame[frame.index <= date]
                if frame is not None else pd.DataFrame()
            )
            close = (
                float(eligible["close"].iloc[-1])
                if not eligible.empty else float(position["entry_price"])
            )
            meta = metadata.get(code, {})
            positions_snapshot.append({
                "code": code,
                "name": meta.get("name", code),
                "sector": meta.get("sector", "?"),
                "shares": int(position["shares"]),
                "entry_price": float(position["entry_price"]),
                "close": close,
                "market_value": round(int(position["shares"]) * close, 2),
                "unrealized_pct": round(
                    (close / float(position["entry_price"]) - 1) * 100,
                    4,
                ),
            })
        new_events.append({
            "event_type": "mark",
            "natural_key": f"mark:{date_str}",
            "protocol_sha256": protocol["protocol_sha256"],
            "date": date_str,
            "cash": round(float(cash), 2),
            "stock_value": round(float(stock_value), 2),
            "equity": round(float(equity), 2),
            "daily_return_pct": round(
                (float(equity) / prior_equity - 1) * 100, 6
            ),
            "cumulative_return_pct": round(
                (
                    float(equity) / float(protocol["initial_cash"]) - 1
                ) * 100,
                6,
            ),
            "drawdown_pct": round(
                (float(equity) / high_water - 1) * 100,
                6,
            ),
            "benchmark_value": round(benchmark_value, 2),
            "benchmark_return_pct": round(
                (
                    benchmark_value / float(protocol["initial_cash"]) - 1
                ) * 100,
                6,
            ),
            "excess_return_pp": round(
                (
                    float(equity) - benchmark_value
                ) / float(protocol["initial_cash"]) * 100,
                6,
            ),
            "positions": positions_snapshot,
            "scored": protocol["account_mode"] == "official_forward",
        })
        prior_equity = float(equity)
    appended = _append_events(output_dir, new_events)
    if appended:
        refresh_views(output_dir)
    return appended


def update_tracker(
    *,
    as_of: str,
    output_dir: Path = DEFAULT_OUTPUT,
    price_data: Optional[dict[str, pd.DataFrame]] = None,
    index_data: Optional[pd.DataFrame] = None,
) -> dict:
    fills = execute_due_signals(
        as_of=as_of,
        output_dir=output_dir,
        price_data=price_data,
        index_data=index_data,
    )
    marks = mark_to_market(
        as_of=as_of,
        output_dir=output_dir,
        price_data=price_data,
        index_data=index_data,
    )
    return {
        "rebalance_events_added": len(fills),
        "mark_events_added": len(marks),
        "status": refresh_views(Path(output_dir)),
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8-sig",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _fees_from_events(events: list[dict]) -> dict[str, float]:
    totals = {
        "commission": 0.0,
        "transfer_fee": 0.0,
        "stamp_duty": 0.0,
        "slippage_cost": 0.0,
        "total_fees": 0.0,
        "gross_turnover": 0.0,
    }
    for event in events:
        if event.get("event_type") != "rebalance":
            continue
        for fill in event.get("buys", []) + event.get("sells", []):
            fees = fill.get("fees", {})
            for name in ("commission", "transfer_fee", "stamp_duty"):
                totals[name] += float(fees.get(name, 0.0))
            totals["total_fees"] += float(fees.get("total", 0.0))
            totals["slippage_cost"] += float(fill.get("slippage_cost", 0.0))
            totals["gross_turnover"] += float(fill.get("gross_value", 0.0))
    return {key: round(value, 2) for key, value in totals.items()}


def _status_from_events(protocol: dict, events: list[dict]) -> dict:
    signals = [e for e in events if e.get("event_type") == "signal"]
    rebalances = [e for e in events if e.get("event_type") == "rebalance"]
    marks = sorted(
        (e for e in events if e.get("event_type") == "mark"),
        key=lambda event: event["date"],
    )
    latest = marks[-1] if marks else None
    cash, positions = _latest_rebalance_state(events, protocol)
    scored_marks = [event for event in marks if event.get("scored")]
    return {
        "schema_version": SCHEMA_VERSION,
        "account_mode": protocol["account_mode"],
        "strategy_id": protocol["strategy_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "research_status": protocol["research_status"],
        "deployment_authorized": protocol["deployment_authorized"],
        "scoring_start": protocol["scoring_start"],
        "initial_cash": protocol["initial_cash"],
        "signal_count": len(signals),
        "rebalance_count": len(rebalances),
        "scored_observation_count": len(scored_marks),
        "latest_mark_date": latest["date"] if latest else None,
        "current_equity": (
            latest["equity"] if latest else protocol["initial_cash"]
        ),
        "cumulative_return_pct": (
            latest["cumulative_return_pct"] if latest else 0.0
        ),
        "benchmark_return_pct": (
            latest["benchmark_return_pct"] if latest else 0.0
        ),
        "excess_return_pp": (
            latest["excess_return_pp"] if latest else 0.0
        ),
        "maximum_drawdown_pct": (
            min((float(mark["drawdown_pct"]) for mark in marks), default=0.0)
        ),
        "cash": latest["cash"] if latest else round(cash, 2),
        "position_count": (
            len(latest["positions"]) if latest else len(positions)
        ),
        "execution_totals": _fees_from_events(events),
        "readiness": (
            "synthetic_dry_run_only"
            if protocol["account_mode"] == "synthetic_dry_run"
            else (
                "forward_tracking_active"
                if scored_marks else "initialized_waiting_for_first_observation"
            )
        ),
        "updated": _now(),
    }


def _render_report(
    output_dir: Path,
    protocol: dict,
    status: dict,
    events: list[dict],
) -> None:
    marks = sorted(
        (event for event in events if event.get("event_type") == "mark"),
        key=lambda event: event["date"],
    )
    latest_positions = marks[-1].get("positions", []) if marks else []
    rebalances = sorted(
        (
            event for event in events
            if event.get("event_type") == "rebalance"
        ),
        key=lambda event: event["execution_date"],
        reverse=True,
    )
    position_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['code'])}</td>"
        f"<td>{html.escape(str(row.get('name', '')))}</td>"
        f"<td>{int(row['shares']):,}</td>"
        f"<td>{float(row['close']):,.2f}</td>"
        f"<td>{float(row['market_value']):,.0f}</td>"
        f"<td class=\"{'pos' if float(row['unrealized_pct']) >= 0 else 'neg'}\">"
        f"{float(row['unrealized_pct']):+.2f}%</td></tr>"
        for row in latest_positions
    ) or "<tr><td colspan='6' class='muted'>尚未成交，当前全部为现金。</td></tr>"
    fill_rows: list[str] = []
    for event in rebalances[:4]:
        for side, key in (("买入", "buys"), ("卖出", "sells")):
            for fill in event.get(key, []):
                fill_rows.append(
                    "<tr>"
                    f"<td>{event['execution_date']}</td><td>{side}</td>"
                    f"<td>{html.escape(fill['code'])}</td>"
                    f"<td>{int(fill['shares']):,}</td>"
                    f"<td>{float(fill['price']):,.2f}</td>"
                    f"<td>{float(fill.get('fees', {}).get('total', 0)):,.2f}</td>"
                    "</tr>"
                )
    fills_html = "".join(fill_rows) or (
        "<tr><td colspan='6' class='muted'>尚无模拟成交。</td></tr>"
    )
    mode_label = (
        "合成演练（不计分）"
        if protocol["account_mode"] == "synthetic_dry_run"
        else "正式前向模拟盘"
    )
    status_text = (
        "等待首次真实观察"
        if status["scored_observation_count"] == 0
        else f"已记录 {status['scored_observation_count']} 个交易日"
    )
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结策略前向模拟盘</title><style>
:root{{--ink:#17212b;--muted:#687580;--line:#dce4e8;--paper:#f4f6f7;--navy:#17344d;
--green:#157a55;--red:#b42318}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);
color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}
main{{max-width:1060px;margin:auto;padding:36px 22px 70px}}header{{background:var(--navy);color:#fff;
border-radius:18px;padding:30px}}h1{{font-size:31px;margin:4px 0 10px}}.eyebrow{{font-size:12px;
letter-spacing:.12em;opacity:.75}}.warning{{background:#fff7e8;border-left:5px solid #d97706;
padding:16px 20px;border-radius:10px;margin:16px 0}}.grid{{display:grid;
grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}.card,section{{background:#fff;
border:1px solid var(--line);border-radius:14px;padding:20px}}.card b{{display:block;font-size:23px;
margin-top:4px}}section{{margin:16px 0}}h2{{font-size:21px;margin:0 0 12px}}table{{width:100%;
border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e8edef;text-align:left}}
th{{background:#edf2f4}}.muted{{color:var(--muted)}}.pos{{color:var(--green)}}.neg{{color:var(--red)}}
code{{word-break:break-all}}@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}
main{{padding:14px 9px}}header{{padding:23px}}h1{{font-size:25px}}}}
</style></head><body><main><header><div class="eyebrow">FROZEN FORWARD PAPER ACCOUNT</div>
<h1>{mode_label}</h1><p>{status_text} · 策略参数冻结 · 下一真实交易日开盘模拟成交</p></header>
<div class="warning"><b>研究用途，不是可部署赚钱策略。</b> 历史准入与holdout均未通过；
本账户只记录未来表现，不因短期盈亏修改参数。</div>
<div class="grid">
<div class="card"><span class="muted">当前权益</span><b>¥{status['current_equity']:,.0f}</b></div>
<div class="card"><span class="muted">策略累计</span><b class="{'pos' if status['cumulative_return_pct'] >= 0 else 'neg'}">{status['cumulative_return_pct']:+.2f}%</b></div>
<div class="card"><span class="muted">沪深300</span><b>{status['benchmark_return_pct']:+.2f}%</b></div>
<div class="card"><span class="muted">累计超额</span><b class="{'pos' if status['excess_return_pp'] >= 0 else 'neg'}">{status['excess_return_pp']:+.2f}pp</b></div>
</div>
<section><h2>当前持仓</h2><table><thead><tr><th>代码</th><th>名称</th><th>股数</th>
<th>收盘价</th><th>市值</th><th>未实现收益</th></tr></thead><tbody>{position_rows}</tbody></table></section>
<section><h2>最近模拟成交</h2><table><thead><tr><th>日期</th><th>方向</th><th>代码</th>
<th>股数</th><th>成交价</th><th>费用</th></tr></thead><tbody>{fills_html}</tbody></table></section>
<section><h2>账户审计</h2><p>正式计分起点：<b>{protocol['scoring_start']}</b>；
已计分观察：<b>{status['scored_observation_count']}</b>；最大回撤：
<b>{status['maximum_drawdown_pct']:.2f}%</b>；累计费用：
<b>¥{status['execution_totals']['total_fees']:,.2f}</b>。</p>
<p class="muted">协议哈希：<code>{protocol['protocol_sha256']}</code><br>
最近净值日期：{status['latest_mark_date'] or '尚无'}；基准锚定为正式计分首个交易日开盘。</p></section>
</main></body></html>"""
    _atomic_write_text(output_dir / "report.html", report)


def refresh_views(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Regenerate audit-friendly CSV/JSON/HTML views from the event ledger."""
    output_dir = Path(output_dir)
    protocol = load_protocol(output_dir)
    events = load_events(output_dir)
    status = _status_from_events(protocol, events)

    signal_rows: list[dict] = []
    order_rows: list[dict] = []
    fill_rows: list[dict] = []
    nav_rows: list[dict] = []
    position_rows: list[dict] = []
    for event in events:
        if event.get("event_type") == "signal":
            signal_rows.append({
                "signal_date": event["signal_date"],
                "market_data_as_of": event["market_data_as_of"],
                "expected_execution_date": event["expected_execution_date"],
                "targets": "|".join(event["targets"]),
                "scored": event["scored"],
                "event_id": event["event_id"],
            })
            for code in event["targets"]:
                meta = event.get("security_metadata", {}).get(code, {})
                order_rows.append({
                    "signal_date": event["signal_date"],
                    "expected_execution_date": event["expected_execution_date"],
                    "code": code,
                    "name": meta.get("name", code),
                    "sector": meta.get("sector", "?"),
                    "target_weight": event["target_weights"].get(code),
                    "signal_price": meta.get("signal_price"),
                    "estimated_shares": meta.get("estimated_shares"),
                    "status": "planned",
                    "signal_event_id": event["event_id"],
                })
        elif event.get("event_type") == "rebalance":
            for side, key in (("BUY", "buys"), ("SELL", "sells")):
                for fill in event.get(key, []):
                    fees = fill.get("fees", {})
                    fill_rows.append({
                        "signal_date": event["signal_date"],
                        "execution_date": event["execution_date"],
                        "side": side,
                        "code": fill["code"],
                        "shares": fill["shares"],
                        "reference_price": fill["reference_price"],
                        "execution_price": fill["price"],
                        "gross_value": fill.get("gross_value"),
                        "commission": fees.get("commission", 0),
                        "transfer_fee": fees.get("transfer_fee", 0),
                        "stamp_duty": fees.get("stamp_duty", 0),
                        "total_fees": fees.get("total", 0),
                        "slippage_cost": fill.get("slippage_cost", 0),
                        "note": fill.get("note", ""),
                        "rebalance_event_id": event["event_id"],
                    })
        elif event.get("event_type") == "mark":
            nav_rows.append({
                field: event.get(field)
                for field in (
                    "date", "cash", "stock_value", "equity",
                    "daily_return_pct", "cumulative_return_pct",
                    "drawdown_pct", "benchmark_value",
                    "benchmark_return_pct", "excess_return_pp", "scored",
                )
            })
            for row in event.get("positions", []):
                position_rows.append({"date": event["date"], **row})

    _write_csv(
        output_dir / "signals.csv",
        signal_rows,
        [
            "signal_date", "market_data_as_of", "expected_execution_date",
            "targets", "scored", "event_id",
        ],
    )
    _write_csv(
        output_dir / "orders.csv",
        order_rows,
        [
            "signal_date", "expected_execution_date", "code", "name",
            "sector", "target_weight", "signal_price", "estimated_shares",
            "status", "signal_event_id",
        ],
    )
    _write_csv(
        output_dir / "fills.csv",
        fill_rows,
        [
            "signal_date", "execution_date", "side", "code", "shares",
            "reference_price", "execution_price", "gross_value",
            "commission", "transfer_fee", "stamp_duty", "total_fees",
            "slippage_cost", "note", "rebalance_event_id",
        ],
    )
    _write_csv(
        output_dir / "nav.csv",
        nav_rows,
        [
            "date", "cash", "stock_value", "equity", "daily_return_pct",
            "cumulative_return_pct", "drawdown_pct", "benchmark_value",
            "benchmark_return_pct", "excess_return_pp", "scored",
        ],
    )
    _write_csv(
        output_dir / "positions.csv",
        position_rows,
        [
            "date", "code", "name", "sector", "shares", "entry_price",
            "close", "market_value", "unrealized_pct",
        ],
    )
    _atomic_write_json(output_dir / "status.json", status)
    _render_report(output_dir, protocol, status, events)
    return status


def run_synthetic_dry_run(
    output_dir: Path = DRY_RUN_OUTPUT,
) -> dict:
    """Exercise signal → fill → NAV → benchmark without official observations."""
    output_dir = Path(output_dir)
    initialize_tracker(output_dir, initial_cash=100_000, dry_run=True)
    plan = {
        "status": "pending",
        "signal_date": "2026-07-31",
        "market_data_as_of": "2026-07-31",
        "expected_execution_date": "2026-08-03",
        "positions": [
            {
                "code": "000001", "name": "演练A", "sector": "银行",
                "signal_price": 10.0, "shares": 5000,
            },
            {
                "code": "600000", "name": "演练B", "sector": "银行",
                "signal_price": 20.0, "shares": 2500,
            },
        ],
        "target_weights": {"000001": 0.5, "600000": 0.5},
        "backup_pool": [],
    }
    record_signal(plan, output_dir)
    dates = pd.to_datetime(["2026-07-31", "2026-08-03", "2026-08-04"])
    price_data = {
        "000001": pd.DataFrame({
            "open": [10.0, 10.1, 10.3],
            "high": [10.2, 10.4, 10.6],
            "low": [9.9, 10.0, 10.2],
            "close": [10.0, 10.2, 10.5],
            "volume": [5_000_000, 5_100_000, 5_200_000],
        }, index=dates),
        "600000": pd.DataFrame({
            "open": [20.0, 19.8, 20.1],
            "high": [20.2, 20.2, 20.4],
            "low": [19.7, 19.6, 19.9],
            "close": [20.0, 20.0, 20.2],
            "volume": [4_000_000, 4_100_000, 4_200_000],
        }, index=dates),
    }
    index_data = pd.DataFrame({
        "open": [4000.0, 4020.0, 4040.0],
        "high": [4010.0, 4050.0, 4070.0],
        "low": [3980.0, 4010.0, 4030.0],
        "close": [4005.0, 4040.0, 4060.0],
        "volume": [1, 1, 1],
    }, index=dates)
    return update_tracker(
        as_of="2026-08-04",
        output_dir=output_dir,
        price_data=price_data,
        index_data=index_data,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="冻结策略前向模拟盘记录器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="初始化正式模拟盘")
    init_parser.add_argument("--capital", type=float, default=100_000)
    init_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    capture_parser = subparsers.add_parser(
        "capture", help="记录 weekly 生成的待执行计划"
    )
    capture_parser.add_argument("--pending", type=Path, default=DEFAULT_PENDING)
    capture_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    update_parser = subparsers.add_parser(
        "update", help="确认到期模拟成交并补记每日净值"
    )
    update_parser.add_argument(
        "--as-of",
        default=datetime.now().strftime("%Y-%m-%d"),
    )
    update_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    status_parser = subparsers.add_parser("status", help="刷新并显示账户状态")
    status_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    dry_parser = subparsers.add_parser(
        "dry-run", help="运行与正式账本隔离的合成演练"
    )
    dry_parser.add_argument("--output-dir", type=Path, default=DRY_RUN_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init":
        result = initialize_tracker(
            args.output_dir,
            initial_cash=args.capital,
            dry_run=False,
        )
    elif args.command == "capture":
        result = capture_pending_plan(args.pending, args.output_dir)
    elif args.command == "update":
        result = update_tracker(as_of=args.as_of, output_dir=args.output_dir)
    elif args.command == "status":
        result = refresh_views(args.output_dir)
    else:
        result = run_synthetic_dry_run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
