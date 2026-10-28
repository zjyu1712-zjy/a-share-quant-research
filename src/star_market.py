"""Isolated STAR Market controls for the v2 monthly research strategy.

This module intentionally has no dependency on ``src.paper_trading`` and
never writes to the v1.0 ``data/paper_trading`` account.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import html
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "v2_monthly_top10_dd10" / "star_market.json"
)
DEFAULT_STAR_UNIVERSE_DIR = (
    PROJECT_ROOT / "data" / "v2_monthly_top10_dd10" / "star_market" / "universe"
)
DEFAULT_STAR_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "v2_monthly_top10_dd10" / "star_market" / "reports"
)
V1_PAPER_TRADING_DIR = (PROJECT_ROOT / "data" / "paper_trading").resolve()


class IsolationError(ValueError):
    """Raised when a STAR Market output targets the v1.0 account."""


class UniverseEvidenceError(RuntimeError):
    """Raised when historical membership evidence is missing or unsafe."""


STAR_SNAPSHOT_COLUMNS = {
    "code",
    "name",
    "sector",
    "effective_date",
    "known_at",
    "source",
    "evidence_type",
    "inclusion_reason",
    "exclusion_reason",
    "is_member",
    "is_unprofitable_u",
}
ALLOWED_EVIDENCE_TYPES = {
    "official_historical_snapshot",
    "official_change_notice",
    "licensed_point_in_time_dataset",
}


@dataclass(frozen=True)
class StarMarketConfig:
    """Human-controlled gate and risk limits for STAR Market securities."""

    enabled: bool = False
    investor_permission_confirmed: bool = False
    max_portfolio_weight: float = 0.20
    max_holdings: int = 2
    exclude_unprofitable_u: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "enabled",
            "investor_permission_confirmed",
            "exclude_unprofitable_u",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} 必须是 true 或 false")
        if not 0 < float(self.max_portfolio_weight) <= 1:
            raise ValueError("max_portfolio_weight 必须大于 0 且不超过 1")
        if (
            isinstance(self.max_holdings, bool)
            or not isinstance(self.max_holdings, int)
            or self.max_holdings < 1
        ):
            raise ValueError("max_holdings 必须是大于等于 1 的整数")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StarMarketConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"科创板配置包含未知字段: {sorted(unknown)}")
        return cls(**dict(payload))

    @property
    def entry_open(self) -> bool:
        """The ranking gate opens only after both human-controlled flags."""
        return self.enabled and self.investor_permission_confirmed


def assert_isolated_output_path(path: str | Path) -> Path:
    """Reject the v1.0 account directory and all of its descendants."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(V1_PAPER_TRADING_DIR)
    except ValueError:
        return resolved
    raise IsolationError(
        "科创板子系统不得写入 current_weekly_top5 的 data/paper_trading 目录"
    )


def load_star_market_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> StarMarketConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"科创板配置不存在: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"科创板配置不是有效 JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("科创板配置顶层必须是对象")
    return StarMarketConfig.from_mapping(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def update_star_market_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    enabled: Optional[bool] = None,
    investor_permission_confirmed: Optional[bool] = None,
) -> StarMarketConfig:
    """Atomically apply human choices without evaluating broker eligibility."""
    config_path = Path(path)
    current = load_star_market_config(config_path)
    payload = asdict(current)
    if enabled is not None:
        payload["enabled"] = enabled
    if investor_permission_confirmed is not None:
        payload["investor_permission_confirmed"] = (
            investor_permission_confirmed
        )
    updated = StarMarketConfig.from_mapping(payload)
    if updated.enabled and not updated.investor_permission_confirmed:
        raise PermissionError(
            "尚未确认券商已开通科创板权限，入口保持关闭；"
            "程序不会判断或绕过券商适当性要求"
        )
    if updated.entry_open and not current.entry_open:
        activation = build_star_market_protocol(
            updated,
            effective_at=datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        )
        _atomic_write_json(
            config_path.with_name("star_market_activation_protocol.json"),
            activation,
        )
    _atomic_write_json(config_path, asdict(updated))
    return updated


def is_star_market_code(code: Any) -> bool:
    normalized = str(code).split(".")[-1].zfill(6)
    return normalized.startswith(("688", "689"))


def _strict_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise UniverseEvidenceError(f"{field_name} 必须是 true 或 false，当前: {value!r}")


@dataclass(frozen=True)
class StarUniverseSnapshot:
    effective_date: pd.Timestamp
    records: pd.DataFrame
    path: Path

    @property
    def members(self) -> pd.DataFrame:
        return self.records[self.records["is_member"]].copy()

    @property
    def codes(self) -> set[str]:
        return set(self.members["code"])


class HistoricalStarUniverse:
    """Load STAR membership only from dated, point-in-time evidence files."""

    def __init__(self, snapshot_dir: str | Path):
        self.snapshot_dir = Path(snapshot_dir)

    def _snapshot_paths(self) -> list[tuple[pd.Timestamp, Path]]:
        if not self.snapshot_dir.exists():
            return []
        snapshots: list[tuple[pd.Timestamp, Path]] = []
        for path in self.snapshot_dir.glob("*.csv"):
            try:
                effective = pd.Timestamp(path.stem).normalize()
            except (TypeError, ValueError):
                continue
            snapshots.append((effective, path))
        return sorted(snapshots, key=lambda item: item[0])

    def snapshot_for(self, as_of: str | pd.Timestamp) -> StarUniverseSnapshot:
        requested = pd.Timestamp(as_of).normalize()
        eligible = [item for item in self._snapshot_paths() if item[0] <= requested]
        if not eligible:
            raise UniverseEvidenceError(
                f"{requested.date()} 没有可靠的科创板历史时点股票池；"
                "研究已停止，禁止用当前成分名单回填"
            )
        effective, path = eligible[-1]
        records = self._read_and_validate(path, effective)
        return StarUniverseSnapshot(effective, records, path)

    def _read_and_validate(
        self,
        path: Path,
        effective: pd.Timestamp,
    ) -> pd.DataFrame:
        records = pd.read_csv(path, dtype={"code": str}).fillna("")
        missing = STAR_SNAPSHOT_COLUMNS - set(records.columns)
        if missing:
            raise UniverseEvidenceError(f"{path} 缺少字段: {sorted(missing)}")
        if records.empty:
            raise UniverseEvidenceError(f"{path} 是空快照，不能用于历史研究")

        records = records[list(sorted(STAR_SNAPSHOT_COLUMNS))].copy()
        records["code"] = records["code"].astype(str).str.split(".").str[-1].str.zfill(6)
        if records["code"].duplicated().any():
            duplicates = sorted(records.loc[records["code"].duplicated(), "code"].unique())
            raise UniverseEvidenceError(f"{path} 存在重复代码: {duplicates}")
        invalid_codes = sorted(code for code in records["code"] if not is_star_market_code(code))
        if invalid_codes:
            raise UniverseEvidenceError(f"{path} 包含非科创板代码: {invalid_codes}")

        row_effective = pd.to_datetime(records["effective_date"], errors="coerce")
        known_at = pd.to_datetime(records["known_at"], errors="coerce")
        if row_effective.isna().any() or known_at.isna().any():
            raise UniverseEvidenceError(f"{path} 含无效 effective_date 或 known_at")
        if not row_effective.dt.normalize().eq(effective).all():
            raise UniverseEvidenceError(
                f"{path} 的 effective_date 必须全部等于文件名 {effective.date()}"
            )
        if (known_at.dt.normalize() > effective).any():
            raise UniverseEvidenceError(
                f"{path} 含生效日之后才知道的信息，禁止未来信息泄漏"
            )

        text_columns = ["name", "sector", "source", "evidence_type"]
        for column in text_columns:
            if records[column].astype(str).str.strip().eq("").any():
                raise UniverseEvidenceError(f"{path} 的 {column} 不能为空")
        invalid_evidence = sorted(
            set(records["evidence_type"].astype(str)) - ALLOWED_EVIDENCE_TYPES
        )
        if invalid_evidence:
            raise UniverseEvidenceError(
                f"{path} 使用了不允许的证据类型 {invalid_evidence}；"
                "当前成分回填不能作为历史证据"
            )

        records["is_member"] = [
            _strict_bool(value, "is_member") for value in records["is_member"]
        ]
        records["is_unprofitable_u"] = [
            _strict_bool(value, "is_unprofitable_u")
            for value in records["is_unprofitable_u"]
        ]
        included_without_reason = (
            records["is_member"]
            & records["inclusion_reason"].astype(str).str.strip().eq("")
        )
        excluded_without_reason = (
            ~records["is_member"]
            & records["exclusion_reason"].astype(str).str.strip().eq("")
        )
        if included_without_reason.any() or excluded_without_reason.any():
            raise UniverseEvidenceError(
                f"{path} 的入选或剔除记录缺少对应原因"
            )
        records["effective_date"] = row_effective.dt.normalize()
        records["known_at"] = known_at.dt.normalize()
        return records.reset_index(drop=True)

    def audit(self, as_of: str | pd.Timestamp) -> dict[str, Any]:
        requested = pd.Timestamp(as_of).normalize()
        snapshot = self.snapshot_for(requested)
        records = snapshot.records
        excluded = records.loc[~records["is_member"]]
        digest = hashlib.sha256(snapshot.path.read_bytes()).hexdigest()
        reason_counts = {
            str(reason): int(count)
            for reason, count in excluded["exclusion_reason"].value_counts().items()
        }
        return {
            "requested_date": requested.strftime("%Y-%m-%d"),
            "effective_date": snapshot.effective_date.strftime("%Y-%m-%d"),
            "known_at_max": records["known_at"].max().strftime("%Y-%m-%d"),
            "snapshot_path": str(snapshot.path.resolve()),
            "sha256": digest,
            "sources": sorted(set(records["source"].astype(str))),
            "evidence_types": sorted(set(records["evidence_type"].astype(str))),
            "raw_row_count": int(len(records)),
            "included_count": int(records["is_member"].sum()),
            "excluded_count": int((~records["is_member"]).sum()),
            "exclusion_reason_counts": reason_counts,
            "unprofitable_u_count": int(records["is_unprofitable_u"].sum()),
            "historical_evidence_complete": True,
            "current_constituent_backfill_used": False,
        }


STAR_TARGET_COLUMNS = {
    "code",
    "name",
    "rank",
    "target_weight",
    "selection_reason",
}


def _validate_targets(frame: pd.DataFrame) -> pd.DataFrame:
    missing = STAR_TARGET_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"科创板候选缺少字段: {sorted(missing)}")
    targets = frame.copy()
    targets["code"] = targets["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    targets["rank"] = pd.to_numeric(targets["rank"], errors="raise")
    targets["target_weight"] = pd.to_numeric(
        targets["target_weight"], errors="raise"
    )
    if targets["code"].duplicated().any():
        raise ValueError("科创板候选代码不能重复")
    if (targets["rank"] <= 0).any():
        raise ValueError("科创板候选 rank 必须大于 0")
    if (targets["target_weight"] <= 0).any():
        raise ValueError("科创板候选 target_weight 必须大于 0")
    return targets.sort_values(["rank", "code"]).reset_index(drop=True)


def _audit_row(
    row: pd.Series,
    *,
    decision: str,
    reason: str,
    effective_date: Optional[pd.Timestamp] = None,
    final_weight: float = 0.0,
) -> dict[str, Any]:
    return {
        "code": row["code"],
        "name": row["name"],
        "rank": int(row["rank"]),
        "decision": decision,
        "reason": reason,
        "selection_reason": row["selection_reason"],
        "universe_effective_date": (
            effective_date.strftime("%Y-%m-%d")
            if effective_date is not None else ""
        ),
        "original_target_weight": float(row["target_weight"]),
        "final_target_weight": float(final_weight),
    }


def _name_has_u_marker(name: Any) -> bool:
    normalized = str(name).strip().upper().replace("－", "-")
    return normalized == "U" or normalized.endswith("-U")


def screen_star_candidates(
    candidates: pd.DataFrame,
    snapshot: StarUniverseSnapshot,
    config: StarMarketConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply point-in-time membership, double confirmation, and U gate."""
    ranked = _validate_targets(candidates)
    member_records = snapshot.members.set_index("code")
    eligible_indexes: list[int] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in ranked.iterrows():
        code = row["code"]
        if not is_star_market_code(code) or code not in member_records.index:
            reason = "not_in_historical_snapshot"
        elif not config.entry_open:
            reason = "star_market_entry_closed"
        else:
            evidence = member_records.loc[code]
            is_u = bool(evidence["is_unprofitable_u"]) or _name_has_u_marker(
                evidence.get("name", row["name"])
            )
            if config.exclude_unprofitable_u and is_u:
                reason = "unprofitable_u_excluded"
            else:
                reason = "selected"
                eligible_indexes.append(index)
        audit_rows.append(
            _audit_row(
                row,
                decision="selected" if reason == "selected" else "excluded",
                reason=reason,
                effective_date=snapshot.effective_date,
                final_weight=(
                    float(row["target_weight"]) if reason == "selected" else 0.0
                ),
            )
        )
    return (
        ranked.loc[eligible_indexes].reset_index(drop=True),
        pd.DataFrame(audit_rows),
    )


def enforce_star_market_limits(
    targets: pd.DataFrame,
    config: StarMarketConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap STAR holdings and weight without silently selecting substitutes."""
    ranked = _validate_targets(targets)
    audit_rows: list[dict[str, Any]] = []
    if not config.entry_open:
        for _, row in ranked.iterrows():
            audit_rows.append(
                _audit_row(
                    row,
                    decision="excluded",
                    reason="star_market_entry_closed",
                )
            )
        return ranked.iloc[0:0].copy(), pd.DataFrame(audit_rows)

    retained = ranked.iloc[: config.max_holdings].copy()
    retained_total = float(retained["target_weight"].sum())
    scale = (
        config.max_portfolio_weight / retained_total
        if retained_total > config.max_portfolio_weight
        else 1.0
    )
    retained["target_weight"] = retained["target_weight"] * scale

    retained_by_code = retained.set_index("code")
    for _, row in ranked.iterrows():
        code = row["code"]
        if code not in retained_by_code.index:
            audit_rows.append(
                _audit_row(
                    row,
                    decision="excluded",
                    reason="max_star_holdings_exceeded",
                )
            )
            continue
        final_weight = float(retained_by_code.loc[code, "target_weight"])
        audit_rows.append(
            _audit_row(
                row,
                decision="selected",
                reason=(
                    "selected_weight_capped" if scale < 1 else "selected_within_limits"
                ),
                final_weight=final_weight,
            )
        )
    return retained.reset_index(drop=True), pd.DataFrame(audit_rows)


STAR_NAV_COLUMNS = {
    "date",
    "portfolio_equity",
    "star_equity",
    "star_daily_net_pnl",
    "star_fees",
    "star_slippage_cost",
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_star_market_protocol(
    config: StarMarketConfig,
    *,
    effective_at: Optional[str] = None,
) -> dict[str, Any]:
    """Freeze the isolated subsystem rules into a content-addressed protocol."""
    core = {
        "schema_version": 1,
        "strategy_id": "v2_monthly_top10_dd10",
        "subsystem": "star_market",
        "account_isolation": "independent_from_current_weekly_top5",
        "entry_gate": {
            "requires_enabled": True,
            "requires_investor_permission_confirmed": True,
            "broker_eligibility_evaluated_by_program": False,
        },
        "config": asdict(config),
        "historical_universe_policy": (
            "point_in_time_evidence_required_no_current_constituent_backfill"
        ),
        "reporting": (
            "return_drawdown_fees_slippage_selection_and_exclusion_separate"
        ),
    }
    if effective_at is not None:
        core["effective_at"] = effective_at
    return {**core, "protocol_sha256": _canonical_hash(core)}


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        newline="",
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_nav(nav: pd.DataFrame) -> pd.DataFrame:
    missing = STAR_NAV_COLUMNS - set(nav.columns)
    if missing:
        raise ValueError(f"科创板净值数据缺少字段: {sorted(missing)}")
    if nav.empty:
        raise ValueError("科创板净值数据为空，不能生成绩效报告")
    result = nav.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    numeric_columns = STAR_NAV_COLUMNS - {"date"}
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if result[column].isna().any():
            raise ValueError(f"科创板净值字段 {column} 含空值")
    if (result["portfolio_equity"] <= 0).any():
        raise ValueError("portfolio_equity 必须大于 0")
    if (result["star_equity"] < 0).any():
        raise ValueError("star_equity 不能为负数")
    if result["date"].duplicated().any():
        raise ValueError("科创板净值日期不能重复")
    return result.sort_values("date").reset_index(drop=True)


def write_star_market_report(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    selection_audit: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: Optional[StarMarketConfig] = None,
) -> dict[str, Any]:
    """Write an independent STAR performance, cost, and decision report."""
    output = assert_isolated_output_path(output_dir)
    clean_nav = _validated_nav(nav)
    required_audit = {"decision", "reason"}
    missing_audit = required_audit - set(selection_audit.columns)
    if missing_audit:
        raise ValueError(f"科创板选择审计缺少字段: {sorted(missing_audit)}")

    positive_indexes = clean_nav.index[clean_nav["star_equity"] > 0]
    if len(positive_indexes) == 0:
        star_return_pct = 0.0
        max_drawdown_pct = 0.0
    else:
        active_equity = clean_nav.loc[positive_indexes[0] :, "star_equity"]
        star_return_pct = float(
            (active_equity.iloc[-1] / active_equity.iloc[0] - 1) * 100
        )
        drawdown = active_equity / active_equity.cummax() - 1
        max_drawdown_pct = float(drawdown.min() * 100)

    latest = clean_nav.iloc[-1]
    reason_counts = {
        str(reason): int(count)
        for reason, count in selection_audit["reason"].value_counts().items()
    }
    resolved_config = config or load_star_market_config()
    protocol = build_star_market_protocol(resolved_config)
    summary = {
        "schema_version": 1,
        "strategy_id": "v2_monthly_top10_dd10",
        "subsystem": "star_market",
        "entry_open": resolved_config.entry_open,
        "protocol_sha256": protocol["protocol_sha256"],
        "first_date": clean_nav.iloc[0]["date"].strftime("%Y-%m-%d"),
        "last_date": latest["date"].strftime("%Y-%m-%d"),
        "star_market_return_pct": star_return_pct,
        "star_market_max_drawdown_pct": max_drawdown_pct,
        "net_pnl": float(clean_nav["star_daily_net_pnl"].sum()),
        "fees": float(clean_nav["star_fees"].sum()),
        "slippage_cost": float(clean_nav["star_slippage_cost"].sum()),
        "latest_star_equity": float(latest["star_equity"]),
        "latest_portfolio_weight_pct": float(
            latest["star_equity"] / latest["portfolio_equity"] * 100
        ),
        "selected_count": int((selection_audit["decision"] == "selected").sum()),
        "excluded_count": int((selection_audit["decision"] == "excluded").sum()),
        "selection_reason_counts": reason_counts,
    }

    output.mkdir(parents=True, exist_ok=True)
    export_nav = clean_nav.copy()
    export_nav["date"] = export_nav["date"].dt.strftime("%Y-%m-%d")
    _atomic_write_csv(output / "star_market_nav.csv", export_nav)
    _atomic_write_csv(output / "star_market_fills.csv", fills.copy())
    _atomic_write_csv(
        output / "star_market_selection_audit.csv",
        selection_audit.copy(),
    )
    _atomic_write_json(output / "star_market_protocol.json", protocol)
    _atomic_write_json(output / "star_market_summary.json", summary)

    reason_rows = "".join(
        f"<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>"
        for reason, count in sorted(reason_counts.items())
    ) or "<tr><td>尚无选择记录</td><td>0</td></tr>"
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>科创板独立报告</title><style>
body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;
max-width:980px;margin:36px auto;padding:0 20px;color:#17212b}}.warning{{padding:14px;
background:#fff7e8;border-left:5px solid #d97706}}.grid{{display:grid;
grid-template-columns:repeat(4,1fr);gap:10px}}.card{{border:1px solid #dce4e8;
border-radius:12px;padding:15px}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px;border-bottom:1px solid #e8edef;text-align:left}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr 1fr}}}}</style></head>
<body><h1>科创板子系统独立报告</h1><div class="warning">入口状态：
<b>{'已开启' if resolved_config.entry_open else '关闭'}</b>。券商权限只由用户人工确认，
程序不判断或绕过适当性要求。</div><div class="grid">
<div class="card">科创板收益<br><b>{star_return_pct:+.2f}%</b></div>
<div class="card">科创板最大回撤<br><b>{max_drawdown_pct:.2f}%</b></div>
<div class="card">费用<br><b>{summary['fees']:.2f}</b></div>
<div class="card">滑点<br><b>{summary['slippage_cost']:.2f}</b></div></div>
<h2>入选与剔除原因</h2><table><thead><tr><th>原因</th><th>数量</th></tr></thead>
<tbody>{reason_rows}</tbody></table><h2>审计边界</h2><p>策略：v2_monthly_top10_dd10；
与 current_weekly_top5 正式模拟盘完全隔离。</p><p>协议哈希：
<code>{protocol['protocol_sha256']}</code></p></body></html>"""
    _atomic_write_text(output / "star_market_report.html", report)
    return summary


def _status_payload(config: StarMarketConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "entry_open": config.entry_open,
        "entry_status": "open" if config.entry_open else "closed",
        "broker_eligibility_evaluated_by_program": False,
        "reference_only": {
            "average_assets": "申请前20个交易日日均资产不低于50万元",
            "trading_experience": "交易经验满24个月",
            "risk_disclosure": "签署风险揭示书",
            "authority": "实际以届时券商和上交所规则为准",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v2 月频策略的独立科创板入口设置（不操作 v1.0 账本）"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="科创板独立配置文件",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="查看入口和人工确认状态")

    enable = subparsers.add_parser("enable", help="开启科创板入口")
    enable.add_argument(
        "--permission-confirmed",
        action="store_true",
        help="用户确认券商已实际开通科创板交易权限",
    )
    subparsers.add_parser("disable", help="关闭入口，保留权限确认记录")
    subparsers.add_parser(
        "revoke-permission",
        help="撤销权限确认并同时关闭入口",
    )
    audit = subparsers.add_parser(
        "audit-universe",
        help="审计指定日期的历史时点科创板股票池",
    )
    audit.add_argument("--as-of", required=True, help="信号日 YYYY-MM-DD")
    audit.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_STAR_UNIVERSE_DIR,
    )
    audit.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            result = _status_payload(load_star_market_config(args.config))
        elif args.command == "enable":
            result = _status_payload(update_star_market_config(
                args.config,
                enabled=True,
                investor_permission_confirmed=(
                    True if args.permission_confirmed else None
                ),
            ))
        elif args.command == "disable":
            result = _status_payload(update_star_market_config(
                args.config,
                enabled=False,
            ))
        elif args.command == "revoke-permission":
            result = _status_payload(update_star_market_config(
                args.config,
                enabled=False,
                investor_permission_confirmed=False,
            ))
        else:
            result = HistoricalStarUniverse(args.snapshot_dir).audit(args.as_of)
            if args.output is not None:
                audit_output = assert_isolated_output_path(args.output)
                _atomic_write_json(audit_output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        FileNotFoundError,
        IsolationError,
        PermissionError,
        UniverseEvidenceError,
        ValueError,
    ) as exc:
        print(f"操作未完成：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
