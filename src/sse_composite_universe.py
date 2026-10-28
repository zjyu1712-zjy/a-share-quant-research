"""Point-in-time reconstruction of the tradable SSE Composite A-share pool.

The official current constituent file must never be backfilled into history.
Historical snapshots are conservatively reconstructed from securities active
on each date, stable IPO dates, exchange/board rules and contemporaneous names.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import pandas as pd

from src.universe import UniverseSnapshotError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "universe" / "sse_composite_a"
MEMBERSHIP_METHOD = "conservative_point_in_time_reconstruction"
MAIN_BOARD_PREFIXES = ("600", "601", "603", "605")
STAR_PREFIXES = ("688", "689")
B_SHARE_PREFIX = "900"


class SSEDataAdapter(Protocol):
    def stock_basics(self) -> pd.DataFrame: ...

    def active_securities(self, as_of: str) -> pd.DataFrame: ...

    def industries(self, as_of: str) -> dict[str, str]: ...


def _rows(result) -> list[list[str]]:
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(
            f"baostock 查询失败: {result.error_code} {result.error_msg}"
        )
    return rows


class BaoStockSSEAdapter:
    """Small adapter around Baostock with deterministic DataFrame schemas."""

    def __init__(self):
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError(
                "缺少 baostock；请先执行 pip install -r requirements.txt"
            ) from exc
        self.bs = bs
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(
                f"baostock 登录失败: {login.error_code} {login.error_msg}"
            )
        self._closed = False
        self._basics: Optional[pd.DataFrame] = None
        self._active_cache: dict[str, pd.DataFrame] = {}

    def close(self) -> None:
        if not self._closed:
            self.bs.logout()
            self._closed = True

    def __enter__(self) -> "BaoStockSSEAdapter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def stock_basics(self) -> pd.DataFrame:
        if self._basics is None:
            result = self.bs.query_stock_basic()
            self._basics = pd.DataFrame(_rows(result), columns=result.fields)
        return self._basics.copy()

    def active_securities(self, as_of: str) -> pd.DataFrame:
        key = str(pd.Timestamp(as_of).date())
        if key not in self._active_cache:
            result = self.bs.query_all_stock(day=key)
            self._active_cache[key] = pd.DataFrame(
                _rows(result), columns=result.fields
            )
        return self._active_cache[key].copy()

    def industries(self, as_of: str) -> dict[str, str]:
        result = self.bs.query_stock_industry(
            date=str(pd.Timestamp(as_of).date())
        )
        frame = pd.DataFrame(_rows(result), columns=result.fields)
        if frame.empty or not {"code", "industry"}.issubset(frame.columns):
            return {}
        return {
            str(code).split(".")[-1].zfill(6): (industry or "未知")
            for code, industry in zip(frame["code"], frame["industry"])
        }


def _required_columns(frame: pd.DataFrame, names: set[str], label: str) -> None:
    missing = names - set(frame.columns)
    if missing:
        raise UniverseSnapshotError(f"{label}缺少字段: {sorted(missing)}")


def build_snapshot(
    as_of: str,
    adapter: SSEDataAdapter,
    *,
    include_star: bool = False,
    minimum_listing_days: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """Build one conservative, auditable, point-in-time snapshot."""
    effective = pd.Timestamp(as_of).normalize()
    try:
        basics = adapter.stock_basics()
        active = adapter.active_securities(str(effective.date()))
    except Exception as exc:
        raise UniverseSnapshotError(
            f"{effective.date()} 历史时点证券名单读取失败: {exc}"
        ) from exc
    if active.empty:
        raise UniverseSnapshotError(
            f"{effective.date()} 没有历史时点证券名单，拒绝用当前名单回填"
        )
    _required_columns(
        basics,
        {"code", "code_name", "ipoDate", "type"},
        "证券基础数据",
    )
    _required_columns(
        active,
        {"code", "tradeStatus", "code_name"},
        "历史活跃证券数据",
    )

    base = basics[["code", "code_name", "ipoDate", "type"]].copy()
    active = active[["code", "tradeStatus", "code_name"]].copy()
    merged = active.merge(
        base,
        on="code",
        how="left",
        suffixes=("_asof", "_basic"),
        validate="many_to_one",
    )
    merged["raw_code"] = merged["code"].astype(str)
    merged["numeric_code"] = (
        merged["raw_code"].str.split(".").str[-1].str.zfill(6)
    )
    merged["name"] = merged["code_name_asof"].fillna(
        merged["code_name_basic"]
    ).fillna("")
    merged["list_date"] = pd.to_datetime(merged["ipoDate"], errors="coerce")
    merged["listing_days"] = (effective - merged["list_date"]).dt.days
    merged["is_stock"] = merged["type"].astype(str).eq("1")
    merged["is_shanghai"] = merged["raw_code"].str.startswith("sh.")
    merged["is_b_share"] = merged["numeric_code"].str.startswith(B_SHARE_PREFIX)
    merged["is_star"] = merged["numeric_code"].str.startswith(STAR_PREFIXES)
    merged["is_main_board"] = merged["numeric_code"].str.startswith(
        MAIN_BOARD_PREFIXES
    )
    merged["is_st"] = merged["name"].str.upper().str.contains("ST", na=False)
    merged["is_suspended"] = ~merged["tradeStatus"].astype(str).eq("1")
    merged["listing_age_ok"] = merged["listing_days"].ge(
        minimum_listing_days
    )

    audit = {
        "effective_date": str(effective.date()),
        "raw_securities": int(len(merged)),
        "excluded_non_stock": int((~merged["is_stock"]).sum()),
        "excluded_other_exchange": int(
            (merged["is_stock"] & ~merged["is_shanghai"]).sum()
        ),
        "excluded_b_share": int(
            (merged["is_stock"] & merged["is_shanghai"] & merged["is_b_share"]).sum()
        ),
        "excluded_star": int(
            (merged["is_stock"] & merged["is_shanghai"] & merged["is_star"]).sum()
        ) if not include_star else 0,
        "excluded_listing_age": int(
            (
                merged["is_stock"]
                & merged["is_shanghai"]
                & (merged["is_main_board"] | (include_star & merged["is_star"]))
                & ~merged["listing_age_ok"]
            ).sum()
        ),
        "excluded_st": int(
            (
                merged["is_stock"]
                & merged["is_shanghai"]
                & (merged["is_main_board"] | (include_star & merged["is_star"]))
                & merged["listing_age_ok"]
                & merged["is_st"]
            ).sum()
        ),
        "excluded_suspended": int(
            (
                merged["is_stock"]
                & merged["is_shanghai"]
                & (merged["is_main_board"] | (include_star & merged["is_star"]))
                & merged["listing_age_ok"]
                & ~merged["is_st"]
                & merged["is_suspended"]
            ).sum()
        ),
        "include_star": bool(include_star),
        "minimum_listing_days": int(minimum_listing_days),
        "source": "baostock_as_of",
        "membership_method": MEMBERSHIP_METHOD,
    }

    board_ok = merged["is_main_board"] | (include_star & merged["is_star"])
    keep = (
        merged["is_stock"]
        & merged["is_shanghai"]
        & board_ok
        & ~merged["is_b_share"]
        & merged["listing_age_ok"]
        & ~merged["is_st"]
        & ~merged["is_suspended"]
    )
    selected = merged.loc[keep].copy()
    industry_map = adapter.industries(str(effective.date()))
    members = pd.DataFrame({
        "code": selected["numeric_code"].astype(str),
        "name": selected["name"].astype(str),
        "sector": selected["numeric_code"].map(industry_map).fillna("未知"),
        "list_date": selected["list_date"].dt.strftime("%Y-%m-%d"),
        "board": selected["is_star"].map({True: "star", False: "main"}),
        "trade_status": selected["tradeStatus"].astype(str),
        "is_st": selected["is_st"].astype(bool),
        "source": "baostock_as_of",
        "membership_method": MEMBERSHIP_METHOD,
        "effective_date": str(effective.date()),
    }).sort_values("code").drop_duplicates("code").reset_index(drop=True)
    audit["final_members"] = int(len(members))
    audit["main_board_members"] = int((members["board"] == "main").sum())
    audit["star_members"] = int((members["board"] == "star").sum())
    return members, audit


def _resolve_session_on_or_before(
    target: pd.Timestamp,
    adapter: SSEDataAdapter,
    *,
    max_lookback_days: int = 10,
) -> pd.Timestamp:
    for offset in range(max_lookback_days + 1):
        candidate = target - pd.Timedelta(days=offset)
        try:
            frame = adapter.active_securities(str(candidate.date()))
        except Exception as exc:
            raise UniverseSnapshotError(
                f"{candidate.date()} 历史时点证券名单读取失败: {exc}"
            ) from exc
        if not frame.empty:
            return candidate.normalize()
    raise UniverseSnapshotError(
        f"{target.date()} 向前{max_lookback_days}天没有可验证交易日"
    )


def _requested_month_ends(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError(f"起始日期晚于结束日期: {start} > {end}")
    periods = pd.period_range(start_ts, end_ts, freq="M")
    targets: list[pd.Timestamp] = []
    for period in periods:
        target = min(period.end_time.normalize(), end_ts)
        if target >= start_ts:
            targets.append(target)
    return targets


def _complete_existing_snapshot(
    snapshot_dir: Path,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> Optional[Path]:
    """Return a complete, provenance-valid snapshot already saved in a month."""
    for csv_path in sorted(snapshot_dir.glob("*.csv")):
        try:
            effective = pd.Timestamp(csv_path.stem).normalize()
        except (TypeError, ValueError):
            continue
        if not period_start <= effective <= period_end:
            continue
        audit_path = csv_path.with_suffix(".audit.json")
        if not audit_path.exists():
            continue
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            columns = set(pd.read_csv(csv_path, nrows=1).columns)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            audit.get("effective_date") == str(effective.date())
            and audit.get("membership_method") == MEMBERSHIP_METHOD
            and int(audit.get("final_members", 0)) > 0
            and {"code", "name", "sector", "effective_date"} <= columns
        ):
            return csv_path
    return None


def update_sse_snapshots(
    start: str,
    end: Optional[str] = None,
    *,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    adapter: Optional[SSEDataAdapter] = None,
    signal_dates: Optional[Iterable[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """Build resumable monthly snapshots and matching audit sidecars."""
    end = end or date.today().isoformat()
    owns_adapter = adapter is None
    resolved_adapter = adapter or BaoStockSSEAdapter()
    target_dir = Path(snapshot_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        written: list[Path] = []
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if signal_dates is None:
            work_items = []
            for target in _requested_month_ends(start, end):
                period_start = max(target.to_period("M").start_time.normalize(), start_ts)
                period_end = min(target.to_period("M").end_time.normalize(), end_ts)
                work_items.append((target, period_start, period_end, True))
        else:
            work_items = []
            for item in sorted(set(signal_dates)):
                target = pd.Timestamp(item).normalize()
                work_items.append((target, target, target, False))

        total = len(work_items)
        for index, (target, period_start, period_end, resolve_session) in enumerate(
            work_items, start=1
        ):
            if target < start_ts or target > end_ts:
                continue
            existing = _complete_existing_snapshot(
                target_dir, period_start, period_end
            )
            if existing is not None:
                if progress:
                    progress(f"[{index}/{total}] 跳过已完成月份：{existing.stem}")
                continue
            session = (
                _resolve_session_on_or_before(target, resolved_adapter)
                if resolve_session
                else target
            )
            members, audit = build_snapshot(
                str(session.date()), resolved_adapter, include_star=False
            )
            output = target_dir / f"{session.date()}.csv"
            members.to_csv(output, index=False, encoding="utf-8-sig")
            audit_path = target_dir / f"{session.date()}.audit.json"
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(output)
            if progress:
                progress(
                    f"[{index}/{total}] 已保存：{session.date()}，"
                    f"{len(members)} 只"
                )
        return written
    finally:
        if owns_adapter and hasattr(resolved_adapter, "close"):
            resolved_adapter.close()  # type: ignore[attr-defined]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="更新历史时点上证综指可交易A股池（默认排除科创板）"
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--dir", default=str(DEFAULT_SNAPSHOT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)
    written = update_sse_snapshots(
        args.start, args.end, snapshot_dir=Path(args.dir), progress=print
    )
    print(f"完成：写入 {len(written)} 个历史时点快照")
    if written:
        print(f"最新快照：{written[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
