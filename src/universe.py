"""Point-in-time stock-universe snapshots.

The backtest must never use today's constituents for an old rebalance date.
Snapshots are stored as ``data/universe/hs300/YYYY-MM-DD.csv`` where the file
name is the constituent list's effective date.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "universe" / "hs300"
REQUIRED_COLUMNS = {"code", "name", "sector"}


class UniverseSnapshotError(RuntimeError):
    """Raised when a point-in-time universe cannot be resolved safely."""


@dataclass(frozen=True)
class UniverseSnapshot:
    effective_date: pd.Timestamp
    members: pd.DataFrame
    path: Path

    @property
    def codes(self) -> set[str]:
        return set(self.members["code"].astype(str).str.zfill(6))

    @property
    def code_to_name(self) -> dict[str, str]:
        return dict(zip(self.members["code"], self.members["name"]))

    @property
    def code_to_sector(self) -> dict[str, str]:
        return dict(zip(self.members["code"], self.members["sector"]))


class PointInTimeUniverse:
    """Read and validate locally cached constituent snapshots."""

    def __init__(
        self,
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        *,
        min_members: int = 100,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.min_members = int(min_members)

    def _snapshot_paths(self) -> list[tuple[pd.Timestamp, Path]]:
        snapshots: list[tuple[pd.Timestamp, Path]] = []
        if not self.snapshot_dir.exists():
            return snapshots
        for path in self.snapshot_dir.glob("*.csv"):
            try:
                effective = pd.Timestamp(path.stem).normalize()
            except ValueError:
                continue
            snapshots.append((effective, path))
        return sorted(snapshots, key=lambda item: item[0])

    @property
    def available_dates(self) -> list[pd.Timestamp]:
        return [effective for effective, _ in self._snapshot_paths()]

    def snapshot_for(
        self,
        as_of: str | pd.Timestamp,
        *,
        strict: bool = True,
    ) -> UniverseSnapshot:
        """Return the latest snapshot effective on or before ``as_of``."""
        target = pd.Timestamp(as_of).normalize()
        eligible = [
            (effective, path)
            for effective, path in self._snapshot_paths()
            if effective <= target
        ]
        if not eligible:
            message = (
                f"{target.date()} 没有可用的历史股票池快照；"
                f"请先运行 python -m src.universe --start {target.date()}"
            )
            if strict:
                raise UniverseSnapshotError(message)
            logger.warning(message)
            return UniverseSnapshot(target, pd.DataFrame(columns=sorted(REQUIRED_COLUMNS)), Path())

        effective, path = eligible[-1]
        members = pd.read_csv(path, dtype={"code": str})
        missing = REQUIRED_COLUMNS - set(members.columns)
        if missing:
            raise UniverseSnapshotError(f"{path} 缺少字段: {sorted(missing)}")
        members["code"] = members["code"].astype(str).str.zfill(6)
        members = members.drop_duplicates("code").reset_index(drop=True)
        if len(members) < self.min_members:
            raise UniverseSnapshotError(
                f"{path} 仅有 {len(members)} 只股票，低于最小要求"
                f" {self.min_members}，拒绝用于回测"
            )
        return UniverseSnapshot(effective, members, path)

    def validate_range(self, start: str, end: str) -> None:
        """Ensure the range begins after the first cached effective snapshot."""
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError(f"起始日期晚于结束日期: {start} > {end}")
        self.snapshot_for(start, strict=True)
        self.snapshot_for(end, strict=True)

    def union_for_range(self, start: str, end: str) -> pd.DataFrame:
        """Return the union of every constituent active in the date range."""
        self.validate_range(start, end)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        dated_paths = self._snapshot_paths()
        prior = [(d, p) for d, p in dated_paths if d <= start_ts][-1:]
        in_range = [(d, p) for d, p in dated_paths if start_ts < d <= end_ts]
        selected_paths = prior + in_range
        frames = []
        for effective, path in selected_paths:
            frame = pd.read_csv(path, dtype={"code": str})
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            frame["effective_date"] = effective
            frames.append(frame)
        union = pd.concat(frames, ignore_index=True)
        # Prefer the latest known name/sector when a code occurs repeatedly.
        return (
            union.sort_values("effective_date")
            .drop_duplicates("code", keep="last")
            .reset_index(drop=True)
        )


def _rows(result) -> list[list[str]]:
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"baostock 查询失败: {result.error_code} {result.error_msg}")
    return rows


def _query_industries(bs, effective_date: str) -> dict[str, str]:
    result = bs.query_stock_industry(date=effective_date)
    rows = _rows(result)
    fields = result.fields
    code_idx = fields.index("code")
    industry_idx = fields.index("industry")
    return {
        row[code_idx].split(".")[-1].zfill(6): (row[industry_idx] or "未知")
        for row in rows
    }


def update_hs300_snapshots(
    start: str,
    end: Optional[str] = None,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
) -> list[Path]:
    """Download historical HS300 memberships and save effective snapshots.

    Baostock exposes the historical constituent ``updateDate``. Weekly query
    dates are sufficient to discover each effective list; duplicate effective
    dates are written only once.
    """
    try:
        import baostock as bs
    except ImportError as exc:
        raise RuntimeError("缺少 baostock；请先执行 pip install -r requirements.txt") from exc

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end or date.today().isoformat()).normalize()
    if start_ts > end_ts:
        raise ValueError(f"起始日期晚于结束日期: {start} > {end_ts.date()}")

    query_dates = pd.date_range(start_ts, end_ts, freq="W-FRI")
    query_dates = pd.DatetimeIndex(sorted(set(query_dates) | {start_ts, end_ts}))
    target_dir = Path(snapshot_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login.error_code} {login.error_msg}")

    discovered: dict[str, list[list[str]]] = {}
    fields: list[str] = []
    previous_signature: tuple[str, ...] = ()
    try:
        for query_date in query_dates:
            result = bs.query_hs300_stocks(query_date.strftime("%Y-%m-%d"))
            rows = _rows(result)
            if not rows:
                continue
            fields = result.fields
            update_idx = fields.index("updateDate")
            code_idx = fields.index("code")
            signature = tuple(sorted(row[code_idx] for row in rows))
            effective_date = rows[0][update_idx]
            # updateDate may refresh weekly even when membership is unchanged.
            # Persist only when the actual 300-code set changes.
            if effective_date and signature != previous_signature:
                discovered.setdefault(effective_date, rows)
                previous_signature = signature

        written: list[Path] = []
        for effective_date, rows in sorted(discovered.items()):
            output = target_dir / f"{effective_date}.csv"
            code_idx = fields.index("code")
            name_idx = fields.index("code_name")
            industries = _query_industries(bs, effective_date)
            records = []
            for row in rows:
                code = row[code_idx].split(".")[-1].zfill(6)
                records.append({
                    "code": code,
                    "name": row[name_idx],
                    "sector": industries.get(code, "未知"),
                    "source": "baostock_hs300",
                })
            frame = pd.DataFrame(records).drop_duplicates("code")
            if len(frame) != 300:
                raise UniverseSnapshotError(
                    f"{effective_date} 沪深300成分数量异常: {len(frame)}"
                )
            frame.to_csv(output, index=False, encoding="utf-8-sig")
            written.append(output)
            logger.info(f"股票池快照: {effective_date}, {len(frame)} 只 -> {output}")
        return written
    finally:
        bs.logout()


def filter_codes_for_date(
    frame: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    universe: PointInTimeUniverse,
) -> pd.DataFrame:
    """Filter a factor frame to constituents known at the rebalance date."""
    snapshot = universe.snapshot_for(rebalance_date)
    return frame[frame["code"].astype(str).str.zfill(6).isin(snapshot.codes)].copy()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="更新历史沪深300成分快照")
    parser.add_argument("--start", default="2019-01-01", help="查询起始日期")
    parser.add_argument("--end", default=None, help="查询结束日期，默认今天")
    parser.add_argument("--dir", default=str(DEFAULT_SNAPSHOT_DIR), help="快照目录")
    args = parser.parse_args(list(argv) if argv is not None else None)
    written = update_hs300_snapshots(args.start, args.end, args.dir)
    print(f"完成：发现并写入 {len(written)} 个沪深300有效快照")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
