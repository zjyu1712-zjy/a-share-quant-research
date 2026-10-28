"""Audited, resumable data layer for v2 independent-factor research."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "v2_factor_cache"
DEFAULT_PRICE_DIR = DEFAULT_CACHE_ROOT / "prices"
DEFAULT_VALUATION_DIR = DEFAULT_CACHE_ROOT / "valuation"
DEFAULT_AUDIT_DIR = DEFAULT_CACHE_ROOT / "audit"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "v2_independent_factor_screening"
)
DEFAULT_SNAPSHOT_DIR = (
    PROJECT_ROOT / "data" / "universe" / "sse_composite_a"
)

FACTOR_NAMES = [
    "low_volatility_60",
    "sector_neutral_book_to_price_mrq",
    "momentum_12_1",
]
PRICE_FIELDS = [
    "date", "code", "open", "high", "low", "close",
    "volume", "amount", "tradestatus",
]
VALUATION_FIELDS = ["date", "code", "pbMRQ"]
FIXED_WINDOWS = [
    {"id": 1, "start": "2022-01-01", "end": "2022-06-30"},
    {"id": 2, "start": "2022-07-01", "end": "2022-12-31"},
    {"id": 3, "start": "2023-01-01", "end": "2023-06-30"},
    {"id": 4, "start": "2023-07-01", "end": "2023-12-31"},
    {"id": 5, "start": "2024-01-01", "end": "2024-06-30"},
    {"id": 6, "start": "2024-07-01", "end": "2024-12-31"},
    {"id": 7, "start": "2025-01-01", "end": "2025-05-12"},
]


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_path(data_path: Path) -> Path:
    return Path(data_path).with_suffix(".meta.json")


def write_cache_pair(
    frame: pd.DataFrame,
    data_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist a CSV and its integrity metadata."""
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _metadata_path(path)
    token = f"{os.getpid()}.{id(frame)}"
    temp_data = path.with_name(f".{path.name}.{token}.tmp")
    temp_meta = meta_path.with_name(f".{meta_path.name}.{token}.tmp")
    try:
        frame.to_csv(temp_data, index=False, encoding="utf-8-sig")
        completed = dict(metadata)
        dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
        completed.update({
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "actual_start": str(dates.min().date()) if len(dates) else None,
            "actual_end": str(dates.max().date()) if len(dates) else None,
            "data_sha256": _file_sha256(temp_data),
        })
        temp_meta.write_text(
            json.dumps(completed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_data.replace(path)
        temp_meta.replace(meta_path)
        return completed
    finally:
        for temporary in (temp_data, temp_meta):
            if temporary.exists():
                temporary.unlink()


def cache_pair_valid(
    data_path: Path,
    *,
    code: str,
    start: str,
    end: str,
    kind: str,
) -> bool:
    path = Path(data_path)
    meta_path = _metadata_path(path)
    if not path.exists() or not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_columns = PRICE_FIELDS if kind == "price" else VALUATION_FIELDS
        if int(metadata.get("schema_version", 0)) != 1:
            return False
        if metadata.get("kind") != kind:
            return False
        if str(metadata.get("code", "")).zfill(6) != str(code).zfill(6):
            return False
        if metadata.get("source") != "baostock_query_history_k_data_plus":
            return False
        if pd.Timestamp(metadata["requested_start"]) > pd.Timestamp(start):
            return False
        if pd.Timestamp(metadata["requested_end"]) < pd.Timestamp(end):
            return False
        if metadata.get("columns") != expected_columns:
            return False
        if kind == "price" and (
            metadata.get("adjustflag") != "2"
            or metadata.get("amount_source") != "baostock_reported"
        ):
            return False
        if metadata.get("data_sha256") != _file_sha256(path):
            return False
        frame = pd.read_csv(path)
        if list(frame.columns) != expected_columns:
            return False
        if int(metadata.get("rows", -1)) != len(frame):
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def discover_universe_union(snapshot_dir: Path) -> list[str]:
    """Return the sorted code union from all point-in-time snapshots."""
    codes: set[str] = set()
    for path in sorted(Path(snapshot_dir).glob("*.csv")):
        frame = pd.read_csv(path, dtype={"code": str})
        if "code" not in frame.columns:
            raise ValueError(f"股票池快照缺少code字段: {path}")
        codes.update(frame["code"].dropna().astype(str).str.zfill(6))
    if not codes:
        raise ValueError(f"未发现历史股票池快照: {snapshot_dir}")
    return sorted(codes)


def bao_code(code: str) -> str:
    normalized = str(code).split(".")[-1].zfill(6)
    prefix = "sh" if normalized.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}.{normalized}"


def _normalize_rows(
    rows: list[list[str]],
    fields: list[str],
    *,
    code: str,
    expected_fields: list[str],
    numeric_fields: list[str],
) -> pd.DataFrame:
    missing = set(expected_fields) - set(fields)
    if missing:
        raise ValueError(f"数据源缺少字段: {sorted(missing)}")
    frame = pd.DataFrame(rows, columns=fields)[expected_fields].copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["code"] = str(code).split(".")[-1].zfill(6)
    for column in numeric_fields:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def normalize_price_rows(
    rows: list[list[str]], fields: list[str], code: str
) -> pd.DataFrame:
    return _normalize_rows(
        rows,
        fields,
        code=code,
        expected_fields=PRICE_FIELDS,
        numeric_fields=[
            "open", "high", "low", "close", "volume", "amount",
            "tradestatus",
        ],
    )


def normalize_valuation_rows(
    rows: list[list[str]], fields: list[str], code: str
) -> pd.DataFrame:
    return _normalize_rows(
        rows,
        fields,
        code=code,
        expected_fields=VALUATION_FIELDS,
        numeric_fields=["pbMRQ"],
    )


def _result_rows(result: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(
            f"BaoStock查询失败: {result.error_code} {result.error_msg}"
        )
    return rows


class BaoStockFactorAdapter:
    """One authenticated BaoStock connection for a resumable batch."""

    def __init__(self):
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError("缺少baostock，请先运行setup_mac.command") from exc
        self.bs = bs
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(
                f"BaoStock登录失败: {login.error_code} {login.error_msg}"
            )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.bs.logout()
            self._closed = True

    def __enter__(self) -> "BaoStockFactorAdapter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def query_price(self, code: str, start: str, end: str) -> pd.DataFrame:
        result = self.bs.query_history_k_data_plus(
            bao_code(code),
            ",".join(PRICE_FIELDS),
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
        )
        return normalize_price_rows(
            _result_rows(result), result.fields or PRICE_FIELDS, code
        )

    def query_valuation(
        self, code: str, start: str, end: str
    ) -> pd.DataFrame:
        result = self.bs.query_history_k_data_plus(
            bao_code(code),
            ",".join(VALUATION_FIELDS),
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )
        return normalize_valuation_rows(
            _result_rows(result), result.fields or VALUATION_FIELDS, code
        )


def _price_cache_path(directory: Path, code: str) -> Path:
    return Path(directory) / f"price_{str(code).zfill(6)}.csv"


def _valuation_cache_path(directory: Path, code: str) -> Path:
    return Path(directory) / f"valuation_{str(code).zfill(6)}.csv"


def _price_metadata(code: str, start: str, end: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "price",
        "code": str(code).zfill(6),
        "source": "baostock_query_history_k_data_plus",
        "adjustflag": "2",
        "requested_start": start,
        "requested_end": end,
        "amount_source": "baostock_reported",
    }


def _valuation_metadata(code: str, start: str, end: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "valuation",
        "code": str(code).zfill(6),
        "source": "baostock_query_history_k_data_plus",
        "adjustflag": "3",
        "requested_start": start,
        "requested_end": end,
        "amount_source": None,
    }


def _prior_no_data_codes(audit_dir: Path) -> set[str]:
    path = Path(audit_dir) / "download_status.csv"
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, dtype={"code": str})
    except (OSError, ValueError):
        return set()
    if not {"code", "status"}.issubset(frame.columns):
        return set()
    return set(
        frame.loc[frame["status"] == "no_historical_data", "code"]
        .astype(str)
        .str.zfill(6)
    )


def update_factor_cache(
    *,
    codes: list[str],
    start: str,
    end: str,
    price_dir: Path,
    valuation_dir: Path,
    audit_dir: Path,
    adapter: Optional[Any] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Download each stock independently and preserve every completed pair."""
    normalized = sorted({str(code).zfill(6) for code in codes})
    price_dir = Path(price_dir)
    valuation_dir = Path(valuation_dir)
    audit_dir = Path(audit_dir)
    for directory in (price_dir, valuation_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    owns_adapter = adapter is None
    source = adapter or BaoStockFactorAdapter()
    known_no_data = _prior_no_data_codes(audit_dir)
    rows: list[dict[str, Any]] = []
    try:
        for index, code in enumerate(normalized, start=1):
            price_path = _price_cache_path(price_dir, code)
            valuation_path = _valuation_cache_path(valuation_dir, code)
            price_valid = cache_pair_valid(
                price_path, code=code, start=start, end=end, kind="price"
            )
            valuation_valid = cache_pair_valid(
                valuation_path,
                code=code,
                start=start,
                end=end,
                kind="valuation",
            )
            status = "complete"
            error = ""
            downloaded_price = False
            downloaded_valuation = False
            if price_valid and valuation_valid:
                status = "skipped"
            elif code in known_no_data and not price_valid:
                status = "no_historical_data"
            else:
                try:
                    if not price_valid:
                        price_frame = source.query_price(code, start, end)
                        if price_frame.empty:
                            status = "no_historical_data"
                        else:
                            write_cache_pair(
                                price_frame,
                                price_path,
                                _price_metadata(code, start, end),
                            )
                            downloaded_price = True
                            price_valid = True
                    if status != "no_historical_data" and not valuation_valid:
                        valuation_frame = source.query_valuation(
                            code, start, end
                        )
                        write_cache_pair(
                            valuation_frame,
                            valuation_path,
                            _valuation_metadata(code, start, end),
                        )
                        downloaded_valuation = True
                        valuation_valid = True
                    if status != "no_historical_data" and not (
                        price_valid and valuation_valid
                    ):
                        status = "failed"
                        error = "cache_validation_failed_after_download"
                except Exception as exc:  # preserve successful earlier codes
                    status = "failed"
                    error = str(exc)
            rows.append({
                "code": code,
                "status": status,
                "price_valid": bool(price_valid),
                "valuation_valid": bool(valuation_valid),
                "downloaded_price": downloaded_price,
                "downloaded_valuation": downloaded_valuation,
                "error": error,
            })
            if progress and (
                index == 1 or index % 25 == 0 or index == len(normalized)
            ):
                progress(
                    f"[{index}/{len(normalized)}] {code}: {status}"
                )
    finally:
        if owns_adapter and hasattr(source, "close"):
            source.close()

    status_frame = pd.DataFrame(rows)
    status_frame.to_csv(
        audit_dir / "download_status.csv", index=False, encoding="utf-8-sig"
    )
    failures = status_frame[status_frame["status"] == "failed"]
    failures.to_csv(
        audit_dir / "download_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "requested": len(normalized),
        "complete": int((status_frame["status"] == "complete").sum()),
        "skipped": int((status_frame["status"] == "skipped").sum()),
        "no_historical_data": int(
            (status_frame["status"] == "no_historical_data").sum()
        ),
        "failed": int((status_frame["status"] == "failed").sum()),
    }
    (audit_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def audit_factor_cache(
    *,
    codes: list[str],
    start: str,
    end: str,
    price_dir: Path,
    valuation_dir: Path,
    audit_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit every expected code; a valid subset can never be 'ready'."""
    normalized = sorted({str(code).zfill(6) for code in codes})
    status_path = Path(audit_dir) / "download_status.csv"
    status_map: dict[str, str] = {}
    if status_path.exists():
        try:
            status_frame = pd.read_csv(status_path, dtype={"code": str})
            if {"code", "status"}.issubset(status_frame.columns):
                status_map = {
                    str(code).zfill(6): str(status)
                    for code, status in zip(
                        status_frame["code"], status_frame["status"]
                    )
                }
        except (OSError, ValueError):
            status_map = {}

    rows: list[dict[str, Any]] = []
    for code in normalized:
        price_path = _price_cache_path(Path(price_dir), code)
        valuation_path = _valuation_cache_path(Path(valuation_dir), code)
        price_valid = cache_pair_valid(
            price_path, code=code, start=start, end=end, kind="price"
        )
        valuation_valid = cache_pair_valid(
            valuation_path,
            code=code,
            start=start,
            end=end,
            kind="valuation",
        )
        recorded_status = status_map.get(code, "")
        has_any_artifact = any((
            price_path.exists(),
            _metadata_path(price_path).exists(),
            valuation_path.exists(),
            _metadata_path(valuation_path).exists(),
        ))
        attempted = bool(recorded_status or has_any_artifact)
        complete = bool(price_valid and valuation_valid)
        no_data = recorded_status == "no_historical_data"
        failed = bool(
            recorded_status == "failed"
            or (attempted and not complete and not no_data)
        )
        if complete:
            audit_status = "complete"
        elif no_data:
            audit_status = "no_historical_data"
        elif failed:
            audit_status = "failed"
        else:
            audit_status = "unattempted"
        rows.append({
            "code": code,
            "attempted": attempted,
            "price_valid": price_valid,
            "valuation_valid": valuation_valid,
            "status": audit_status,
        })

    frame = pd.DataFrame(rows)
    complete_count = int((frame["status"] == "complete").sum())
    no_data_count = int(
        (frame["status"] == "no_historical_data").sum()
    )
    failed_count = int((frame["status"] == "failed").sum())
    unattempted_count = int((frame["status"] == "unattempted").sum())
    summary = {
        "expected_codes": len(normalized),
        "attempted_codes": int(frame["attempted"].sum()),
        "valid_price_codes": int(frame["price_valid"].sum()),
        "valid_valuation_codes": int(frame["valuation_valid"].sum()),
        "complete_codes": complete_count,
        "no_historical_data_codes": no_data_count,
        "failed_codes": failed_count,
        "unattempted_codes": unattempted_count,
        "ready_for_screening": bool(
            complete_count + no_data_count == len(normalized)
            and failed_count == 0
            and unattempted_count == 0
        ),
    }
    return frame, summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载并审计v2独立因子所需的全量历史数据"
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)

    protocol = build_factor_protocol()
    codes = discover_universe_union(Path(args.snapshot_dir))
    expected = int(protocol["universe"]["expected_union_codes"])
    if len(codes) != expected:
        print(f"数据准备未开始：股票池并集{len(codes)}只，预期{expected}只")
        return 2
    cache_root = Path(args.cache_root)
    update = update_factor_cache(
        codes=codes,
        start=args.start,
        end=args.end,
        price_dir=cache_root / "prices",
        valuation_dir=cache_root / "valuation",
        audit_dir=cache_root / "audit",
        progress=print,
    )
    audit, summary = audit_factor_cache(
        codes=codes,
        start=args.start,
        end=args.end,
        price_dir=cache_root / "prices",
        valuation_dir=cache_root / "valuation",
        audit_dir=cache_root / "audit",
    )
    audit.to_csv(
        cache_root / "audit" / "readiness_by_code.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_payload = {
        **summary,
        "new_complete": update["complete"],
        "skipped": update["skipped"],
    }
    (cache_root / "audit" / "readiness_summary.json").write_text(
        json.dumps(_write_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "v2因子数据："
        f"完整{summary['complete_codes']}只，"
        f"无历史数据{summary['no_historical_data_codes']}只，"
        f"失败{summary['failed_codes']}只，"
        f"未尝试{summary['unattempted_codes']}只"
    )
    print(
        "数据状态："
        f"{'ready_for_screening' if summary['ready_for_screening'] else 'not_ready'}"
    )
    return 0 if summary["ready_for_screening"] else 2


def build_factor_protocol() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "strategy_id": "v2_monthly_top10_dd10",
        "stage_id": "v2_independent_factor_screening",
        "research_status": "factor_admission_only",
        "deployment_authorized": False,
        "cache_window": {"start": "2019-01-01", "end": "2025-05-12"},
        "research_window": {"start": "2020-01-01", "end": "2025-05-12"},
        "universe": {
            "source": "sse_composite_a_pit",
            "snapshot_dir": "data/universe/sse_composite_a",
            "expected_snapshots": 65,
            "expected_union_codes": 1648,
            "include_star_market": False,
            "strict_point_in_time": True,
            "minimum_20d_average_amount": 50_000_000.0,
            "minimum_cross_section": 100,
        },
        "cache_paths": {
            "prices": "data/v2_factor_cache/prices",
            "valuation": "data/v2_factor_cache/valuation",
            "audit": "data/v2_factor_cache/audit",
        },
        "data_sources": {
            "price": {
                "provider": "baostock",
                "api": "query_history_k_data_plus",
                "adjustflag": "2",
                "amount_source": "baostock_reported",
            },
            "valuation": {
                "provider": "baostock",
                "api": "query_history_k_data_plus",
                "field": "pbMRQ",
            },
        },
        "signal_frequency": "monthly_snapshot_dates",
        "forward_return": "next_session_open_to_20_40_60th_later_open",
        "horizons_sessions": {"4w": 20, "8w": 40, "12w": 60},
        "factors": {
            "low_volatility_60": {
                "formula": "-std(daily_close_return,60)*sqrt(252)",
                "direction": "higher_is_better",
            },
            "sector_neutral_book_to_price_mrq": {
                "formula": "winsorize(1/pbMRQ,2.5%,97.5%)-sector_median",
                "direction": "higher_is_better",
                "positive_pb_only": True,
            },
            "momentum_12_1": {
                "formula": "P(t-20)/P(t-252)-1",
                "direction": "higher_is_better",
            },
        },
        "fixed_windows": FIXED_WINDOWS,
        "gates": {
            "median_coverage_ratio": 0.90,
            "minimum_full_year_median_coverage_ratio": 0.85,
            "four_week_mean_rank_ic": 0.020,
            "four_week_median_rank_ic_strictly_positive": True,
            "four_week_positive_ic_ratio": 0.55,
            "four_week_hac_t_stat": 2.50,
            "four_week_one_sided_p_value": 0.01,
            "four_week_fdr_q_value": 0.10,
            "annualized_gross_top_bottom_spread_pct": 4.0,
            "aggregate_quintile_monotonic_corr": 0.80,
            "correct_adjacent_quintile_steps": 3,
            "annualized_icir": 0.50,
            "positive_full_years_of_four": 3,
            "positive_fixed_windows_of_seven": 5,
            "eight_week_mean_rank_ic_strictly_positive": True,
            "max_monthly_top_quintile_turnover": 0.70,
        },
        "decision_boundary": (
            "通过仅可进入组合回测；不得生成订单、买入清单、模拟盘或实盘授权。"
        ),
    }
    payload["protocol_hash"] = canonical_sha256(payload)
    return payload


def ensure_factor_protocol(output_dir: Path) -> tuple[Path, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "research_protocol.json"
    expected = build_factor_protocol()
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"已有研究协议无法解析: {path}") from exc
        if existing != expected:
            raise ValueError("已有v2因子研究协议与冻结协议不一致，拒绝覆盖")
    else:
        path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return path, str(expected["protocol_hash"])


if __name__ == "__main__":
    raise SystemExit(main())
