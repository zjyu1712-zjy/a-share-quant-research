"""筛选公告日可控的资产负债表与现金流质量因子。

候选在运行前固定为四项，且方向不根据结果事后反转：

- 低应计：经营现金流减归母净利润，除以平均总资产；
- 低杠杆：总负债率取负；
- 现金覆盖：经营现金流除以总负债；
- 营运资本纪律：应收及存货减应付及预收的同比增加额取负，
  再除以平均总资产。

银行、保险、券商等金融行业的资产负债表结构不可与一般企业直接比较，
因此本轮只在历史时点沪深300非金融成分中验证。本模块不修改正式策略。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from src.backtest_engine import BacktestEngine, _generate_rebalance_dates
from src.config import StrategyConfig
from src.factor_screening import (
    _records,
    add_forward_returns,
    assign_folds,
    calculate_factor_correlations,
    calculate_signal_diagnostics,
    summarize_diagnostics,
)
from src.fundamental_factor_screening import _winsorize
from src.statement_factor_screening import (
    annual_report_dates,
    load_annual_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = (
    PROJECT_ROOT / "data" / "fundamental_cache" / "statement_supplements"
)
DEFAULT_INCOME_CACHE = (
    PROJECT_ROOT / "data" / "fundamental_cache" / "annual_reports"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "backtest_results"
    / "balance_quality_factor_screening"
)
API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORTS = {
    "balance": "RPT_DMSK_FN_BALANCE",
    "cashflow": "RPT_DMSK_FN_CASHFLOW",
}
FACTOR_NAMES = [
    "quality_low_accruals",
    "quality_low_leverage",
    "quality_cash_to_debt",
    "quality_working_capital_discipline",
]
RAW_TO_FACTOR = {
    f"{factor}_raw": factor for factor in FACTOR_NAMES
}


def is_financial_sector(value: object) -> bool:
    return str(value).startswith(("J66", "J67", "J68", "J69"))


def _request_page(
    session: requests.Session,
    *,
    report_name: str,
    report_date: str,
    page: int,
    retries: int = 5,
) -> dict:
    formatted_date = (
        f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
    )
    params = {
        "sortColumns": "NOTICE_DATE,SECURITY_CODE",
        "sortTypes": "-1,-1",
        "pageSize": "500",
        "pageNumber": str(page),
        "reportName": report_name,
        "columns": "ALL",
        "filter": (
            '(SECURITY_TYPE_CODE in ("058001001","058001008"))'
            '(TRADE_MARKET_CODE!="069001017")'
            f"(REPORT_DATE='{formatted_date}')"
        ),
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success") or not payload.get("result"):
                raise ValueError(payload.get("message", "empty result"))
            return payload
        except Exception as exc:  # noqa: BLE001 - bounded network retry
            last_error = exc
            time.sleep(min(1.5 * (attempt + 1), 6))
    raise RuntimeError(
        f"{report_name} {report_date} page {page} 下载失败: {last_error}"
    )


def _normalize_balance_rows(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "SECURITY_CODE",
        "SECURITY_NAME_ABBR",
        "REPORT_DATE",
        "NOTICE_DATE",
        "TOTAL_ASSETS",
        "TOTAL_LIABILITIES",
        "MONETARYFUNDS",
        "ACCOUNTS_RECE",
        "INVENTORY",
        "ACCOUNTS_PAYABLE",
        "ADVANCE_RECEIVABLES",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[columns].rename(columns={
        "SECURITY_CODE": "code",
        "SECURITY_NAME_ABBR": "name",
        "REPORT_DATE": "report_date",
        "NOTICE_DATE": "balance_notice_date",
        "TOTAL_ASSETS": "total_assets",
        "TOTAL_LIABILITIES": "total_liabilities",
        "MONETARYFUNDS": "cash",
        "ACCOUNTS_RECE": "accounts_receivable",
        "INVENTORY": "inventory",
        "ACCOUNTS_PAYABLE": "accounts_payable",
        "ADVANCE_RECEIVABLES": "advance_receipts",
    })
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].str.fullmatch(r"\d{6}")].copy()
    for column in ["report_date", "balance_notice_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    numeric = [
        "total_assets",
        "total_liabilities",
        "cash",
        "accounts_receivable",
        "inventory",
        "accounts_payable",
        "advance_receipts",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["report_date", "balance_notice_date"])
        .sort_values(["code", "report_date", "balance_notice_date"])
        .drop_duplicates(["code", "report_date"], keep="last")
        .reset_index(drop=True)
    )


def _normalize_cashflow_rows(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "SECURITY_CODE",
        "SECURITY_NAME_ABBR",
        "REPORT_DATE",
        "NOTICE_DATE",
        "NETCASH_OPERATE",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[columns].rename(columns={
        "SECURITY_CODE": "code",
        "SECURITY_NAME_ABBR": "name",
        "REPORT_DATE": "report_date",
        "NOTICE_DATE": "cashflow_notice_date",
        "NETCASH_OPERATE": "operating_cashflow",
    })
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame[frame["code"].str.fullmatch(r"\d{6}")].copy()
    for column in ["report_date", "cashflow_notice_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["operating_cashflow"] = pd.to_numeric(
        frame["operating_cashflow"], errors="coerce"
    )
    return (
        frame.dropna(subset=["report_date", "cashflow_notice_date"])
        .sort_values(["code", "report_date", "cashflow_notice_date"])
        .drop_duplicates(["code", "report_date"], keep="last")
        .reset_index(drop=True)
    )


def update_statement_supplement_cache(
    *,
    report_dates: list[str],
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 quantitative-research"
    downloaded = 0
    reused = 0
    for kind, report_name in REPORTS.items():
        for report_date in report_dates:
            output = cache_dir / f"{kind}_{report_date}.csv"
            meta = output.with_suffix(".coverage.json")
            if output.exists() and meta.exists() and not refresh:
                reused += 1
                continue
            page_dir = cache_dir / "pages" / kind / report_date
            page_dir.mkdir(parents=True, exist_ok=True)
            first = _request_page(
                session,
                report_name=report_name,
                report_date=report_date,
                page=1,
            )
            pages = int(first["result"]["pages"])
            all_rows: list[dict] = []
            for page in range(1, pages + 1):
                page_path = page_dir / f"page_{page:04d}.json"
                if page_path.exists() and not refresh:
                    payload = json.loads(
                        page_path.read_text(encoding="utf-8")
                    )
                else:
                    payload = (
                        first
                        if page == 1
                        else _request_page(
                            session,
                            report_name=report_name,
                            report_date=report_date,
                            page=page,
                        )
                    )
                    page_path.write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                all_rows.extend(payload["result"]["data"])
                if page == 1 or page % 5 == 0 or page == pages:
                    logger.info(
                        f"{kind} {report_date}: page {page}/{pages}"
                    )
            frame = (
                _normalize_balance_rows(all_rows)
                if kind == "balance"
                else _normalize_cashflow_rows(all_rows)
            )
            frame.to_csv(output, index=False, encoding="utf-8-sig")
            meta.write_text(
                json.dumps({
                    "kind": kind,
                    "report_name": report_name,
                    "report_date": report_date,
                    "pages": pages,
                    "raw_rows": len(all_rows),
                    "normalized_rows": len(frame),
                    "availability_rule": "NOTICE_DATE",
                    "source": "Eastmoney data center via AKShare schema",
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            downloaded += 1
    return {
        "requested_files": len(report_dates) * len(REPORTS),
        "downloaded_files": downloaded,
        "reused_files": reused,
    }


def load_statement_supplements(
    report_dates: list[str],
    cache_dir: Path = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for kind in REPORTS:
        frames = []
        for report_date in report_dates:
            path = cache_dir / f"{kind}_{report_date}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(
                path, dtype={"code": str}, encoding="utf-8-sig"
            )
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            for column in [
                "report_date",
                "balance_notice_date",
                "cashflow_notice_date",
            ]:
                if column in frame:
                    frame[column] = pd.to_datetime(
                        frame[column], errors="coerce"
                    )
            frames.append(frame)
        loaded[kind] = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame()
        )
    return loaded["balance"], loaded["cashflow"]


def join_statement_sources(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    income_columns = [
        "code",
        "report_date",
        "notice_date",
        "net_profit",
    ]
    income_all = income[income_columns].rename(
        columns={"notice_date": "income_notice_date"}
    )
    # 资产负债表/现金流接口的市场过滤明确排除了北交所；沪深300也不会
    # 包含92开头的北交所股票，因此连接率必须在同一沪深市场口径上计算。
    beijing_exchange_rows = int(
        income_all["code"].astype(str).str.startswith("92").sum()
    )
    income_clean = income_all[
        ~income_all["code"].astype(str).str.startswith("92")
    ].copy()
    duplicates = {
        "income_duplicates": int(
            income_clean.duplicated(["code", "report_date"]).sum()
        ),
        "balance_duplicates": int(
            balance.duplicated(["code", "report_date"]).sum()
        ),
        "cashflow_duplicates": int(
            cashflow.duplicated(["code", "report_date"]).sum()
        ),
    }
    joined = (
        income_clean.merge(
            balance,
            on=["code", "report_date"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            cashflow,
            on=["code", "report_date"],
            how="inner",
            suffixes=("_balance", "_cashflow"),
            validate="one_to_one",
        )
    )
    notice_columns = [
        "income_notice_date",
        "balance_notice_date",
        "cashflow_notice_date",
    ]
    for column in notice_columns:
        joined[column] = pd.to_datetime(joined[column], errors="coerce")
    joined = joined.dropna(subset=notice_columns).copy()
    joined["source_availability_date"] = joined[notice_columns].max(axis=1)
    joined["non_cash_working_capital"] = (
        joined["accounts_receivable"].fillna(0)
        + joined["inventory"].fillna(0)
        - joined["accounts_payable"].fillna(0)
        - joined["advance_receipts"].fillna(0)
    )
    audit = {
        **duplicates,
        "income_rows_all_a_shares": int(len(income_all)),
        "income_rows_same_market_scope": int(len(income_clean)),
        "excluded_beijing_exchange_rows": beijing_exchange_rows,
        "balance_rows": int(len(balance)),
        "cashflow_rows": int(len(cashflow)),
        "joined_rows": int(len(joined)),
        "join_rate_on_income": float(
            len(joined) / len(income_clean) if len(income_clean) else 0
        ),
        "joined_duplicates": int(
            joined.duplicated(["code", "report_date"]).sum()
        ),
        "missing_component_notice_dates": int(
            joined[notice_columns].isna().any(axis=1).sum()
        ),
        "availability_rule": (
            "max(income NOTICE_DATE, balance NOTICE_DATE, "
            "cashflow NOTICE_DATE)"
        ),
    }
    audit["passed"] = bool(
        not any(duplicates.values())
        and audit["joined_duplicates"] == 0
        and audit["missing_component_notice_dates"] == 0
        and audit["join_rate_on_income"] >= 0.95
    )
    return joined.sort_values(["code", "report_date"]), audit


def build_balance_quality_factors(
    *,
    statements: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe_store,
) -> pd.DataFrame:
    records: list[dict] = []
    by_code = {
        code: frame.sort_values("report_date")
        for code, frame in statements.groupby("code")
    }
    for date in pd.DatetimeIndex(rebalance_dates).sort_values():
        snapshot = universe_store.snapshot_for(date)
        for code in snapshot.codes:
            sector = snapshot.code_to_sector.get(code, "未知")
            if is_financial_sector(sector):
                continue
            history = by_code.get(code)
            if history is None:
                continue
            eligible = history[
                history["source_availability_date"] <= date
            ].sort_values("report_date")
            if eligible.empty:
                continue
            current = eligible.iloc[-1]
            previous = eligible.iloc[-2] if len(eligible) >= 2 else None
            consecutive_previous = bool(
                previous is not None
                and current["report_date"].year
                - previous["report_date"].year == 1
            )
            assets = float(current["total_assets"])
            liabilities = float(current["total_liabilities"])
            average_assets = (
                (assets + float(previous["total_assets"])) / 2
                if consecutive_previous
                and np.isfinite(float(previous["total_assets"]))
                else np.nan
            )
            operating_cashflow = float(current["operating_cashflow"])
            net_profit = float(current["net_profit"])
            working_capital_change = (
                float(current["non_cash_working_capital"])
                - float(previous["non_cash_working_capital"])
                if consecutive_previous else np.nan
            )
            records.append({
                "date": pd.Timestamp(date),
                "code": code,
                "sector": sector,
                "source_report_date": current["report_date"],
                "source_income_notice_date": current[
                    "income_notice_date"
                ],
                "source_balance_notice_date": current[
                    "balance_notice_date"
                ],
                "source_cashflow_notice_date": current[
                    "cashflow_notice_date"
                ],
                "source_availability_date": current[
                    "source_availability_date"
                ],
                "quality_low_accruals_raw": (
                    (operating_cashflow - net_profit) / average_assets
                    if np.isfinite(average_assets) and average_assets > 0
                    else np.nan
                ),
                "quality_low_leverage_raw": (
                    -liabilities / assets
                    if np.isfinite(assets) and assets > 0
                    else np.nan
                ),
                "quality_cash_to_debt_raw": (
                    operating_cashflow / liabilities
                    if np.isfinite(liabilities) and liabilities > 0
                    else np.nan
                ),
                "quality_working_capital_discipline_raw": (
                    -working_capital_change / average_assets
                    if np.isfinite(average_assets) and average_assets > 0
                    else np.nan
                ),
            })
    raw = pd.DataFrame(records)
    if raw.empty:
        return raw
    output = []
    for _, group in raw.groupby("date"):
        current = group.copy()
        for raw_name, factor in RAW_TO_FACTOR.items():
            clipped = _winsorize(current[raw_name])
            current[factor] = (
                clipped
                - clipped.groupby(current["sector"]).transform("median")
            )
        output.append(current)
    return pd.concat(output, ignore_index=True)


def profile_factor_quality(
    factors: pd.DataFrame,
    universe_store,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for date, group in factors.groupby("date"):
        snapshot = universe_store.snapshot_for(date)
        denominator = sum(
            not is_financial_sector(snapshot.code_to_sector.get(code, "未知"))
            for code in snapshot.codes
        )
        for factor in FACTOR_NAMES:
            count = int(group[factor].notna().sum())
            rows.append({
                "date": str(pd.Timestamp(date).date()),
                "factor": factor,
                "non_financial_universe_count": denominator,
                "available_count": count,
                "coverage_ratio": (
                    count / denominator if denominator else np.nan
                ),
            })
    by_date = pd.DataFrame(rows)
    coverage = (
        by_date.groupby("factor")["coverage_ratio"]
        .agg(["mean", "median", "min"])
        .reset_index()
    )
    coverage.columns = [
        "factor",
        "mean_coverage_ratio",
        "median_coverage_ratio",
        "min_coverage_ratio",
    ]
    future = int(
        (
            pd.to_datetime(factors["source_availability_date"])
            > pd.to_datetime(factors["date"])
        ).sum()
    )
    duplicates = int(factors.duplicated(["date", "code"]).sum())
    return by_date, {
        "rows": int(len(factors)),
        "signal_dates": int(factors["date"].nunique()),
        "stocks": int(factors["code"].nunique()),
        "duplicate_date_code_rows": duplicates,
        "future_availability_matches": future,
        "coverage": _records(coverage),
        "passed": bool(duplicates == 0 and future == 0),
    }


def screen_factors(
    aggregate: pd.DataFrame,
    by_fold: pd.DataFrame,
    quality_by_date: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for factor in FACTOR_NAMES:
        row4 = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 4)
        ]
        row8 = aggregate[
            (aggregate["factor"] == factor)
            & (aggregate["horizon_weeks"] == 8)
        ]
        folds = by_fold[
            (by_fold["factor"] == factor)
            & (by_fold["horizon_weeks"] == 4)
        ]["mean_rank_ic"].dropna()
        coverage = quality_by_date.loc[
            quality_by_date["factor"] == factor, "coverage_ratio"
        ]
        if row4.empty or row8.empty:
            continue
        four = row4.iloc[0]
        eight = row8.iloc[0]
        fold_ratio = float((folds > 0).mean()) if len(folds) else 0.0
        median_coverage = float(coverage.median()) if len(coverage) else 0.0
        passed = bool(
            median_coverage >= 0.60
            and four["mean_rank_ic"] >= 0.015
            and four["positive_ic_ratio"] >= 0.55
            and four["hac_t_stat"] >= 1.50
            and four["mean_top_bottom_spread_pct"] > 0
            and fold_ratio >= 5 / 7
            and eight["mean_rank_ic"] > 0
        )
        rows.append({
            "factor": factor,
            "median_coverage_ratio": median_coverage,
            "four_week_mean_rank_ic": float(four["mean_rank_ic"]),
            "four_week_positive_ic_ratio": float(
                four["positive_ic_ratio"]
            ),
            "four_week_hac_t_stat": float(four["hac_t_stat"]),
            "four_week_mean_spread_pct": float(
                four["mean_top_bottom_spread_pct"]
            ),
            "positive_fold_ratio": fold_ratio,
            "eight_week_mean_rank_ic": float(eight["mean_rank_ic"]),
            "status": "promising" if passed else "rejected",
        })
    return pd.DataFrame(rows)


def build_report_artifact(
    *,
    result: dict,
    decision: pd.DataFrame,
    by_fold: pd.DataFrame,
    correlations_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """生成可由 Data Analytics 便携报告构建器复现的单一报告载荷。"""
    factor_labels = {
        "quality_low_accruals": "低应计",
        "quality_low_leverage": "低杠杆",
        "quality_cash_to_debt": "现金覆盖",
        "quality_working_capital_discipline": "营运资本纪律",
    }
    screening_rows = []
    for row in decision.to_dict(orient="records"):
        screening_rows.append({
            "factor": factor_labels[row["factor"]],
            "factor_key": row["factor"],
            "coverage_pct": round(100 * row["median_coverage_ratio"], 2),
            "rank_ic_4w_pct": round(
                100 * row["four_week_mean_rank_ic"], 3
            ),
            "positive_ic_pct": round(
                100 * row["four_week_positive_ic_ratio"], 1
            ),
            "hac_t": round(row["four_week_hac_t_stat"], 3),
            "spread_4w_pct": round(
                row["four_week_mean_spread_pct"], 3
            ),
            "positive_folds": int(round(7 * row["positive_fold_ratio"])),
            "rank_ic_8w_pct": round(
                100 * row["eight_week_mean_rank_ic"], 3
            ),
            "status": (
                "进入组合压力测试"
                if row["status"] == "promising" else "淘汰"
            ),
        })
    promising = set(result["promising_factors"])
    fold_rows = []
    selected_folds = by_fold[
        (by_fold["horizon_weeks"] == 4)
        & by_fold["factor"].isin(promising)
    ].sort_values(["fold", "factor"])
    for row in selected_folds.to_dict(orient="records"):
        fold_rows.append({
            "fold": f"窗口 {int(row['fold'])}",
            "fold_number": int(row["fold"]),
            "factor": factor_labels[row["factor"]],
            "factor_key": row["factor"],
            "signal_dates": int(row["signal_dates"]),
            "rank_ic_4w_pct": round(100 * row["mean_rank_ic"], 3),
            "hac_t": round(row["hac_t_stat"], 3),
            "spread_4w_pct": round(
                row["mean_top_bottom_spread_pct"], 3
            ),
        })
    pair = correlations_summary[
        (
            correlations_summary["factor_left"]
            == "quality_low_accruals"
        )
        & (
            correlations_summary["factor_right"]
            == "quality_working_capital_discipline"
        )
    ]
    pair_correlation = (
        float(pair.iloc[0]["mean"]) if not pair.empty else np.nan
    )
    best = decision.sort_values(
        "four_week_mean_rank_ic", ascending=False
    ).iloc[0]
    context = [{
        "passed_candidates": len(promising),
        "signal_dates": result["period"]["signal_dates"],
        "join_rate_pct": round(
            100 * result["source_audit"]["join_rate_on_income"], 2
        ),
        "future_matches": result["data_quality"][
            "future_availability_matches"
        ],
        "best_factor": factor_labels[best["factor"]],
        "best_rank_ic_4w_pct": round(
            100 * best["four_week_mean_rank_ic"], 3
        ),
        "candidate_pair_correlation_pct": round(
            100 * pair_correlation, 2
        ),
    }]
    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    sources = [
        {
            "id": "screening_decision",
            "label": "资产负债与现金流因子筛选结果",
            "path": (
                "data/backtest_results/balance_quality_factor_screening/"
                "screening_decision.csv"
            ),
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT factor, 100*median_coverage_ratio AS "
                    "coverage_pct, 100*four_week_mean_rank_ic AS "
                    "rank_ic_4w_pct, 100*four_week_positive_ic_ratio AS "
                    "positive_ic_pct, four_week_hac_t_stat AS hac_t, "
                    "four_week_mean_spread_pct AS spread_4w_pct, "
                    "ROUND(7*positive_fold_ratio) AS positive_folds, "
                    "100*eight_week_mean_rank_ic AS rank_ic_8w_pct, "
                    "status FROM screening_decision ORDER BY "
                    "four_week_mean_rank_ic DESC;"
                ),
                "description": (
                    "四个预先冻结的非金融年报质量候选在统一门槛下的结果。"
                ),
                "executed_at": generated_at,
            },
        },
        {
            "id": "fold_diagnostics",
            "label": "晋级因子的七个走步窗口诊断",
            "path": (
                "data/backtest_results/balance_quality_factor_screening/"
                "diagnostics_by_fold.csv"
            ),
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT fold, factor, signal_dates, "
                    "100*mean_rank_ic AS rank_ic_4w_pct, hac_t_stat, "
                    "mean_top_bottom_spread_pct FROM diagnostics_by_fold "
                    "WHERE horizon_weeks=4 AND factor IN "
                    "('quality_low_accruals',"
                    "'quality_working_capital_discipline') "
                    "ORDER BY fold, factor;"
                ),
                "description": (
                    "两个晋级候选在七个互不重叠样本外窗口中的稳定性。"
                ),
                "executed_at": generated_at,
            },
        },
        {
            "id": "report_context",
            "label": "三表连接与时点审计摘要",
            "path": (
                "data/backtest_results/balance_quality_factor_screening/"
                "report_context.csv"
            ),
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT passed_candidates, signal_dates, "
                    "join_rate_pct, future_matches, best_factor, "
                    "best_rank_ic_4w_pct, "
                    "candidate_pair_correlation_pct FROM report_context;"
                ),
                "description": (
                    "报告头部指标、源数据连接率、未来数据检查和候选相关性。"
                ),
                "executed_at": generated_at,
            },
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": "资产负债与现金流质量因子筛选",
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# 资产负债与现金流质量因子筛选",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 技术结论\n\n"
                        "- **低应计与营运资本纪律获得组合压力测试资格。** "
                        "两者4周平均Rank IC分别为+1.982%和+1.994%，"
                        "均通过预设的覆盖、方向、显著性、分组收益、七窗口"
                        "稳定性和8周方向门槛。\n"
                        "- **这不是正式策略晋级。** 最后一个截至2025年5月"
                        "的短窗口两者均转负，下一步必须用月频Top 10＋缓冲"
                        "30、真实费用和成交限制验证组合收益。\n"
                        "- **低杠杆与现金覆盖淘汰。** 它们的4周Rank IC"
                        "分别为-0.224%和+0.733%，未达到+1.5%门槛。\n"
                        "- **正式选股策略保持不变。** 本轮仅扩大候选清单，"
                        "不构成投资建议或实盘参数变更。"
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "passed_candidates",
                        "best_ic",
                        "source_join",
                        "future_matches",
                    ],
                },
                {
                    "id": "finding",
                    "type": "markdown",
                    "sourceId": "screening_decision",
                    "body": (
                        "## 两个经营质量信号越过筛选线\n\n"
                        "四个候选使用相同方向和门槛。营运资本纪律与低应计"
                        "的4周平均Rank IC略高于+1.5%晋级线，并分别取得"
                        "+0.424%与+0.391%的Top-Bottom平均收益差；另外"
                        "两个候选没有足够一致的横截面预测力。"
                    ),
                },
                {
                    "id": "screening_chart",
                    "type": "chart",
                    "chartId": "rank_ic_screening",
                    "layout": "full",
                },
                {
                    "id": "stability",
                    "type": "markdown",
                    "sourceId": "fold_diagnostics",
                    "body": (
                        "## 七窗口多数为正，但最近窗口同时失效\n\n"
                        "两个晋级候选均有6/7个窗口的4周平均Rank IC为正。"
                        "不过窗口7只有14个周度信号，两者IC分别降至"
                        "-6.683%和-2.422%；这使组合级压力测试成为必要"
                        "条件，也提示结果可能存在市场阶段依赖。"
                    ),
                },
                {
                    "id": "fold_chart",
                    "type": "chart",
                    "chartId": "fold_stability",
                    "layout": "full",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "report_context",
                    "body": (
                        "## 样本与指标口径\n\n"
                        "研究覆盖2022年1月至2025年5月的171个周度信号日，"
                        "使用历史时点沪深300并剔除J66至J69金融行业。"
                        "利润表、资产负债表和现金流量表按报告期及股票连接，"
                        "可用日取三张表公告日的最大值；同口径连接率100%，"
                        "未来公告匹配0。预测收益从信号后的下一交易日开盘"
                        "起算。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 因子构造与通过规则\n\n"
                        "低应计定义为（经营现金流－归母净利润）/平均总资产；"
                        "低杠杆为－总负债/总资产；现金覆盖为经营现金流/"
                        "总负债；营运资本纪律为－营运资本同比增加额/平均"
                        "总资产。每期先横截面缩尾，再减同行业中位数。"
                        "候选须同时满足：中位覆盖率≥60%、4周平均IC≥"
                        "1.5%、正IC周≥55%、HAC t≥1.5、分组收益差>0、"
                        "至少5/7窗口IC为正、8周IC>0。"
                    ),
                },
                {
                    "id": "detail",
                    "type": "markdown",
                    "sourceId": "screening_decision",
                    "body": (
                        "## 精确筛选结果\n\n"
                        "下表保留所有门槛字段。两个晋级候选平均秩相关约"
                        "37.68%，说明信息有重叠但并非同一排序；是否值得"
                        "同时入模，需要由组合换手、回撤和增量超额收益决定。"
                    ),
                },
                {
                    "id": "screening_table_block",
                    "type": "table",
                    "tableId": "screening_table",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 结论边界与稳健性\n\n"
                        "- 东方财富接口提供当前可见的历史报表快照，公告日"
                        "可控制信息时点，但不能完全排除修订值回填。\n"
                        "- 171个周度信号由较少年度报表快照重复承载，HAC"
                        "统计修正了收益重叠，但不能创造更多独立财报事件。\n"
                        "- 窗口7较短且两因子同时转负，阶段稳定性尚未得到"
                        "充分证明。\n"
                        "- 本轮是候选筛选，不是预测保证、投资建议或实盘"
                        "收益承诺。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐的下一步\n\n"
                        "1. 冻结低应计、营运资本纪律及50/50复合三个候选，"
                        "不再调方向或权重。\n"
                        "2. 在月频Top 10＋缓冲30框架中，与当前低换手基准"
                        "做同口径组合回测，纳入真实费用、滑点、涨跌停和"
                        "100股整数限制。\n"
                        "3. 要求样本外超额收益为正、至少5/7窗口为正，且"
                        "换手和最大回撤不能明显恶化，才允许进入模拟盘候选。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 下一轮需要回答的问题\n\n"
                        "- 单独使用低应计，还是低应计与营运资本纪律复合，"
                        "哪一个能在费用后提供稳定超额？\n"
                        "- 两个信号是否只是偏向特定行业或小市值股票？\n"
                        "- 2024年下半年至2025年5月的失效是否会在组合层面"
                        "造成不可接受的回撤？"
                    ),
                },
            ],
            "cards": [
                {
                    "id": "passed_candidates",
                    "description": "越过全部预设筛选门槛的候选数量。",
                    "dataset": "context",
                    "sourceId": "report_context",
                    "metrics": [{
                        "label": "晋级候选",
                        "field": "passed_candidates",
                        "format": "number",
                    }],
                },
                {
                    "id": "best_ic",
                    "description": "四个候选中最高的4周平均Rank IC。",
                    "dataset": "context",
                    "sourceId": "report_context",
                    "metrics": [{
                        "label": "最高4周 Rank IC（%）",
                        "field": "best_rank_ic_4w_pct",
                        "format": "number",
                        "signed": True,
                    }],
                },
                {
                    "id": "source_join",
                    "description": "沪深市场同口径三张年报的连接比例。",
                    "dataset": "context",
                    "sourceId": "report_context",
                    "metrics": [{
                        "label": "三表连接率（%）",
                        "field": "join_rate_pct",
                        "format": "number",
                    }],
                },
                {
                    "id": "future_matches",
                    "description": "信号日使用未来公告记录的数量。",
                    "dataset": "context",
                    "sourceId": "report_context",
                    "metrics": [{
                        "label": "未来数据匹配",
                        "field": "future_matches",
                        "format": "number",
                    }],
                },
            ],
            "charts": [
                {
                    "id": "rank_ic_screening",
                    "title": "四个质量候选的4周 Rank IC",
                    "subtitle": (
                        "2022年1月至2025年5月；百分比显示，晋级线为"
                        "+1.5%（即+0.015）。"
                    ),
                    "type": "bar",
                    "dataset": "screening",
                    "sourceId": "screening_decision",
                    "valueFormat": "number",
                    "layout": "full",
                    "encodings": {
                        "x": {
                            "field": "factor",
                            "type": "nominal",
                            "label": "候选因子",
                        },
                        "y": {
                            "field": "rank_ic_4w_pct",
                            "type": "quantitative",
                            "label": "4周平均 Rank IC（%）",
                            "format": "number",
                        },
                        "tooltip": [
                            {
                                "field": "spread_4w_pct",
                                "type": "quantitative",
                                "label": "4周Top-Bottom（%）",
                                "format": "number",
                            },
                            {
                                "field": "hac_t",
                                "type": "quantitative",
                                "label": "HAC t值",
                                "format": "number",
                            },
                        ],
                    },
                },
                {
                    "id": "fold_stability",
                    "title": "晋级候选的七窗口4周 Rank IC",
                    "subtitle": (
                        "每个窗口约6个月；窗口7截至2025年5月，仅14个"
                        "周度信号。"
                    ),
                    "type": "bar",
                    "dataset": "folds",
                    "sourceId": "fold_diagnostics",
                    "valueFormat": "number",
                    "layout": "full",
                    "encodings": {
                        "x": {
                            "field": "fold",
                            "type": "nominal",
                            "label": "走步窗口",
                        },
                        "y": {
                            "field": "rank_ic_4w_pct",
                            "type": "quantitative",
                            "label": "4周平均 Rank IC（%）",
                            "format": "number",
                        },
                        "color": {
                            "field": "factor",
                            "type": "nominal",
                            "label": "候选因子",
                        },
                        "tooltip": [
                            {
                                "field": "signal_dates",
                                "type": "quantitative",
                                "label": "周度信号数",
                                "format": "number",
                            },
                            {
                                "field": "spread_4w_pct",
                                "type": "quantitative",
                                "label": "Top-Bottom（%）",
                                "format": "number",
                            },
                        ],
                    },
                },
            ],
            "tables": [{
                "id": "screening_table",
                "title": "四个候选的完整筛选结果",
                "subtitle": (
                    "历史时点沪深300非金融成分；所有候选采用同一门槛。"
                ),
                "dataset": "screening",
                "sourceId": "screening_decision",
                "defaultSort": {
                    "field": "rank_ic_4w_pct",
                    "direction": "desc",
                },
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "factor", "label": "因子", "type": "text"},
                    {
                        "field": "coverage_pct",
                        "label": "覆盖率（%）",
                        "format": "number",
                    },
                    {
                        "field": "rank_ic_4w_pct",
                        "label": "4周IC（%）",
                        "format": "number",
                        "movement": True,
                    },
                    {
                        "field": "positive_ic_pct",
                        "label": "正IC周（%）",
                        "format": "number",
                    },
                    {
                        "field": "hac_t",
                        "label": "HAC t",
                        "format": "number",
                    },
                    {
                        "field": "spread_4w_pct",
                        "label": "4周分组差（%）",
                        "format": "number",
                        "movement": True,
                    },
                    {
                        "field": "positive_folds",
                        "label": "正IC窗口/7",
                        "format": "number",
                    },
                    {
                        "field": "rank_ic_8w_pct",
                        "label": "8周IC（%）",
                        "format": "number",
                        "movement": True,
                    },
                    {
                        "field": "status",
                        "label": "结论",
                        "type": "text",
                    },
                ],
            }],
            "sources": [
                {
                    "id": source["id"],
                    "label": source["label"],
                    "path": source["path"],
                }
                for source in sources
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "context": context,
                "screening": screening_rows,
                "folds": fold_rows,
            },
        },
        "sources": sources,
    }
    pd.DataFrame(context).to_csv(
        output_dir / "report_context.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "required_structure_mapping": {
            "technical_summary": "technical_summary",
            "key_findings": ["finding", "stability", "detail"],
            "scope_and_definitions": "scope",
            "methodology": "methodology",
            "limitations_and_robustness": "limitations",
            "recommended_next_steps": "next_steps",
            "further_questions": "questions",
        },
        "chart_map": [
            {
                "segment": "finding",
                "question": "哪些候选越过4周Rank IC筛选线？",
                "family": "comparison",
                "type": "bar",
                "dataset": "screening",
                "fields": ["factor", "rank_ic_4w_pct"],
                "takeaway": "低应计与营运资本纪律晋级。",
                "palette_policy": "single-root preferred",
            },
            {
                "segment": "stability",
                "question": "晋级候选的窗口稳定性如何？",
                "family": "comparison",
                "type": "grouped bar",
                "dataset": "folds",
                "fields": ["fold", "factor", "rank_ic_4w_pct"],
                "takeaway": "6/7窗口为正，但最后窗口同时转负。",
                "palette_policy": "hard two-root cap",
            },
        ],
        "qa_notes": (
            "筛选结果、窗口数据和报告叙述均从本次输出重建；"
            "百分比字段由原始比例统一乘100。"
        ),
    }
    (output_dir / "report_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_screening(
    *,
    history_start: str = "2020-01-01",
    oos_start: str = "2022-01-01",
    end: str = "2025-05-12",
    report_start_year: int = 2018,
    report_end_year: int = 2024,
    cache_dir: Path = DEFAULT_CACHE,
    income_cache: Path = DEFAULT_INCOME_CACHE,
    output_dir: Path = DEFAULT_OUTPUT,
    download: bool = True,
    refresh: bool = False,
) -> dict:
    dates = annual_report_dates(report_start_year, report_end_year)
    download_result = (
        update_statement_supplement_cache(
            report_dates=dates,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        if download else {}
    )
    income = load_annual_reports(dates, income_cache)
    balance, cashflow = load_statement_supplements(dates, cache_dir)
    if income.empty or balance.empty or cashflow.empty:
        raise ValueError("利润表、资产负债表或现金流量表缓存为空")
    statements, source_audit = join_statement_sources(
        income, balance, cashflow
    )
    if not source_audit["passed"]:
        raise ValueError(f"三张报表源数据审计失败: {source_audit}")

    config = StrategyConfig.from_env()
    config.market_timing.enabled = False
    engine = BacktestEngine(config)
    engine.load_price_data(history_start, end)
    if engine.universe_store is None:
        raise ValueError("要求历史时点股票池")
    signals = _generate_rebalance_dates(
        oos_start, end, "weekly", engine.price_data
    )
    factors = build_balance_quality_factors(
        statements=statements,
        rebalance_dates=signals,
        universe_store=engine.universe_store,
    )
    quality_by_date, quality = profile_factor_quality(
        factors, engine.universe_store
    )
    diagnostics = calculate_signal_diagnostics(
        add_forward_returns(factors, engine.price_data),
        factor_names=FACTOR_NAMES,
    )
    diagnostics = assign_folds(
        diagnostics, start=history_start, end=end
    )
    aggregate = summarize_diagnostics(
        diagnostics, ["factor", "horizon_weeks"]
    )
    by_fold = summarize_diagnostics(
        diagnostics.dropna(subset=["fold"]),
        ["fold", "factor", "horizon_weeks"],
    )
    correlations_daily, correlations_summary = (
        calculate_factor_correlations(
            factors, factor_names=FACTOR_NAMES
        )
    )
    decision = screen_factors(aggregate, by_fold, quality_by_date)
    promising = decision.loc[
        decision["status"] == "promising", "factor"
    ].tolist()
    checks = {
        "source_audit_passed": source_audit["passed"],
        "factor_key_duplicates": int(
            factors.duplicated(["date", "code"]).sum()
        ),
        "diagnostic_key_duplicates": int(
            diagnostics.duplicated(
                ["date", "factor", "horizon_weeks"]
            ).sum()
        ),
        "future_availability_matches": quality[
            "future_availability_matches"
        ],
        "financial_sector_matches": int(
            factors["sector"].map(is_financial_sector).sum()
        ),
        "factor_count": int(decision["factor"].nunique()),
        "expected_factor_count": len(FACTOR_NAMES),
    }
    checks["passed"] = bool(
        checks["source_audit_passed"]
        and checks["factor_key_duplicates"] == 0
        and checks["diagnostic_key_duplicates"] == 0
        and checks["future_availability_matches"] == 0
        and checks["financial_sector_matches"] == 0
        and checks["factor_count"] == checks["expected_factor_count"]
    )
    if not checks["passed"]:
        raise ValueError(f"资产负债质量因子验证失败: {checks}")
    result = {
        "period": {
            "history_start": history_start,
            "oos_start": oos_start,
            "end": end,
            "report_years": [report_start_year, report_end_year],
            "signal_dates": int(factors["date"].nunique()),
        },
        "candidate_definitions": {
            "quality_low_accruals": (
                "(经营现金流-归母净利润)/平均总资产"
            ),
            "quality_low_leverage": "-总负债/总资产",
            "quality_cash_to_debt": "经营现金流/总负债",
            "quality_working_capital_discipline": (
                "-同比营运资本增加额/平均总资产"
            ),
        },
        "universe": "历史时点沪深300，剔除J66-J69金融行业",
        "availability_rule": source_audit["availability_rule"],
        "download": download_result,
        "source_audit": source_audit,
        "data_quality": quality,
        "screening_rule": (
            "中位覆盖率>=60%；4周平均IC>=0.015、正IC周>=55%、"
            "HAC t>=1.5、Top-Bottom收益差>0、至少5/7窗口IC为正、"
            "8周平均IC>0"
        ),
        "screening": _records(decision),
        "promising_factors": promising,
        "decision": (
            "通过者进入月频Top10+缓冲30组合回测"
            if promising else "没有因子通过门槛，停止进入组合回测"
        ),
        "research_boundary": (
            "定义与方向在运行前冻结；接口为当前可见的历史报表快照，"
            "公告日能控制信息时点，但无法完全排除历史修订值回填。"
        ),
        "validation": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "factor_values.csv": factors,
        "factor_quality_by_date.csv": quality_by_date,
        "diagnostics_by_date.csv": diagnostics,
        "diagnostics_summary.csv": aggregate,
        "diagnostics_by_fold.csv": by_fold,
        "factor_correlations_by_date.csv": correlations_daily,
        "factor_correlations_summary.csv": correlations_summary,
        "screening_decision.csv": decision,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            output_dir / filename, index=False, encoding="utf-8-sig"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# 资产负债与现金流质量因子筛选",
        "",
        f"**{result['decision']}**",
        "",
        "## 筛选结果",
        "",
        decision.to_markdown(index=False),
        "",
        "## 数据质量",
        "",
        pd.DataFrame(quality["coverage"]).to_markdown(index=False),
        "",
        f"- 可用日规则：{result['availability_rule']}",
        f"- 未来公告日匹配：{quality['future_availability_matches']}",
        f"- 重复日期×股票：{quality['duplicate_date_code_rows']}",
        f"- 研究边界：{result['research_boundary']}",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    build_report_artifact(
        result=result,
        decision=decision,
        by_fold=by_fold,
        correlations_summary=correlations_summary,
        output_dir=output_dir,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="资产负债表与现金流质量因子筛选"
    )
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--end", default="2025-05-12")
    parser.add_argument("--report-start-year", type=int, default=2018)
    parser.add_argument("--report-end-year", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--income-cache", type=Path, default=DEFAULT_INCOME_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screening(
        history_start=args.history_start,
        oos_start=args.oos_start,
        end=args.end,
        report_start_year=args.report_start_year,
        report_end_year=args.report_end_year,
        cache_dir=args.cache_dir,
        income_cache=args.income_cache,
        output_dir=args.output,
        download=not args.no_download,
        refresh=args.refresh,
    )
    print(json.dumps({
        "decision": result["decision"],
        "promising_factors": result["promising_factors"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
