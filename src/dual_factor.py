"""
双重因子引擎 — 桥接 SQLite 情感数据与价格数据
- 从 news_archive 查询个股情感序列
- 计算价格因子 (动量+回撤) + 情感因子 (情感动量/波动率/利好占比)
- 截面 Z-score 标准化后按权重合并
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.indicators import (
    zscore, momentum, pullback_from_high,
    sentiment_momentum, sentiment_volatility, positive_ratio,
)
from src.config import FactorConfig, ScoringConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "financial_news.db"
MY_QUANT_DIR = PROJECT_ROOT / "MyQuantAgent"


# ======================================================================
# 股票映射
# ======================================================================

def get_stock_mapping() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """一键获取三种映射。

    Returns:
        code_to_name: {"000001": "平安银行", ...}
        name_to_code: {"比亚迪": "002594", ...}
        code_to_sector: {"000001": "银行", ...}
    """
    config_path = MY_QUANT_DIR / "strategy_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    stocks = config.get("universe", {}).get("stocks", {})
    sectors = config.get("sectors", {})

    code_to_name = dict(stocks)
    code_to_sector = dict(sectors)
    name_to_code = {name: code for code, name in stocks.items()}
    return code_to_name, name_to_code, code_to_sector


def _resolve_stock_name(code: str, code_to_name: dict) -> str:
    return code_to_name.get(code, code)


def _resolve_stock_code(name: str, name_to_code: dict) -> Optional[str]:
    """模糊匹配：先精确匹配，再子串匹配。"""
    if name in name_to_code:
        return name_to_code[name]
    # 子串匹配
    for pool_name, code in name_to_code.items():
        if pool_name in name or name in pool_name:
            return code
    return None


# ======================================================================
# 情感数据提取
# ======================================================================

def load_sentiment_series(
    stock_name: str,
    start_date: str,
    end_date: str,
    db_path: Optional[str | Path] = None,
) -> pd.Series:
    """从 news_archive 查询某只股票的每日情感均分。

    Args:
        stock_name: 中文股票名称，如 "比亚迪"
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        db_path: SQLite 数据库路径

    Returns:
        pd.Series, index=date, values=avg sentiment_score (0 if no news)
    """
    import sqlite3

    path = str(db_path or DB_PATH)
    if not Path(path).exists():
        return pd.Series(dtype=float)

    conn = sqlite3.connect(path)
    like = f"%{stock_name}%"

    sql = """
        SELECT fetch_date,
               AVG(sentiment_score) AS avg_score
        FROM news_archive
        WHERE fetch_date >= ? AND fetch_date <= ?
          AND sentiment_score IS NOT NULL
          AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
          AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
          AND (title LIKE ? OR tags LIKE ? OR mentioned_stocks LIKE ?
               OR reasoning LIKE ? OR summary LIKE ?)
        GROUP BY fetch_date
        ORDER BY fetch_date ASC
    """
    params = (start_date, end_date, like, like, like, like, like)

    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["fetch_date"])
    conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["fetch_date"] = pd.to_datetime(df["fetch_date"])
    s = df.set_index("fetch_date")["avg_score"]
    s = s.sort_index()
    return s


def load_sentiment_for_pool(
    name_to_code: dict[str, str],
    start_date: str,
    end_date: str,
    db_path: Optional[str | Path] = None,
    price_data: Optional[dict[str, pd.DataFrame]] = None,
    min_sqlite_days: int = 30,
    min_pool_coverage: float = 0.60,
    allow_price_proxy: bool = False,
) -> dict[str, pd.Series]:
    """批量加载全池情感序列，并执行整池覆盖率质量检查。

    默认只接受真实新闻情感。当达到 ``min_sqlite_days`` 的股票比例低于
    ``min_pool_coverage`` 时，整池返回空数据，调用方应退化为纯价格策略。
    这样不会把“无新闻”当作中性，也不会混入由价格构造的伪情感。

    ``allow_price_proxy`` 仅保留给显式对照实验；启用时会对整池统一使用
    价格代理，不会把真实新闻和价格代理混在同一个截面里。
    """
    if not 0 <= min_pool_coverage <= 1:
        raise ValueError("min_pool_coverage 必须在 0 到 1 之间")

    real_result: dict[str, pd.Series] = {}
    for name, code in name_to_code.items():
        try:
            s = load_sentiment_series(name, start_date, end_date, db_path)
            if len(s) >= min_sqlite_days:
                real_result[code] = s
        except Exception:
            logger.debug(f"{name} 新闻情感读取失败", exc_info=True)

    total = len(name_to_code)
    coverage_ratio = len(real_result) / total if total else 0.0
    if coverage_ratio >= min_pool_coverage:
        logger.info(
            f"真实新闻情感覆盖达标: {len(real_result)}/{total} "
            f"({coverage_ratio:.1%})"
        )
        return real_result

    logger.warning(
        f"真实新闻情感覆盖不足: {len(real_result)}/{total} "
        f"({coverage_ratio:.1%} < {min_pool_coverage:.1%})；"
        "本期不使用情感因子"
    )
    if not allow_price_proxy:
        return {}

    if not price_data:
        logger.warning("已允许价格代理情感，但没有价格数据，返回空情感")
        return {}

    from src.price_sentiment import compute_daily_sentiment

    proxy_result: dict[str, pd.Series] = {}
    for code in name_to_code.values():
        if code not in price_data:
            continue
        try:
            series = compute_daily_sentiment(price_data[code]).dropna()
            if len(series) > 20:
                proxy_result[code] = series
        except Exception:
            logger.debug(f"{code} 价格代理情感计算失败", exc_info=True)
    logger.warning(
        f"显式实验模式：整池使用价格代理情感 {len(proxy_result)}/{total}；"
        "该数据不应被解释为独立新闻信息"
    )
    return proxy_result


# ======================================================================
# 因子计算
# ======================================================================

def build_price_factors(
    price_data: dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    config: FactorConfig,
) -> pd.DataFrame:
    """计算每个调仓日的价格因子（动量+回撤）。

    对每只股票、每个调仓日：
      - momentum: close.pct_change(momentum_window)[date]
      - pullback: (close - rolling_high) / rolling_high[date]

    Returns DataFrame with columns: [date, code, close, momentum, pullback]
    """
    records = []

    for code, df in price_data.items():
        if "close" not in df.columns or len(df) < config.momentum_window:
            continue

        close = df["close"].sort_index()

        mom = close.pct_change(config.momentum_window)  # type: ignore[arg-type]
        rolling_high = close.rolling(config.pullback_window).max()
        pullback = (close - rolling_high) / rolling_high

        for d in rebalance_dates:
            if d not in close.index:
                continue
            c = close[d]
            m = mom.get(d, np.nan)
            pb = pullback.get(d, np.nan)

            if pd.isna(m) or pd.isna(pb):
                continue

            records.append({
                "date": d,
                "code": code,
                "close": c,
                "momentum": m,
                "pullback": pb,
            })

    if not records:
        return pd.DataFrame(columns=["date", "code", "close", "momentum", "pullback"])

    df = pd.DataFrame(records)
    logger.info(f"价格因子: {len(df)} 条 ({df['code'].nunique()} 只股票, {df['date'].nunique()} 个调仓日)")
    return df


def _fill_sentiment_daily(
    sentiment_series: pd.Series,
    date_index: pd.DatetimeIndex,
    max_gap: int = 5,
) -> pd.Series:
    """将稀疏的情感序列对齐到每日索引，前向填充缺失值。

    max_gap 控制最长填充天数：DAILY/月频用 5 天，WEEKLY/周频用 10 天。
    """
    s = sentiment_series.reindex(date_index)
    s = s.ffill(limit=max_gap)
    return s


def build_sentiment_factors(
    sentiment_data: dict[str, pd.Series],
    rebalance_dates: pd.DatetimeIndex,
    price_data: dict[str, pd.DataFrame],
    config: FactorConfig,
    rebalance_freq: str = "monthly",
) -> pd.DataFrame:
    """计算每个调仓日的情感因子。

    对每只股票，先将情感序列对齐到该股票价格数据的日期索引，
    再计算滚动指标，最后取调仓日的值。

    周频调仓时，前向填充容差扩大到 10 天以覆盖数据稀疏期。

    Returns DataFrame with columns: [date, code, sent_momentum, sent_volatility, positive_ratio]
    """
    records = []
    sent_ma = config.sentiment_ma_window

    # 周频用更大的填充容差
    fill_limit = 10 if rebalance_freq == "weekly" else 5

    for code, sent_series in sentiment_data.items():
        if len(sent_series) < sent_ma:
            continue

        # 对齐到该股票价格数据的日期索引
        if code not in price_data:
            continue
        price_index = price_data[code].index.sort_values()
        if len(price_index) < sent_ma:
            continue

        daily_sent = _fill_sentiment_daily(sent_series, price_index, max_gap=fill_limit)
        if daily_sent.dropna().empty:
            continue

        s_mom = sentiment_momentum(daily_sent, window=sent_ma)
        s_vol = sentiment_volatility(daily_sent, window=sent_ma)
        p_ratio = positive_ratio(daily_sent, window=sent_ma)

        for d in rebalance_dates:
            if d not in daily_sent.index:
                continue
            sm = s_mom.get(d, np.nan)
            sv = s_vol.get(d, np.nan)
            pr = p_ratio.get(d, np.nan)

            # 至少要有 sentiment_momentum 才算有效
            if pd.isna(sm):
                continue

            records.append({
                "date": d,
                "code": code,
                "sent_momentum": sm if not pd.isna(sm) else 0.0,
                "sent_volatility": sv if not pd.isna(sv) else 0.0,
                "positive_ratio": pr if not pd.isna(pr) else 0.0,
            })

    if not records:
        return pd.DataFrame(columns=["date", "code", "sent_momentum", "sent_volatility", "positive_ratio"])

    df = pd.DataFrame(records)
    logger.info(f"情感因子: {len(df)} 条 ({df['code'].nunique()} 只股票, {df['date'].nunique()} 个调仓日)")
    return df


def build_weekly_sentiment_factors(
    name_to_code: dict[str, str],
    start_date: str,
    end_date: str,
    db_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """从日频新闻情感数据构建周线情感因子。

    对每只股票，将日度情感序列按周一为周起始聚合，每周输出三个因子：
      - week_mean:       情感分周均值（情绪强度）
      - week_std:        情感分周标准差（分歧度）
      - positive_ratio:  利好占比
      - news_count:      该周匹配到的新闻总数

    Args:
        name_to_code: {"比亚迪": "002594", ...}
        start_date, end_date: "YYYY-MM-DD"
        db_path: 数据库路径

    Returns:
        DataFrame, columns: [week_start, code, week_mean, week_std,
                             positive_ratio, news_count]
    """
    records = []

    for name, code in name_to_code.items():
        daily = load_sentiment_series(name, start_date, end_date, db_path)
        if daily.empty:
            continue

        # 按周一为周起始聚合
        weekly = daily.resample("W-MON").agg(
            week_mean="mean",
            week_std=lambda x: x.std() if len(x) >= 2 else 0.0,
            positive_ratio=lambda x: (x > 0.1).mean(),
            news_count="count",
        )

        for week_start, row in weekly.iterrows():
            if pd.isna(row["week_mean"]):
                continue
            records.append({
                "week_start": week_start,
                "code": code,
                "week_mean": round(row["week_mean"], 4),
                "week_std": round(row["week_std"], 4),
                "positive_ratio": round(row["positive_ratio"], 4),
                "news_count": int(row["news_count"]),
            })

    if not records:
        return pd.DataFrame(
            columns=["week_start", "code", "week_mean",
                     "week_std", "positive_ratio", "news_count"]
        )

    df = pd.DataFrame(records)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["week_start", "code"]).reset_index(drop=True)
    logger.info(
        f"周线情感因子: {len(df)} 条 ({df['code'].nunique()} 只股票, "
        f"{df['week_start'].nunique()} 周, "
        f"覆盖 {df['news_count'].sum()} 条新闻)"
    )
    return df


def build_combined_factors(
    price_factors: pd.DataFrame,
    sentiment_factors: Optional[pd.DataFrame],
    scoring_config: ScoringConfig,
) -> pd.DataFrame:
    """合并价格因子与情感因子，截面 Z-score 后加权求和。

    score = score_price * 1.0 + score_sentiment * sentiment_weight

    Returns DataFrame with columns: [date, code, close, score, score_price,
    score_sentiment, momentum, pullback, sent_momentum, ...]
    """
    pf = price_factors.copy()

    # --- 价格子得分：截面 zscore 等权加总 ---
    for col in ["momentum", "pullback"]:
        pf[f"z_{col}"] = pf.groupby("date")[col].transform(lambda x: zscore(x))

    pf["score_price"] = (
        pf["z_momentum"] * scoring_config.momentum_weight
        + pf["z_pullback"] * scoring_config.pullback_weight
    )

    # --- 情感子得分 ---
    if sentiment_factors is not None and len(sentiment_factors) > 0:
        sf = sentiment_factors.copy()

        # 情感子权重
        s_mom_w = 0.5
        s_vol_w = -0.25
        p_ratio_w = 0.25

        for col in ["sent_momentum", "sent_volatility", "positive_ratio"]:
            if col in sf.columns:
                sf[f"z_{col}"] = sf.groupby("date")[col].transform(lambda x: zscore(x))

        sf["score_sentiment"] = (
            sf["z_sent_momentum"] * s_mom_w
            + sf["z_sent_volatility"] * s_vol_w
            + sf["z_positive_ratio"] * p_ratio_w
        )

        # 从 pf 中移除 score_sentiment 预置列（否则 merge 后缀冲突）
        if "score_sentiment" in pf.columns:
            pf.drop(columns=["score_sentiment"], inplace=True)

        # 合并回 price_factors
        sent_cols = ["date", "code", "score_sentiment",
                     "sent_momentum", "sent_volatility", "positive_ratio"]
        sent_cols = [c for c in sent_cols if c in sf.columns]
        if "z_sent_momentum" in sf.columns:
            sent_cols.append("z_sent_momentum")

        pf = pf.merge(
            sf[sent_cols],
            on=["date", "code"],
            how="left",
        )
    else:
        pf["score_sentiment"] = 0.0

    # 确保列存在 + 填充缺失
    if "score_sentiment" not in pf.columns:
        pf["score_sentiment"] = 0.0
    else:
        pf["score_sentiment"] = pf["score_sentiment"].fillna(0.0)

    for c in ["sent_momentum", "sent_volatility", "positive_ratio"]:
        if c not in pf.columns:
            pf[c] = 0.0
        else:
            pf[c] = pf[c].fillna(0.0)

    pf["score"] = pf["score_price"] + pf["score_sentiment"] * scoring_config.sentiment_weight

    # 排序
    pf = pf.sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)

    sent_cov = (pf["score_sentiment"] != 0).mean() * 100
    logger.info(f"综合因子: {len(pf)} 条, 情感覆盖率 {sent_cov:.1f}%")
    return pf


# ======================================================================
# 便捷入口
# ======================================================================

if __name__ == "__main__":
    _, name_to_code, _ = get_stock_mapping()
    print(f"股票池: {len(name_to_code)} 只")

    # 测试情感序列
    test_name = "比亚迪"
    test_code = name_to_code.get(test_name, "?")
    print(f"\n{test_name} ({test_code}) 情感序列 (近30天):")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    s = load_sentiment_series(test_name, start, end)
    if len(s) > 0:
        print(s.tail(10).to_string())
    else:
        print("  无情感数据（需先运行 main.py 采集新闻）")
