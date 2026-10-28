"""
价格数据获取模块
- 通过 akshare 获取 A 股 / 指数日线数据，三源自动切换
- CSV 缓存（含日期覆盖检查）
- 股票池加载（从 MyQuantAgent/strategy_config.json）
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ---- paths ----

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_CACHE_DIR = PROJECT_ROOT / "data" / "price_cache"
MY_QUANT_DIR = PROJECT_ROOT / "MyQuantAgent"
LEGACY_DATA_DIR = MY_QUANT_DIR / "data"


def _symbol_to_tx(symbol: str) -> str:
    """纯数字 symbol → 腾讯格式 sz000001 / sh600000"""
    if symbol.startswith(("sh", "sz")):
        return symbol
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    return f"sh{symbol}"


def _to_yyyymmdd(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD"""
    return date_str.replace("-", "")


# ======================================================================
# 标准化
# ======================================================================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 akshare 返回的中文列名标准化为英文，保留 OHLCV 和成交额。"""
    rename_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "date" not in df.columns:
        raise ValueError(f"akshare 返回数据缺少 date 列，实际列: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if "volume" not in df.columns:
        df["volume"] = 0
    if "amount" in df.columns:
        amount_source = "reported"
    else:
        df["amount"] = df["close"] * df["volume"]
        amount_source = "estimated_close_times_volume"

    result = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    result.attrs["amount_source"] = amount_source
    return result


# ======================================================================
# 数据获取（三源回退）
# ======================================================================

def fetch_stock_daily(
    symbol: str,
    start_date: str = "20100101",
    end_date: str = "20251231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """获取 A 股个股日线数据。主源: 腾讯 → 东方财富 → 新浪。"""
    start_ts = _to_yyyymmdd(start_date) if "-" in start_date else start_date
    end_ts = _to_yyyymmdd(end_date) if "-" in end_date else end_date
    tx_symbol = _symbol_to_tx(symbol)

    # 方式1: 腾讯源
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist_tx(
            symbol=tx_symbol,
            start_date=start_ts,
            end_date=end_ts,
            adjust=adjust,
        )
        if len(df) > 0:
            result = _standardize_columns(df)
            result.attrs.update(
                source="akshare_stock_zh_a_hist_tx",
                adjust=adjust or "none",
            )
            return result
    except Exception as e:
        logger.warning(f"[腾讯源] {symbol} 失败: {e}")

    # 方式2: 东方财富
    for attempt in range(2):
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                     start_date=start_ts, end_date=end_ts, adjust=adjust)
            if len(df) > 0:
                result = _standardize_columns(df)
                result.attrs.update(
                    source="akshare_stock_zh_a_hist_eastmoney",
                    adjust=adjust or "none",
                )
                return result
        except Exception as e:
            logger.warning(f"[东方财富] {symbol} 尝试{attempt+1}/2 失败: {e}")
            if attempt < 1:
                time.sleep(2)

    # 方式3: 新浪源
    try:
        import akshare as ak
        sx = f"sz{symbol}" if symbol.startswith(("0", "3")) else f"sh{symbol}"
        df = ak.stock_zh_a_daily(symbol=sx, start_date=start_ts, end_date=end_ts, adjust=adjust)
        if len(df) > 0:
            result = _standardize_columns(df)
            result.attrs.update(
                source="akshare_stock_zh_a_daily_sina",
                adjust=adjust or "none",
            )
            return result
    except Exception as e:
        logger.warning(f"[新浪源] {symbol} 失败: {e}")

    raise RuntimeError(f"无法获取 {symbol} 的行情数据（三源均失败），请检查网络与 akshare 版本。")


def fetch_index_daily(
    symbol: str,
    start_date: str = "20100101",
    end_date: str = "20251231",
) -> pd.DataFrame:
    """获取指数日线数据（如沪深300 = 000300）。"""
    import akshare as ak

    prefix = "sh" if symbol.startswith("000") else "sz"
    df = ak.stock_zh_index_daily(symbol=f"{prefix}{symbol}")
    df = _standardize_columns(df)
    df.attrs.update(
        source="akshare_stock_zh_index_daily",
        adjust="none",
    )

    # 按日期范围筛选（akshare 指数接口不接收日期参数）
    start_ts = _to_yyyymmdd(start_date) if "-" in start_date else start_date
    end_ts = _to_yyyymmdd(end_date) if "-" in end_date else end_date
    start_dt = pd.Timestamp(start_ts)
    end_dt = pd.Timestamp(end_ts)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


# ======================================================================
# 缓存层
# ======================================================================

def _get_cache_path(symbol: str, prefix: str = "stock") -> Path:
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PRICE_CACHE_DIR / f"{prefix}_{symbol}.csv"


def _coverage_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".coverage.json")


def _check_cache_coverage(
    cache_path: Path,
    start: str,
    end: str,
    *,
    required_adjust: str | None = None,
    require_amount: bool = False,
) -> bool:
    """检查缓存日期及来源元数据是否满足请求。

    schema v2以前的缓存没有记录下载源和复权口径，不能安全用于长期价格
    因子，因此会触发一次完整刷新。
    """
    if not cache_path.exists():
        return False
    try:
        meta_path = _coverage_meta_path(cache_path)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if int(meta.get("schema_version", 0)) < 2:
                return False
            if require_amount and (
                int(meta.get("schema_version", 0)) < 3
                or not meta.get("amount_source")
            ):
                return False
            if not meta.get("source"):
                return False
            if (
                required_adjust is not None
                and meta.get("adjust") != required_adjust
            ):
                return False
            requested_start = pd.Timestamp(meta["requested_start"])
            requested_end = pd.Timestamp(meta["requested_end"])
            actual_end = pd.Timestamp(meta["actual_end"])
            end_gap_days = (pd.Timestamp(end) - actual_end).days
            if (
                requested_start <= pd.Timestamp(start)
                and requested_end >= pd.Timestamp(end)
                and (
                    bool(meta.get("download_completed", True))
                    or end_gap_days <= 7
                )
            ):
                return True
        return False
    except Exception:
        return False


def load_price_data(
    symbol: str,
    start: str,
    end: str,
    prefix: str = "stock",
    *,
    require_amount: bool = False,
) -> pd.DataFrame:
    """加载股票价格数据：缓存优先 → 下载并存盘。日期格式 YYYY-MM-DD。"""
    cache_path = _get_cache_path(symbol, prefix)

    required_adjust = "qfq" if prefix == "stock" else "none"
    if _check_cache_coverage(
        cache_path,
        start,
        end,
        required_adjust=required_adjust,
        require_amount=require_amount,
    ):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if require_amount and "amount" not in df.columns:
            return pd.DataFrame()
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        logger.debug(f"[{symbol}] 缓存命中, {len(df)} 条")
        return df

    # 缓存不覆盖 → 下载
    logger.info(f"[{symbol}] 缓存缺失/过期, 正在下载...")
    try:
        if prefix == "index":
            df = fetch_index_daily(symbol, start, end)
        else:
            df = fetch_stock_daily(symbol, start, end)
    except Exception:
        logger.exception(f"[{symbol}] 下载失败")
        # 如果有部分缓存，降级使用
        if cache_path.exists():
            logger.warning(f"[{symbol}] 降级使用过期缓存")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if require_amount and "amount" not in df.columns:
                logger.warning(f"[{symbol}] 过期缓存缺少成交额，v2拒绝使用")
                return pd.DataFrame()
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            return df[(df.index >= start_dt) & (df.index <= end_dt)]
        return pd.DataFrame()

    source = str(df.attrs.get("source", "unknown"))
    resolved_adjust = str(df.attrs.get("adjust", required_adjust))
    amount_source = str(df.attrs.get("amount_source", "unknown"))
    df.to_csv(cache_path, encoding="utf-8-sig")
    with open(_coverage_meta_path(cache_path), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 3,
                "source": source,
                "adjust": resolved_adjust,
                "amount_source": amount_source,
                "download_completed": True,
                "requested_start": str(pd.Timestamp(start).date()),
                "requested_end": str(pd.Timestamp(end).date()),
                "actual_start": (
                    str(pd.Timestamp(df.index.min()).date()) if len(df) else None
                ),
                "actual_end": (
                    str(pd.Timestamp(df.index.max()).date()) if len(df) else None
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[{symbol}] 已缓存 {len(df)} 条 → {cache_path.name}")
    return df


def load_index_data(code: str, start: str, end: str) -> pd.DataFrame:
    """加载指数数据。code 如 '000300'。"""
    return load_price_data(code, start, end, prefix="index")


def load_pool_data(
    codes: list[str],
    start: str,
    end: str,
    *,
    require_amount: bool = False,
) -> dict[str, pd.DataFrame]:
    """批量加载股票池，容错跳过失败股票并并发补齐独立缓存。"""
    if not codes:
        return {}
    workers = max(1, min(int(os.getenv("PRICE_DOWNLOAD_WORKERS", "4")), 8))
    loaded: dict[str, pd.DataFrame] = {}

    def load_one(code: str) -> tuple[str, pd.DataFrame]:
        try:
            df = load_price_data(
                code,
                start,
                end,
                prefix="stock",
                require_amount=require_amount,
            )
            return code, df
        except Exception:
            logger.exception(f"{code} 加载失败，跳过")
            return code, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_one, code): code for code in codes}
        completed = 0
        for future in as_completed(futures):
            code, df = future.result()
            completed += 1
            if len(df) > 0:
                loaded[code] = df
            else:
                logger.warning(f"[{completed}/{len(codes)}] {code} 数据为空，跳过")

    # 保持调用方给定顺序，保证日志与可复现实验输出稳定。
    result = {code: loaded[code] for code in codes if code in loaded}
    logger.info(f"股票池加载完成: {len(result)}/{len(codes)} 只有效")
    return result


# ======================================================================
# 股票池加载
# ======================================================================

def load_stock_pool(
    config_path: Optional[str | Path] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """从 strategy_config.json 加载股票池。

    Returns:
        code_to_name: {"000001": "平安银行", ...}
        code_to_sector: {"000001": "银行", ...}
    """
    if config_path is None:
        config_path = MY_QUANT_DIR / "strategy_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    universe = config.get("universe", {})
    stocks = universe.get("stocks", {})
    sectors = config.get("sectors", {})

    code_to_name = dict(stocks)
    code_to_sector = dict(sectors)

    logger.info(f"股票池加载: {len(code_to_name)} 只股票, {len(set(code_to_sector.values()))} 个行业")
    return code_to_name, code_to_sector


def load_name_to_code_mapping(
    stock_pool_path: Optional[str | Path] = None,
) -> dict[str, str]:
    """从 stock_pool.json 构建名称→代码反向映射。"""
    if stock_pool_path is None:
        stock_pool_path = MY_QUANT_DIR / "stock_pool.json"

    with open(stock_pool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)

    stocks = pool.get("stocks", {})
    return {info["name"]: code for code, info in stocks.items()}


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    import sys

    codes_to_test = sys.argv[1:] if len(sys.argv) > 1 else ["000001", "600519", "300750"]

    for code in codes_to_test:
        print(f"\n{'='*60}")
        print(f"  测试: {code}")
        print(f"{'='*60}")
        try:
            df = load_price_data(code, "2024-01-01", "2024-06-01")
            print(f"  行数: {len(df)}, 列: {list(df.columns)}")
            print(df.head(3).to_string())
            print("  ...")
            print(df.tail(3).to_string())
        except Exception as e:
            print(f"  错误: {e}")

    # 加载股票池
    code_to_name, code_to_sector = load_stock_pool()
    print(f"\n股票池: {len(code_to_name)} 只")
    for c, n in list(code_to_name.items())[:5]:
        print(f"  {c} {n} ({code_to_sector.get(c, '?')})")
