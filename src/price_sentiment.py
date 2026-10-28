"""
价格代理情感因子模块
- 从 OHLCV 价格数据中提取情绪代理信号
- 当 SQLite 新闻情感数据不足时，作为 fallback 提供伪情感序列
- 覆盖维度: 隔夜情绪 / 日内方向 / 资金关注度 / 买卖压力 / 波动率

输出: 日频 pseudo_sentiment score，格式与 load_sentiment_series 兼容
"""

import numpy as np
import pandas as pd
from loguru import logger


# ======================================================================
# 单指标计算
# ======================================================================

def gap_pct(df: pd.DataFrame) -> pd.Series:
    """跳空缺口 = (今开 - 昨收) / 昨收 — 隔夜市场情绪。

    正向缺口 = 盘后/盘前利好 → 看涨情绪
    负向缺口 = 利空消息发酵 → 看空情绪
    """
    return (df["open"] - df["close"].shift(1)) / df["close"].shift(1)


def intraday_momentum(df: pd.DataFrame) -> pd.Series:
    """日内动量 = (收盘 - 开盘) / 开盘 — 日内买卖方向。

    正 = 日内买方主导，负 = 日内卖方主导。
    """
    return (df["close"] - df["open"]) / df["open"]


def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """量比 = 当日成交量 / N日均量 — 资金关注度/交易活跃度。

    放量 = 市场高度关注（可能伴随极端情绪），缩量 = 冷清。
    """
    avg_vol = df["volume"].rolling(window).mean()
    return df["volume"] / (avg_vol + 1e-10)


def close_position(df: pd.DataFrame) -> pd.Series:
    """收盘位 = (收盘 - 最低) / (最高 - 最低) — 日内买卖压力。

    接近 1 = 买方强势（收在高位），接近 0 = 卖方强势（收在低位）。
    """
    hl_range = df["high"] - df["low"]
    return (df["close"] - df["low"]) / (hl_range + 1e-10)


def amplitude(df: pd.DataFrame) -> pd.Series:
    """日内振幅 = (最高 - 最低) / 昨收 — 波动率/分歧程度。

    高振幅 = 多空分歧大（不确定性），低振幅 = 共识强。
    """
    return (df["high"] - df["low"]) / df["close"].shift(1)


def price_trend(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """短期价格趋势 = N日涨跌幅 — 辅助判断方向持续性。

    与 sentiment_momentum 语义对齐。
    """
    return df["close"].pct_change(window)


# ======================================================================
# 合成伪情感分
# ======================================================================

# 各子指标权重（通过历史经验调试）
DEFAULT_WEIGHTS = {
    "gap": 0.35,            # 跳空缺口 — 隔夜情绪信号 (↑优化)
    "intraday": 0.30,       # 日内动量 — 日内情绪方向 (↑优化)
    "volume_ratio": 0.20,   # 量比 — 资金关注度 (↑优化)
    "close_position": 0.10, # 收盘位 — 买卖压力
    "amplitude": -0.05,     # 振幅 — 反向 (↓弱化)
    "trend": 0.00,          # 短期趋势 — 移除贡献
}


def compute_daily_sentiment(
    df: pd.DataFrame,
    weights: dict | None = None,
) -> pd.Series:
    """从 OHLCV DataFrame 计算日频伪情感分。

    流程:
      1. 逐日计算 6 个情绪子指标
      2. 每个指标滚动 Z-score 标准化 (60天窗口)
      3. 按权重加总 → daily_pseudo_sentiment，范围约 [-3, +3]

    Args:
        df: 含 open/high/low/close/volume 的 DataFrame，按 date 索引
        weights: 自定义权重字典，默认使用 DEFAULT_WEIGHTS

    Returns:
        pd.Series: 日频伪情感分 (index=date, values=score)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 计算子指标
    indicators = {
        "gap": gap_pct(df),
        "intraday": intraday_momentum(df),
        "volume_ratio": volume_ratio(df),
        "close_position": close_position(df),
        "amplitude": amplitude(df),
        "trend": price_trend(df),
    }

    # 每个指标滚动 Z-score 标准化 (60天长窗口, 保证稳定性)
    z_scores = {}
    for name, series in indicators.items():
        rolling_mean = series.rolling(60, min_periods=10).mean()
        rolling_std = series.rolling(60, min_periods=10).std()
        z = (series - rolling_mean) / (rolling_std + 1e-10)
        # 截断极端值
        z = z.clip(-4, 4)
        z_scores[name] = z

    # 加权合成
    score = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        if name in z_scores:
            score += z_scores[name] * w

    # 标准化到 [-3, 3] 区间（与 DeepSeek sentiment_score 语义对齐）
    score_std = score.std()
    if score_std > 0:
        score = score / score_std * 1.2  # 缩放使得 ~95% 的值在 [-3, 3]
    score = score.clip(-3, 3)

    score.name = "pseudo_sentiment"
    return score


# ======================================================================
# 与 dual_factor.py 的接口
# ======================================================================

def compute_pseudo_sentiment_for_pool(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """为整个股票池计算价格代理情感序列。

    Args:
        price_data: {code: DataFrame (OHLCV, date index)}

    Returns:
        {code: Series (date → pseudo_sentiment_score)}
    """
    result = {}
    for code, df in price_data.items():
        try:
            s = compute_daily_sentiment(df)
            s = s.dropna()
            if len(s) > 20:
                result[code] = s
        except Exception:
            logger.exception(f"[{code}] 价格代理情感计算失败")

    coverage = len(result)
    total = len(price_data)
    logger.info(f"价格代理情感: {coverage}/{total} 只股票 ({coverage/total*100:.0f}%)")
    return result


# ======================================================================
# 便捷入口
# ======================================================================

if __name__ == "__main__":
    from src.price_feed import load_pool_data, load_stock_pool

    code_to_name, code_to_sector = load_stock_pool()
    codes = list(code_to_name.keys())[:3]  # 测试 3 只

    print(f"测试股票: {codes}")
    data = load_pool_data(codes, "2024-01-01", "2024-12-31")

    for code, df in data.items():
        sentiment = compute_daily_sentiment(df)
        print(f"\n{code} {code_to_name[code]}:")
        print(f"  伪情感: {len(sentiment)} 天, "
              f"范围 [{sentiment.min():.2f}, {sentiment.max():.2f}], "
              f"均值 {sentiment.mean():.3f}, 标准差 {sentiment.std():.3f}")
        print(f"  最后5天:\n{sentiment.tail(5).to_string()}")
