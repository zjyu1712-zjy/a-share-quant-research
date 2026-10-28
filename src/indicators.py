"""
量化指标计算模块
- 从 MyQuantAgent 老项目提取的因子计算、选股、仓位管理、绩效评估逻辑
- 与 news_archive 情感数据对接，形成"情感因子 → 选股 → 回测"闭环
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ======================================================================
# 截面标准化
# ======================================================================

def zscore(series: pd.Series) -> pd.Series:
    """截面 Z-score 标准化（老项目 multifactor_rank.py 核心方法）"""
    return (series - series.mean()) / (series.std() + 1e-10)


# ======================================================================
# 价格因子（可用于价格数据，也适配情感数据的"虚拟价格"回测）
# ======================================================================

def momentum(series: pd.Series, window: int = 40) -> pd.Series:
    """滚动窗口动量 = N 日变化率"""
    return series.pct_change(window)


def pullback_from_high(series: pd.Series, window: int = 90) -> pd.Series:
    """距 N 日高点回撤幅度 = (当前 - 最高) / 最高"""
    rolling_high = series.rolling(window).max()
    return (series - rolling_high) / rolling_high


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """滚动波动率"""
    return series.pct_change().rolling(window).std()


def ma_deviation(series: pd.Series, window: int = 20) -> pd.Series:
    """均线偏离"""
    ma = series.rolling(window).mean()
    return (series - ma) / (ma + 1e-10)


# ======================================================================
# 情感特有因子（基于 news_archive.sentiment_score）
# ======================================================================

def sentiment_momentum(scores: pd.Series, window: int = 20) -> pd.Series:
    """情感分滚动均值 — 反映近期市场情绪趋势"""
    return scores.rolling(window).mean()


def sentiment_volatility(scores: pd.Series, window: int = 20) -> pd.Series:
    """情感分波动率 — 反映市场分歧程度"""
    return scores.rolling(window).std()


def sentiment_skew(scores: pd.Series, window: int = 20) -> pd.Series:
    """情感偏度 — 正偏 = 乐观集中，负偏 = 恐慌蔓延"""
    return scores.rolling(window).skew()


def positive_ratio(scores: pd.Series, window: int = 20) -> pd.Series:
    """窗口内利好占比"""
    return (scores > 0.1).rolling(window).mean()


def negative_ratio(scores: pd.Series, window: int = 20) -> pd.Series:
    """窗口内利空占比"""
    return (scores < -0.1).rolling(window).mean()


# ======================================================================
# 行业中性化选股（老项目 multifactor_rank.py / daily_signal.py）
# ======================================================================

def sector_neutral_select(
    ranked: pd.DataFrame,
    top_n: int,
    sectors: dict,
    max_per_sector: int = 1,
) -> list:
    """行业中性化选股：按得分降序，每个申万一级行业最多选 max_per_sector 只。

    Args:
        ranked: 含 code 列和 score 列的 DataFrame，已按 score 降序
        top_n:  目标持仓数
        sectors: {code: sector_name} 映射
    """
    selected = []
    used = {}

    for _, row in ranked.iterrows():
        code = row["code"]
        sector = sectors.get(code, "未知")
        if used.get(sector, 0) < max_per_sector:
            selected.append(code)
            used[sector] = used.get(sector, 0) + 1
        if len(selected) >= top_n:
            break

    # 如果选不够，忽略行业限制补足
    if len(selected) < top_n:
        for _, row in ranked.iterrows():
            if row["code"] not in selected:
                selected.append(row["code"])
            if len(selected) >= top_n:
                break

    return selected


def select_with_rank_buffer(
    ranked: pd.DataFrame,
    top_n: int,
    sectors: dict,
    max_per_sector: int = 1,
    previous_picks: list[str] | None = None,
    rank_buffer: int = 0,
    sector_enabled: bool = True,
) -> list[str]:
    """带排名缓冲的选股。

    旧持仓只要仍位于 ``Top N + buffer`` 内就优先保留，再按最新排名补足。
    这会减少临界名次反复进出造成的无效换手；行业上限仍然生效。
    """
    if top_n <= 0 or ranked.empty:
        return []
    if not previous_picks or rank_buffer <= 0:
        if sector_enabled:
            return sector_neutral_select(
                ranked, top_n, sectors, max_per_sector=max_per_sector
            )
        return ranked.head(top_n)["code"].tolist()

    ranked_codes = ranked["code"].astype(str).tolist()
    eligible_old = set(ranked_codes[:top_n + max(rank_buffer, 0)])
    previous = set(str(code) for code in previous_picks)
    retained = [
        code for code in ranked_codes
        if code in previous and code in eligible_old
    ]
    ordered_candidates = retained + [
        code for code in ranked_codes if code not in retained
    ]

    selected: list[str] = []
    used: dict[str, int] = {}
    for code in ordered_candidates:
        sector = sectors.get(code, "未知")
        if (
            not sector_enabled
            or used.get(sector, 0) < max_per_sector
        ):
            selected.append(code)
            used[sector] = used.get(sector, 0) + 1
        if len(selected) >= top_n:
            break

    # 与原有行业中性逻辑一致：极端情况下软化行业限制以补足持仓数。
    if len(selected) < top_n:
        for code in ordered_candidates:
            if code not in selected:
                selected.append(code)
            if len(selected) >= top_n:
                break
    return selected


def _normalize_capped_weights(
    raw_weights: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    """按比例归一化，并用水位法施加单股权重上限。"""
    if not raw_weights:
        return {}
    codes = list(raw_weights)
    cap = max(float(max_weight), 1.0 / len(codes))
    remaining = set(codes)
    result = {code: 0.0 for code in codes}
    remaining_mass = 1.0

    while remaining:
        raw_sum = sum(max(raw_weights[code], 0.0) for code in remaining)
        if raw_sum <= 0:
            equal = remaining_mass / len(remaining)
            for code in remaining:
                result[code] = equal
            break
        proposed = {
            code: remaining_mass * max(raw_weights[code], 0.0) / raw_sum
            for code in remaining
        }
        capped = [code for code, value in proposed.items() if value > cap]
        if not capped:
            result.update(proposed)
            break
        for code in capped:
            result[code] = cap
            remaining_mass -= cap
            remaining.remove(code)

    total = sum(result.values())
    return {
        code: value / total for code, value in result.items()
    } if total > 0 else {}


def compute_target_weights(
    target_codes: list[str],
    price_data: dict[str, pd.DataFrame],
    as_of: str | pd.Timestamp,
    method: str = "equal_weight",
    volatility_window: int = 60,
    max_weight: float = 0.35,
) -> dict[str, float]:
    """仅使用信号日及以前的数据生成目标权重。"""
    codes = list(dict.fromkeys(str(code) for code in target_codes))
    if not codes:
        return {}
    if method == "equal_weight":
        return {code: 1.0 / len(codes) for code in codes}
    if method != "inverse_volatility":
        raise ValueError(f"不支持的仓位权重方法: {method}")

    cutoff = pd.Timestamp(as_of)
    volatilities: dict[str, float] = {}
    min_observations = max(10, min(volatility_window // 3, 20))
    for code in codes:
        frame = price_data.get(code)
        if frame is None or "close" not in frame.columns:
            continue
        closes = pd.to_numeric(
            frame.loc[frame.index <= cutoff, "close"], errors="coerce"
        ).dropna().tail(max(volatility_window, 2) + 1)
        returns = closes.pct_change().dropna()
        volatility = float(returns.std()) if len(returns) >= min_observations else np.nan
        if np.isfinite(volatility) and volatility > 0:
            volatilities[code] = volatility

    # 数据不足的股票使用可用样本的中位波动率，避免因缺数获得异常高权重。
    fallback = (
        float(np.median(list(volatilities.values())))
        if volatilities else 1.0
    )
    raw = {
        code: 1.0 / volatilities.get(code, fallback)
        for code in codes
    }
    return _normalize_capped_weights(raw, max_weight=max_weight)


# ======================================================================
# 仓位计算（老项目 daily_signal.py）
# ======================================================================

def calculate_position_sizes(
    target_codes: list,
    prices: dict,
    total_capital: float,
    min_lot: int = 100,
    commission: float = 0.0003,
    min_commission: float = 0.0,
    transfer_fee_rate: float = 0.0,
    slippage_bps: float = 0.0,
    target_weights: dict[str, float] | None = None,
) -> tuple[dict, float, float]:
    """按目标权重分配仓位，遵守 A 股整手交易规则。

    若某只股票单价过高导致 1 手都买不起，则跳过并将资金再分配给其他标的。

    Returns:
        positions: {code: {price, shares, lots, cost, cash_left}}
        total_deployed: 实际投入总资金
        total_leftover: 现金留存
    """
    slip = max(slippage_bps, 0.0) / 10_000
    valid = [
        (c, prices[c] * (1 + slip))
        for c in target_codes if c in prices and prices[c] > 0
    ]
    if not valid:
        return {}, 0.0, total_capital

    raw_weights = {
        code: (
            max(float(target_weights.get(code, 0.0)), 0.0)
            if target_weights else 1.0
        )
        for code, _ in valid
    }

    def budgets_for(items: list[tuple[str, float]]) -> dict[str, float]:
        active = {code: raw_weights.get(code, 0.0) for code, _ in items}
        total = sum(active.values())
        if total <= 0:
            return {code: total_capital / len(items) for code, _ in items}
        return {
            code: total_capital * weight / total
            for code, weight in active.items()
        }

    working = list(valid)
    while len(working) > 1:
        budgets = budgets_for(working)
        affordable = [
            (c, p)
            for c, p in working
            if (
                min_lot * p
                + max(min_lot * p * commission, min_commission)
                + min_lot * p * transfer_fee_rate
            ) <= budgets[c]
        ]
        if len(affordable) == len(working):
            break
        if not affordable:
            affordable = [min(working, key=lambda x: x[1])]
        working = affordable

    budgets = budgets_for(working)
    positions = {}
    deployed = 0.0
    leftover = 0.0

    for code, price in working:
        budget_per_stock = budgets[code]
        variable_rate = commission + transfer_fee_rate
        shares = int(
            budget_per_stock / (price * (1 + variable_rate)) / min_lot
        ) * min_lot
        while shares > 0:
            gross = shares * price
            commission_fee = max(gross * commission, min_commission)
            transfer_fee = gross * transfer_fee_rate
            cost = gross + commission_fee + transfer_fee
            if cost <= budget_per_stock + 1e-8:
                break
            shares -= min_lot
        if shares <= 0:
            continue
        gross = shares * price
        commission_fee = max(gross * commission, min_commission)
        transfer_fee = gross * transfer_fee_rate
        cost = gross + commission_fee + transfer_fee
        positions[code] = {
            "price": float(price),
            "reference_price": float(prices[code]),
            "shares": shares,
            "lots": shares // min_lot,
            "gross_value": round(gross, 2),
            "commission": round(commission_fee, 2),
            "transfer_fee": round(transfer_fee, 2),
            "cost": round(cost, 2),
            "cash_left": round(budget_per_stock - cost, 2),
        }
        deployed += cost
        leftover += budget_per_stock - cost

    return positions, round(deployed, 2), round(leftover, 2)


# ======================================================================
# 绩效指标（老项目 multifactor_rank.py / main.py）
# ======================================================================

def compute_performance(
    equity_curve: pd.Series,
    rf_annual: float = 0.02,
    periods_per_year: int = 12,
) -> dict:
    """从权益曲线计算核心绩效指标。

    Args:
        equity_curve: 按时间索引的权益序列
        rf_annual:    年化无风险利率，默认 2%
        periods_per_year: 年化周期数（日线=252, 周线=52, 月线=12）

    Returns:
        dict with: total_return%, cagr%, sharpe, max_dd%, volatility%,
                   calmar, win_rate% (by period), n_periods
    """
    if len(equity_curve) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "win_rate": 0.0, "n_periods": len(equity_curve)}

    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "win_rate": 0.0, "n_periods": 0}

    final = equity_curve.iloc[-1]
    initial = equity_curve.iloc[0]
    total_ret = (final / initial - 1) * 100

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = max(days / 365.25, 0.02)
    cagr = ((final / initial) ** (1 / years) - 1) * 100

    ann_vol = returns.std() * np.sqrt(periods_per_year) * 100
    ann_excess = returns.mean() * periods_per_year - rf_annual
    sharpe = ann_excess / (ann_vol / 100) if ann_vol > 0 else 0.0

    cummax = equity_curve.cummax()
    drawdowns = (equity_curve - cummax) / cummax
    max_dd = drawdowns.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (returns > 0).mean() * 100

    return {
        "total_return": round(total_ret, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "ann_volatility": round(ann_vol, 2),
        "max_drawdown": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate, 1),
        "n_periods": len(returns),
    }


# ======================================================================
# 风控检查（老项目 daily_signal.py run_risk_check）
# ======================================================================

def check_risk(
    target_codes: list,
    stock_names: dict,
    sectors: dict,
    prices: dict,
    prev_close: dict | None = None,
    portfolio_cost: dict | None = None,
    stop_loss_pct: float = 0.07,
) -> dict:
    """实盘前风控检查。

    Returns:
        {passed: bool, blocks_trading: bool, checks: list[dict]}
    """
    now = datetime.now()
    checks = []
    blocks = False

    # A2: 交易时间
    wd = now.weekday()
    minutes = now.hour * 60 + now.minute
    if wd >= 5:
        checks.append({"cat": "A2", "status": "fail", "msg": f"非交易日 (weekday={wd})"})
        blocks = True
    elif minutes < 570:
        checks.append({"cat": "A2", "status": "warn", "msg": f"距开盘还有 {570 - minutes} 分钟"})
    elif minutes > 900:
        checks.append({"cat": "A2", "status": "warn", "msg": "已收盘"})
    else:
        checks.append({"cat": "A2", "status": "ok", "msg": "交易时段内"})

    # A3: 涨跌停检查
    if prev_close:
        for code in target_codes:
            if code not in prices or code not in prev_close:
                continue
            pc = prev_close[code]
            if pc <= 0:
                continue
            limit_pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
            limit_up = round(pc * (1 + limit_pct), 2)
            limit_down = round(pc * (1 - limit_pct), 2)
            price = prices[code]
            name = stock_names.get(code, code)

            if price >= limit_up * 0.999:
                checks.append({"cat": "A3", "status": "fail",
                               "msg": f"{code} {name} 涨停 (price={price}, limit={limit_up})"})
                blocks = True
            elif price <= limit_down * 1.001:
                checks.append({"cat": "A3", "status": "warn",
                               "msg": f"{code} {name} 跌停 (price={price}, limit={limit_down})"})

    # C1: 止损检查
    if portfolio_cost:
        for code, cost in portfolio_cost.items():
            if code in prices:
                pnl = (prices[code] - cost) / cost
                if pnl <= -stop_loss_pct:
                    checks.append({"cat": "C1", "status": "fail",
                                   "msg": f"{code} 触发止损 (pnl={pnl*100:.1f}% < -{stop_loss_pct*100:.0f}%)"})
                    blocks = True

    # C2: ST/退市
    for code in target_codes:
        name = stock_names.get(code, "")
        if "ST" in name.upper() or "*ST" in name:
            checks.append({"cat": "C2", "status": "fail", "msg": f"{code} {name} ST/退市风险"})
            blocks = True

    return {
        "passed": not blocks,
        "blocks_trading": blocks,
        "checks": checks,
        "check_time": now.isoformat(),
    }
