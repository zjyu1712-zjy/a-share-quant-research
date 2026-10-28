"""统一的事件驱动组合执行器。

所有研究入口都使用同一套成交约定：
1. 信号在收盘后生成；
2. 下一交易日开盘调仓；
3. 按实际现金、整手和手续费记账；
4. 按日收盘价记录净值和最大回撤；
5. 新买入股票遵守 T+1，当日不触发卖出。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _market_calendar(
    price_data: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Build the union trading calendar instead of trusting one stock."""
    dates: set[pd.Timestamp] = set()
    for frame in price_data.values():
        dates.update(pd.DatetimeIndex(frame.index).normalize())
    return pd.DatetimeIndex(sorted(dates))


def generate_rebalance_dates(
    price_data: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    frequency: str = "weekly",
) -> pd.DatetimeIndex:
    """返回各周期最后一个真实交易日，而不是日历桶标签。"""
    if not price_data:
        return pd.DatetimeIndex([])

    calendar = _market_calendar(price_data)
    mask = (
        (calendar >= pd.Timestamp(start_date))
        & (calendar <= pd.Timestamp(end_date))
    )
    actual = calendar[mask]
    if len(actual) == 0:
        return pd.DatetimeIndex([])
    if frequency == "daily":
        return actual

    rule = {"weekly": "W-FRI", "monthly": "ME"}.get(frequency)
    if rule is None:
        raise ValueError(f"不支持的调仓频率: {frequency}")

    values = pd.Series(actual, index=actual).resample(rule).last().dropna()
    return pd.DatetimeIndex(values.to_numpy())


def _row_on_date(
    price_data: dict[str, pd.DataFrame],
    code: str,
    date: pd.Timestamp,
) -> Optional[pd.Series]:
    df = price_data.get(code)
    if df is None or date not in df.index:
        return None
    row = df.loc[date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _price_on_or_before(
    price_data: dict[str, pd.DataFrame],
    code: str,
    date: pd.Timestamp,
    field: str = "close",
) -> Optional[float]:
    df = price_data.get(code)
    if df is None or field not in df.columns:
        return None
    eligible = df[df.index <= date]
    if eligible.empty:
        return None
    value = eligible[field].iloc[-1]
    return float(value) if pd.notna(value) else None


def _portfolio_value(
    cash: float,
    positions: dict[str, dict],
    price_data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    field: str,
) -> tuple[float, float]:
    stock_value = 0.0
    for code, pos in positions.items():
        price = _price_on_or_before(price_data, code, date, field)
        if price is None:
            price = pos["entry_price"]
        stock_value += pos["shares"] * price
    return cash + stock_value, stock_value


def _next_session(calendar: pd.DatetimeIndex, signal_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    future = calendar[calendar > signal_date]
    return pd.Timestamp(future[0]) if len(future) else None


def _stamp_duty_rate(
    date: pd.Timestamp,
    current_rate: float,
    historical_rate: float,
    cutover_date: str,
) -> float:
    return (
        historical_rate
        if date < pd.Timestamp(cutover_date)
        else current_rate
    )


def _fees(
    *,
    gross_value: float,
    side: str,
    date: pd.Timestamp,
    commission_rate: float,
    min_commission: float,
    transfer_fee_rate: float,
    stamp_duty_rate: float,
    historical_stamp_duty_rate: float,
    stamp_duty_cutover_date: str,
) -> dict[str, float]:
    commission_fee = (
        max(gross_value * commission_rate, min_commission)
        if gross_value > 0 and commission_rate > 0 else 0.0
    )
    transfer_fee = gross_value * max(transfer_fee_rate, 0.0)
    stamp_duty = 0.0
    if side == "sell":
        stamp_duty = gross_value * _stamp_duty_rate(
            date,
            stamp_duty_rate,
            historical_stamp_duty_rate,
            stamp_duty_cutover_date,
        )
    total = commission_fee + transfer_fee + stamp_duty
    return {
        "commission": commission_fee,
        "transfer_fee": transfer_fee,
        "stamp_duty": stamp_duty,
        "total": total,
    }


def _execution_price(open_price: float, side: str, slippage_bps: float) -> float:
    slip = max(slippage_bps, 0.0) / 10_000
    multiplier = 1 + slip if side == "buy" else 1 - slip
    return open_price * multiplier


def _limit_pct(code: str, date: pd.Timestamp) -> float:
    """Approximate regular A-share daily price limits by board and date."""
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith(("4", "8")):
        return 0.30
    return 0.10


def _is_limit_locked(
    price_data: dict[str, pd.DataFrame],
    code: str,
    date: pd.Timestamp,
    side: str,
) -> bool:
    """Conservatively reject an order when the stock is locked at its limit."""
    frame = price_data.get(code)
    row = _row_on_date(price_data, code, date)
    if frame is None or row is None:
        return False
    prior = frame[frame.index < date]
    # New listings can trade without regular limits in their first sessions.
    if len(prior) < 5:
        return False
    previous_close = float(prior["close"].iloc[-1])
    pct = _limit_pct(code, date)
    upper = round(previous_close * (1 + pct) + 1e-8, 2)
    lower = round(previous_close * (1 - pct) + 1e-8, 2)
    tolerance = 0.011
    if side == "buy":
        return float(row["low"]) >= upper - tolerance
    return float(row["high"]) <= lower + tolerance


def _capacity_shares(
    price_data: dict[str, pd.DataFrame],
    code: str,
    date: pd.Timestamp,
    *,
    max_volume_participation: Optional[float],
    volume_lookback: int,
    min_lot: int,
) -> Optional[int]:
    """Capacity from prior average volume only; no execution-day look-ahead."""
    if max_volume_participation is None or max_volume_participation <= 0:
        return None
    frame = price_data.get(code)
    if frame is None or "volume" not in frame.columns:
        return 0
    history = pd.to_numeric(
        frame.loc[frame.index < date, "volume"], errors="coerce"
    ).dropna().tail(max(volume_lookback, 1))
    if history.empty or history.mean() <= 0:
        return 0
    shares = int(
        history.mean() * max_volume_participation / min_lot
    ) * min_lot
    return max(shares, 0)


def _max_affordable_shares(
    *,
    budget: float,
    price: float,
    max_shares: Optional[int],
    min_lot: int,
    date: pd.Timestamp,
    commission_rate: float,
    min_commission: float,
    transfer_fee_rate: float,
    stamp_duty_rate: float,
    historical_stamp_duty_rate: float,
    stamp_duty_cutover_date: str,
) -> int:
    if budget <= 0 or price <= 0:
        return 0
    variable_rate = max(commission_rate, 0.0) + max(transfer_fee_rate, 0.0)
    shares = int(budget / (price * (1 + variable_rate)) / min_lot) * min_lot
    if max_shares is not None:
        shares = min(shares, max_shares)
    while shares > 0:
        gross = shares * price
        fees = _fees(
            gross_value=gross,
            side="buy",
            date=date,
            commission_rate=commission_rate,
            min_commission=min_commission,
            transfer_fee_rate=transfer_fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            historical_stamp_duty_rate=historical_stamp_duty_rate,
            stamp_duty_cutover_date=stamp_duty_cutover_date,
        )
        if gross + fees["total"] <= budget + 1e-8:
            return shares
        shares -= min_lot
    return 0


def _rebalance_at_open(
    *,
    date: pd.Timestamp,
    signal_date: pd.Timestamp,
    target: list[str],
    target_weights: Optional[dict[str, float]],
    cash: float,
    positions: dict[str, dict],
    price_data: dict[str, pd.DataFrame],
    commission: float,
    min_lot: int,
    min_commission: float,
    transfer_fee_rate: float,
    stamp_duty_rate: float,
    historical_stamp_duty_rate: float,
    stamp_duty_cutover_date: str,
    slippage_bps: float,
    max_volume_participation: Optional[float],
    volume_lookback: int,
    enforce_price_limits: bool,
    gross_exposure_cap: float = 1.0,
) -> tuple[float, dict[str, dict], dict]:
    """在开盘调仓，计入费用、滑点、价格限制和流动性容量。"""
    open_prices: dict[str, float] = {}
    for code in set(target) | set(positions):
        row = _row_on_date(price_data, code, date)
        if row is not None and pd.notna(row.get("open")) and float(row["open"]) > 0:
            open_prices[code] = float(row["open"])

    sells: list[dict] = []
    buys: list[dict] = []
    skipped: list[str] = []
    constraints: list[dict] = []
    capacity_remaining: dict[str, Optional[int]] = {
        code: _capacity_shares(
            price_data,
            code,
            date,
            max_volume_participation=max_volume_participation,
            volume_lookback=volume_lookback,
            min_lot=min_lot,
        )
        for code in set(target) | set(positions)
    }

    # 无法取得目标股票开盘价时不假设成交。
    valid_target = [c for c in target if c in open_prices]
    for code in target:
        if code not in open_prices:
            constraints.append({
                "code": code, "side": "buy", "reason": "suspended_or_no_open",
            })

    buy_allowed = {
        code: not (
            enforce_price_limits
            and _is_limit_locked(price_data, code, date, "buy")
        )
        for code in valid_target
    }
    sell_allowed = {
        code: not (
            enforce_price_limits
            and _is_limit_locked(price_data, code, date, "sell")
        )
        for code in positions
        if code in open_prices
    }
    for code, allowed in buy_allowed.items():
        if not allowed and code not in positions:
            constraints.append({
                "code": code, "side": "buy", "reason": "limit_up_locked",
            })

    # 先卖掉不再入选的持仓。
    for code in list(positions):
        if code in valid_target or code not in open_prices:
            continue
        if not sell_allowed.get(code, True):
            constraints.append({
                "code": code, "side": "sell", "reason": "limit_down_locked",
                "requested_shares": positions[code]["shares"], "filled_shares": 0,
            })
            continue
        pos = positions[code]
        requested = pos["shares"]
        cap = capacity_remaining.get(code)
        shares = requested if cap is None else min(requested, cap)
        shares = int(shares / min_lot) * min_lot
        if shares <= 0:
            constraints.append({
                "code": code, "side": "sell", "reason": "volume_capacity",
                "requested_shares": requested, "filled_shares": 0,
            })
            continue
        price = _execution_price(open_prices[code], "sell", slippage_bps)
        gross = shares * price
        fees = _fees(
            gross_value=gross, side="sell", date=date,
            commission_rate=commission, min_commission=min_commission,
            transfer_fee_rate=transfer_fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            historical_stamp_duty_rate=historical_stamp_duty_rate,
            stamp_duty_cutover_date=stamp_duty_cutover_date,
        )
        proceeds = gross - fees["total"]
        cash += proceeds
        pos["shares"] -= shares
        if pos["shares"] <= 0:
            positions.pop(code)
        if cap is not None:
            capacity_remaining[code] = cap - shares
        sells.append({
            "code": code,
            "shares": shares,
            "price": price,
            "reference_price": open_prices[code],
            "gross_value": round(gross, 2),
            "fees": {k: round(v, 2) for k, v in fees.items()},
            "slippage_cost": round(
                shares * (open_prices[code] - price), 2
            ),
            "proceeds": round(proceeds, 2),
            "note": "rebalance_exit",
        })
        if shares < requested:
            constraints.append({
                "code": code, "side": "sell", "reason": "volume_capacity",
                "requested_shares": requested, "filled_shares": shares,
            })

    total_value, _ = _portfolio_value(
        cash, positions, price_data, date, field="open"
    )

    gross_exposure_cap = min(max(float(gross_exposure_cap), 0.0), 1.0)
    deployable_value = total_value * gross_exposure_cap
    raw_weights = {
        code: (
            max(float(target_weights.get(code, 0.0)), 0.0)
            if target_weights else 1.0
        )
        for code in valid_target
    }

    def allocations_for(codes: list[str]) -> dict[str, float]:
        active = {code: raw_weights.get(code, 0.0) for code in codes}
        total = sum(active.values())
        if total <= 0:
            return {code: deployable_value / len(codes) for code in codes}
        return {
            code: deployable_value * weight / total
            for code, weight in active.items()
        }

    # 移除按目标预算连一手都买不起的目标，并将预算重新分配。
    working = [
        code for code in valid_target
        if code in positions or buy_allowed.get(code, True)
    ]
    while len(working) > 1:
        allocations = allocations_for(working)
        affordable = [
            c for c in working
            if (
                c in positions
                or _max_affordable_shares(
                    budget=allocations[c],
                    price=_execution_price(open_prices[c], "buy", slippage_bps),
                    max_shares=capacity_remaining.get(c),
                    min_lot=min_lot,
                    date=date,
                    commission_rate=commission,
                    min_commission=min_commission,
                    transfer_fee_rate=transfer_fee_rate,
                    stamp_duty_rate=stamp_duty_rate,
                    historical_stamp_duty_rate=historical_stamp_duty_rate,
                    stamp_duty_cutover_date=stamp_duty_cutover_date,
                ) >= min_lot
            )
        ]
        if len(affordable) == len(working):
            break
        if not affordable:
            affordable = [min(working, key=lambda c: open_prices[c])]
        skipped.extend(c for c in working if c not in affordable)
        working = affordable

    allocations = allocations_for(working) if working else {}
    target_shares: dict[str, int] = {}
    for code in working:
        price = _execution_price(open_prices[code], "buy", slippage_bps)
        shares = _max_affordable_shares(
            budget=allocations[code],
            price=price,
            max_shares=None,
            min_lot=min_lot,
            date=date,
            commission_rate=commission,
            min_commission=min_commission,
            transfer_fee_rate=transfer_fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            historical_stamp_duty_rate=historical_stamp_duty_rate,
            stamp_duty_cutover_date=stamp_duty_cutover_date,
        )
        target_shares[code] = max(shares, 0)

    # 先减仓，为后续买入释放现金。
    for code in list(positions):
        if code not in target_shares or code not in open_prices:
            continue
        current = positions[code]["shares"]
        desired = target_shares[code]
        if desired >= current:
            continue
        requested = current - desired
        if not sell_allowed.get(code, True):
            constraints.append({
                "code": code, "side": "sell", "reason": "limit_down_locked",
                "requested_shares": requested, "filled_shares": 0,
            })
            continue
        cap = capacity_remaining.get(code)
        shares = requested if cap is None else min(requested, cap)
        shares = int(shares / min_lot) * min_lot
        if shares <= 0:
            constraints.append({
                "code": code, "side": "sell", "reason": "volume_capacity",
                "requested_shares": requested, "filled_shares": 0,
            })
            continue
        price = _execution_price(open_prices[code], "sell", slippage_bps)
        gross = shares * price
        fees = _fees(
            gross_value=gross, side="sell", date=date,
            commission_rate=commission, min_commission=min_commission,
            transfer_fee_rate=transfer_fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            historical_stamp_duty_rate=historical_stamp_duty_rate,
            stamp_duty_cutover_date=stamp_duty_cutover_date,
        )
        proceeds = gross - fees["total"]
        cash += proceeds
        positions[code]["shares"] = current - shares
        if cap is not None:
            capacity_remaining[code] = cap - shares
        sells.append({
            "code": code,
            "shares": shares,
            "price": price,
            "reference_price": open_prices[code],
            "gross_value": round(gross, 2),
            "fees": {k: round(v, 2) for k, v in fees.items()},
            "slippage_cost": round(
                shares * (open_prices[code] - price), 2
            ),
            "proceeds": round(proceeds, 2),
            "note": "rebalance_trim",
        })
        if positions[code]["shares"] == 0:
            positions.pop(code, None)
        if shares < requested:
            constraints.append({
                "code": code, "side": "sell", "reason": "volume_capacity",
                "requested_shares": requested, "filled_shares": shares,
            })

    # 再增仓；现金不足时只买得起的整手数量。
    for code in working:
        desired = target_shares[code]
        current = positions.get(code, {}).get("shares", 0)
        requested = max(desired - current, 0)
        if requested == 0:
            continue
        if not buy_allowed.get(code, True):
            constraints.append({
                "code": code, "side": "buy", "reason": "limit_up_locked",
                "requested_shares": requested, "filled_shares": 0,
            })
            continue
        price = _execution_price(open_prices[code], "buy", slippage_bps)
        cap = capacity_remaining.get(code)
        affordable = _max_affordable_shares(
            budget=cash,
            price=price,
            max_shares=cap,
            min_lot=min_lot,
            date=date,
            commission_rate=commission,
            min_commission=min_commission,
            transfer_fee_rate=transfer_fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            historical_stamp_duty_rate=historical_stamp_duty_rate,
            stamp_duty_cutover_date=stamp_duty_cutover_date,
        )
        shares = min(requested, affordable)
        shares = int(shares / min_lot) * min_lot
        if shares <= 0:
            skipped.append(code)
            constraints.append({
                "code": code, "side": "buy",
                "reason": "cash_or_volume_capacity",
                "requested_shares": requested, "filled_shares": 0,
            })
            continue

        gross = shares * price
        fees = _fees(
            gross_value=gross, side="buy", date=date,
            commission_rate=commission, min_commission=min_commission,
            transfer_fee_rate=transfer_fee_rate,
            stamp_duty_rate=stamp_duty_rate,
            historical_stamp_duty_rate=historical_stamp_duty_rate,
            stamp_duty_cutover_date=stamp_duty_cutover_date,
        )
        cost = gross + fees["total"]
        cash -= cost
        if cap is not None:
            capacity_remaining[code] = cap - shares
        if code in positions:
            old_shares = positions[code]["shares"]
            old_cost = positions[code]["entry_price"] * old_shares
            new_total = old_shares + shares
            positions[code]["entry_price"] = (old_cost + price * shares) / new_total
            positions[code]["shares"] = new_total
            positions[code]["last_buy_date"] = date
        else:
            positions[code] = {
                "shares": shares,
                "entry_price": price,
                "entry_date": date,
                "last_buy_date": date,
                "highest": price,
            }
        buys.append({
            "code": code,
            "action": "BUY",
            "shares": shares,
            "total_shares": positions[code]["shares"],
            "lots": shares // min_lot,
            "price": price,
            "reference_price": open_prices[code],
            "gross_value": round(gross, 2),
            "fees": {k: round(v, 2) for k, v in fees.items()},
            "slippage_cost": round(
                shares * (price - open_prices[code]), 2
            ),
            "alloc": round(allocations[code], 2),
            "cost": round(cost, 2),
            "note": "rebalance_buy",
        })
        if shares < requested:
            constraints.append({
                "code": code, "side": "buy", "reason": "cash_or_volume_capacity",
                "requested_shares": requested, "filled_shares": shares,
            })

    total_equity, stock_value = _portfolio_value(
        cash, positions, price_data, date, field="open"
    )
    log = {
        "date": date,
        "signal_date": signal_date,
        "target": list(target),
        "buys": buys,
        "sells": sells,
        "skipped": sorted(set(skipped)),
        "constraints": constraints,
        "risk_on": bool(target),
        "gross_exposure_cap": gross_exposure_cap,
        "cash_after": cash,
        "stock_value": stock_value,
        "equity": total_equity,
    }
    return cash, positions, log


def simulate_portfolio(
    *,
    selection_track: list[dict],
    price_data: dict[str, pd.DataFrame],
    initial_capital: float,
    commission: float = 0.0003,
    min_lot: int = 100,
    min_commission: float = 0.0,
    transfer_fee_rate: float = 0.0,
    stamp_duty_rate: float = 0.0,
    historical_stamp_duty_rate: float = 0.0,
    stamp_duty_cutover_date: str = "2023-08-28",
    slippage_bps: float = 0.0,
    max_volume_participation: Optional[float] = None,
    volume_lookback: int = 20,
    enforce_price_limits: bool = False,
    end_date: Optional[str] = None,
    hard_stop_pct: Optional[float] = None,
    trailing_stop_pct: Optional[float] = None,
    time_stop_days: Optional[int] = None,
    drawdown_policy=None,
) -> dict:
    """按下一交易日开盘执行选股轨迹并生成日频真实资金曲线。"""
    empty = {
        "equity_curve": [],
        "trade_log": [],
        "rebalance_log": [],
        "positions": {},
        "cash": float(initial_capital),
        "stop_loss_events": 0,
        "time_stop_events": 0,
        "risk_events": [],
        "execution_stats": {
            "total_fees": 0.0,
            "commission": 0.0,
            "transfer_fee": 0.0,
            "stamp_duty": 0.0,
            "slippage_cost": 0.0,
            "gross_turnover": 0.0,
            "constrained_orders": 0,
        },
    }
    if not selection_track or not price_data:
        return empty

    calendar = _market_calendar(price_data)
    if end_date is not None:
        calendar = calendar[calendar <= pd.Timestamp(end_date)]
    if len(calendar) == 0:
        return empty

    ordered_track = sorted(selection_track, key=lambda item: pd.Timestamp(item["date"]))
    events: dict[pd.Timestamp, dict] = {}
    for item in ordered_track:
        signal_date = pd.Timestamp(item["date"])
        execution_date = _next_session(calendar, signal_date)
        if execution_date is not None:
            events[execution_date] = item

    if not events:
        return empty

    first_signal = pd.Timestamp(ordered_track[0]["date"])
    active_calendar = calendar[calendar >= first_signal]
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    equity_curve: list[dict] = []
    trade_log: list[dict] = []
    rebalance_log: list[dict] = []
    stop_loss_events = 0
    time_stop_events = 0
    risk_events: list[dict] = []
    last_target: list[str] = []
    last_target_weights: Optional[dict[str, float]] = None
    previous_close_equity: Optional[float] = None
    previous_close_date: Optional[pd.Timestamp] = None
    current_risk_state = "normal"
    current_equity_cap = 1.0
    applied_equity_cap = 1.0

    for date in active_calendar:
        date = pd.Timestamp(date)

        risk_changed = False
        decision = None
        if drawdown_policy is not None and previous_close_equity is not None:
            decision = drawdown_policy.observe(
                previous_close_date, previous_close_equity
            )
            current_risk_state = decision.state
            current_equity_cap = decision.equity_cap
            risk_changed = abs(current_equity_cap - applied_equity_cap) > 1e-9

        event = events.get(date)
        if event is not None:
            last_target = list(event.get("picks", []))
            last_target_weights = event.get("target_weights")

        if event is not None or (risk_changed and last_target):
            signal_date = (
                pd.Timestamp(event["date"])
                if event is not None
                else pd.Timestamp(previous_close_date)
            )
            cash, positions, rebalance = _rebalance_at_open(
                date=date,
                signal_date=signal_date,
                target=last_target,
                target_weights=last_target_weights,
                cash=cash,
                positions=positions,
                price_data=price_data,
                commission=commission,
                min_lot=min_lot,
                min_commission=min_commission,
                transfer_fee_rate=transfer_fee_rate,
                stamp_duty_rate=stamp_duty_rate,
                historical_stamp_duty_rate=historical_stamp_duty_rate,
                stamp_duty_cutover_date=stamp_duty_cutover_date,
                slippage_bps=slippage_bps,
                max_volume_participation=max_volume_participation,
                volume_lookback=volume_lookback,
                enforce_price_limits=enforce_price_limits,
                gross_exposure_cap=current_equity_cap,
            )
            applied_equity_cap = current_equity_cap
            if event is not None:
                rebalance_log.append(rebalance)
            if risk_changed and decision is not None:
                actual_ratio = (
                    rebalance["stock_value"] / rebalance["equity"]
                    if rebalance["equity"] > 0 else 0.0
                )
                if actual_ratio > current_equity_cap + 0.01:
                    rebalance["constraints"].append({
                        "code": "PORTFOLIO",
                        "side": "sell",
                        "reason": "risk_cap_unfilled",
                        "target_equity_cap": current_equity_cap,
                        "actual_equity_ratio": actual_ratio,
                    })
                risk_events.append({
                    "observed_at_close": str(previous_close_date.date()),
                    "executed_at_open": str(date.date()),
                    "state": decision.state,
                    "drawdown": decision.drawdown,
                    "target_equity_cap": current_equity_cap,
                    "actual_equity_ratio": actual_ratio,
                    "reason": decision.reason,
                })
            trade_log.append(rebalance)

        day_sells: list[dict] = []
        day_constraints: list[dict] = []
        for code in list(positions):
            pos = positions[code]
            row = _row_on_date(price_data, code, date)
            if row is None:
                continue

            day_open = float(row["open"])
            day_low = float(row["low"])
            day_high = float(row["high"])
            day_close = float(row["close"])

            # A股新买入仓位当日不可卖出。
            if pd.Timestamp(pos["last_buy_date"]) == date:
                pos["highest"] = max(pos["highest"], day_high)
                continue

            levels: list[tuple[str, float]] = []
            if hard_stop_pct is not None and hard_stop_pct > 0:
                levels.append(("hard", pos["entry_price"] * (1 - hard_stop_pct)))
            if trailing_stop_pct is not None and trailing_stop_pct > 0:
                levels.append(("trailing", pos["highest"] * (1 - trailing_stop_pct)))

            if levels:
                stop_type, stop_level = max(levels, key=lambda item: item[1])
                if day_open <= stop_level:
                    exit_price = day_open
                elif day_low <= stop_level:
                    exit_price = stop_level
                else:
                    exit_price = None

                if exit_price is not None:
                    requested = pos["shares"]
                    if (
                        enforce_price_limits
                        and _is_limit_locked(price_data, code, date, "sell")
                    ):
                        day_constraints.append({
                            "code": code, "side": "sell",
                            "reason": "limit_down_locked",
                            "requested_shares": requested, "filled_shares": 0,
                        })
                        pos["highest"] = max(pos["highest"], day_high)
                        continue
                    cap = _capacity_shares(
                        price_data, code, date,
                        max_volume_participation=max_volume_participation,
                        volume_lookback=volume_lookback,
                        min_lot=min_lot,
                    )
                    shares = requested if cap is None else min(requested, cap)
                    shares = int(shares / min_lot) * min_lot
                    if shares <= 0:
                        day_constraints.append({
                            "code": code, "side": "sell",
                            "reason": "volume_capacity",
                            "requested_shares": requested, "filled_shares": 0,
                        })
                        pos["highest"] = max(pos["highest"], day_high)
                        continue
                    reference_price = exit_price
                    exit_price = _execution_price(
                        reference_price, "sell", slippage_bps
                    )
                    gross = shares * exit_price
                    fees = _fees(
                        gross_value=gross, side="sell", date=date,
                        commission_rate=commission,
                        min_commission=min_commission,
                        transfer_fee_rate=transfer_fee_rate,
                        stamp_duty_rate=stamp_duty_rate,
                        historical_stamp_duty_rate=historical_stamp_duty_rate,
                        stamp_duty_cutover_date=stamp_duty_cutover_date,
                    )
                    proceeds = gross - fees["total"]
                    cash += proceeds
                    day_sells.append({
                        "code": code,
                        "shares": shares,
                        "price": exit_price,
                        "reference_price": reference_price,
                        "gross_value": round(gross, 2),
                        "fees": {k: round(v, 2) for k, v in fees.items()},
                        "slippage_cost": round(
                            shares * (reference_price - exit_price), 2
                        ),
                        "proceeds": round(proceeds, 2),
                        "note": f"{stop_type}_stop",
                    })
                    pos["shares"] -= shares
                    if pos["shares"] <= 0:
                        positions.pop(code)
                    if shares < requested:
                        day_constraints.append({
                            "code": code, "side": "sell",
                            "reason": "volume_capacity",
                            "requested_shares": requested,
                            "filled_shares": shares,
                        })
                    stop_loss_events += 1
                    continue

            if (
                time_stop_days is not None
                and time_stop_days > 0
                and (date - pd.Timestamp(pos["entry_date"])).days >= time_stop_days
                and day_close <= pos["entry_price"] * 1.005
            ):
                requested = pos["shares"]
                if (
                    enforce_price_limits
                    and _is_limit_locked(price_data, code, date, "sell")
                ):
                    day_constraints.append({
                        "code": code, "side": "sell",
                        "reason": "limit_down_locked",
                        "requested_shares": requested, "filled_shares": 0,
                    })
                    pos["highest"] = max(pos["highest"], day_high)
                    continue
                cap = _capacity_shares(
                    price_data, code, date,
                    max_volume_participation=max_volume_participation,
                    volume_lookback=volume_lookback,
                    min_lot=min_lot,
                )
                shares = requested if cap is None else min(requested, cap)
                shares = int(shares / min_lot) * min_lot
                if shares <= 0:
                    day_constraints.append({
                        "code": code, "side": "sell",
                        "reason": "volume_capacity",
                        "requested_shares": requested, "filled_shares": 0,
                    })
                    pos["highest"] = max(pos["highest"], day_high)
                    continue
                reference_price = day_close
                exit_price = _execution_price(
                    reference_price, "sell", slippage_bps
                )
                gross = shares * exit_price
                fees = _fees(
                    gross_value=gross, side="sell", date=date,
                    commission_rate=commission,
                    min_commission=min_commission,
                    transfer_fee_rate=transfer_fee_rate,
                    stamp_duty_rate=stamp_duty_rate,
                    historical_stamp_duty_rate=historical_stamp_duty_rate,
                    stamp_duty_cutover_date=stamp_duty_cutover_date,
                )
                proceeds = gross - fees["total"]
                cash += proceeds
                day_sells.append({
                    "code": code,
                    "shares": shares,
                    "price": exit_price,
                    "reference_price": reference_price,
                    "gross_value": round(gross, 2),
                    "fees": {k: round(v, 2) for k, v in fees.items()},
                    "slippage_cost": round(
                        shares * (reference_price - exit_price), 2
                    ),
                    "proceeds": round(proceeds, 2),
                    "note": "time_stop",
                })
                pos["shares"] -= shares
                if pos["shares"] <= 0:
                    positions.pop(code)
                if shares < requested:
                    day_constraints.append({
                        "code": code, "side": "sell",
                        "reason": "volume_capacity",
                        "requested_shares": requested,
                        "filled_shares": shares,
                    })
                time_stop_events += 1
                continue

            pos["highest"] = max(pos["highest"], day_high)

        if day_sells or day_constraints:
            total_equity, stock_value = _portfolio_value(
                cash, positions, price_data, date, field="close"
            )
            trade_log.append({
                "date": date,
                "signal_date": None,
                "target": list(positions),
                "buys": [],
                "sells": day_sells,
                "skipped": [],
                "constraints": day_constraints,
                "risk_on": bool(positions),
                "cash_after": cash,
                "stock_value": stock_value,
                "equity": total_equity,
            })

        total_equity, stock_value = _portfolio_value(
            cash, positions, price_data, date, field="close"
        )
        equity_curve.append({
            "date": date,
            "cash": round(cash, 2),
            "stock_value": round(stock_value, 2),
            "equity": round(total_equity, 2),
            "risk_state": current_risk_state,
            "target_equity_cap": float(current_equity_cap),
            "actual_equity_ratio": (
                float(stock_value / total_equity) if total_equity > 0 else 0.0
            ),
        })
        previous_close_equity = float(total_equity)
        previous_close_date = date

    execution_stats = {
        "total_fees": 0.0,
        "commission": 0.0,
        "transfer_fee": 0.0,
        "stamp_duty": 0.0,
        "slippage_cost": 0.0,
        "gross_turnover": 0.0,
        "constrained_orders": 0,
    }
    for log in trade_log:
        execution_stats["constrained_orders"] += len(log.get("constraints", []))
        for order in log.get("buys", []) + log.get("sells", []):
            execution_stats["gross_turnover"] += float(
                order.get("gross_value", 0.0)
            )
            execution_stats["slippage_cost"] += float(
                order.get("slippage_cost", 0.0)
            )
            fee_detail = order.get("fees", {})
            for field in ("commission", "transfer_fee", "stamp_duty"):
                execution_stats[field] += float(fee_detail.get(field, 0.0))
            execution_stats["total_fees"] += float(
                fee_detail.get("total", 0.0)
            )
    execution_stats = {
        key: round(value, 2) if isinstance(value, float) else value
        for key, value in execution_stats.items()
    }

    return {
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "rebalance_log": rebalance_log,
        "positions": positions,
        "cash": round(cash, 2),
        "stop_loss_events": stop_loss_events,
        "time_stop_events": time_stop_events,
        "risk_events": risk_events,
        "execution_stats": execution_stats,
    }


def periodic_returns(
    equity_curve: list[dict],
    frequency: str = "weekly",
) -> list[float]:
    """从同一条日频净值曲线计算周期收益百分比。"""
    if not equity_curve:
        return []
    series = pd.Series(
        [row["equity"] for row in equity_curve],
        index=pd.DatetimeIndex([row["date"] for row in equity_curve]),
        dtype=float,
    )
    rule = {"weekly": "W-FRI", "monthly": "ME", "daily": None}[frequency]
    aligned = series if rule is None else series.resample(rule).last().dropna()
    return (aligned.pct_change().dropna() * 100).round(4).tolist()
