"""Point-in-time tradability filters for the v2 research universe."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TradabilityRules:
    min_average_amount: float
    amount_lookback: int
    top_n: int
    initial_cash: float
    min_lot: int
    commission: float
    min_commission: float
    transfer_fee_rate: float
    slippage_bps: float


def _lot_cost(close_price: float, rules: TradabilityRules) -> float:
    execution_price = close_price * (1 + max(rules.slippage_bps, 0.0) / 10_000)
    gross = execution_price * rules.min_lot
    commission = (
        max(gross * max(rules.commission, 0.0), rules.min_commission)
        if gross > 0 and rules.commission > 0
        else 0.0
    )
    transfer_fee = gross * max(rules.transfer_fee_rate, 0.0)
    return gross + commission + transfer_fee


def screen_tradable_codes(
    snapshot: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    signal_date: str | pd.Timestamp,
    rules: TradabilityRules,
) -> tuple[set[str], pd.DataFrame, dict]:
    """Filter using information available no later than ``signal_date``."""
    signal = pd.Timestamp(signal_date).normalize()
    target_budget = rules.initial_cash / max(rules.top_n, 1)
    rows: list[dict] = []

    for item in snapshot.to_dict("records"):
        code = str(item.get("code", "")).zfill(6)
        frame = price_data.get(code)
        reason = "eligible"
        observations = 0
        average_amount = None
        close_price = None
        lot_cost = None

        if frame is None or frame.empty:
            reason = "suspended_or_missing_price"
        else:
            local = frame.copy()
            local.index = pd.DatetimeIndex(local.index).normalize()
            history = local[local.index <= signal]
            if signal not in local.index or history.empty:
                reason = "suspended_or_missing_price"
            elif "amount" not in history.columns:
                reason = "missing_amount"
            else:
                amounts = pd.to_numeric(
                    history["amount"], errors="coerce"
                ).dropna().tail(max(rules.amount_lookback, 1))
                observations = int(len(amounts))
                if observations < max(rules.amount_lookback, 1):
                    reason = "insufficient_amount_history"
                else:
                    average_amount = float(amounts.mean())
                    if average_amount < rules.min_average_amount:
                        reason = "low_liquidity"
                    else:
                        close_value = history.loc[signal, "close"]
                        if isinstance(close_value, pd.Series):
                            close_value = close_value.iloc[-1]
                        close_price = float(close_value)
                        lot_cost = _lot_cost(close_price, rules)
                        if lot_cost > target_budget + 1e-8:
                            reason = "unaffordable_lot"

        rows.append({
            "date": str(signal.date()),
            "code": code,
            "name": str(item.get("name", "")),
            "sector": str(item.get("sector", "未知")),
            "eligible": reason == "eligible",
            "reason": reason,
            "amount_observations": observations,
            "average_amount_20d": average_amount,
            "signal_close": close_price,
            "minimum_lot_cost": lot_cost,
            "target_budget": target_budget,
        })

    details = pd.DataFrame(rows)
    eligible = set(
        details.loc[details["eligible"], "code"].astype(str)
        if not details.empty else []
    )
    audit = {
        "date": str(signal.date()),
        "snapshot_members": int(len(snapshot)),
        "excluded_suspended_or_missing_price": int(
            (details.get("reason") == "suspended_or_missing_price").sum()
        ) if not details.empty else 0,
        "excluded_missing_amount": int(
            (details.get("reason") == "missing_amount").sum()
        ) if not details.empty else 0,
        "excluded_insufficient_amount_history": int(
            (details.get("reason") == "insufficient_amount_history").sum()
        ) if not details.empty else 0,
        "excluded_low_liquidity": int(
            (details.get("reason") == "low_liquidity").sum()
        ) if not details.empty else 0,
        "excluded_unaffordable_lot": int(
            (details.get("reason") == "unaffordable_lot").sum()
        ) if not details.empty else 0,
        "final_eligible": int(len(eligible)),
        "minimum_average_amount": float(rules.min_average_amount),
        "amount_lookback": int(rules.amount_lookback),
        "target_budget": float(target_budget),
    }
    return eligible, details, audit
