"""
周线前向模拟 + 三大模块集成版
================================
集成内容 (2026-07-13):
  1. 集中度风控 (concentration_filter)    — 选股后自动过滤连续超限标的
  2. 持仓状态保存 (daily_risk_monitor)    — 调仓后写入 portfolio_state.json
  3. Cron 兼容 (daily_accumulate 已修复)  — 本脚本同步支持 --single 模式

两种运行模式:
  [回测模式] python -m src.forward_simulation
     从 start_date 到 end_date 逐周模拟，输出完整报告

  [实盘模式] python -m src.forward_simulation --single
     仅生成本周选股信号，不跟踪历史，适合每周五 Cron 调用

流程:
  RSS抓取 → DeepSeek分析 → SQLite入库 (由 daily_accumulate.py 完成)
                              ↓
  价格数据 + 情感因子 → 打分排序 → 行业中性化选股
     → 集中度风控过滤 → 确定最终持仓
       → save_portfolio_state() → 日度监控读取
"""

import argparse
import copy
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import StrategyConfig
from src.dual_factor import (
    get_stock_mapping, load_sentiment_for_pool,
    build_price_factors, build_sentiment_factors, build_combined_factors,
)
from src.indicators import (
    calculate_position_sizes,
    compute_target_weights,
    select_with_rank_buffer,
)
from src.price_feed import load_pool_data, load_stock_pool
from src.execution_engine import (
    generate_rebalance_dates,
    periodic_returns,
    simulate_portfolio,
)
from src.universe import PointInTimeUniverse

# ── ★ 集成模块 1: 集中度风控 ──
from src.concentration_filter import ConcentrationFilter, apply_concentration_control

# ======================================================================
# 核心: 单周选股逻辑 (回测 & 实盘共用)
# ======================================================================

def select_weekly_picks(
    rebalance_date: str,
    price_data_full: dict[str, pd.DataFrame],
    name_to_code: dict[str, str],
    code_to_name: dict[str, str],
    code_to_sector: dict[str, str],
    config: StrategyConfig,
    concentration_filter: Optional[ConcentrationFilter] = None,
    top_n: int = 5,
    min_sqlite_days: int = 3,
    sentiment_weight_override: Optional[float] = None,
    max_per_sector_override: Optional[int] = None,
    eligible_codes: Optional[set[str]] = None,
    previous_picks: Optional[list[str]] = None,
    rank_buffer_override: Optional[int] = None,
) -> Optional[dict]:
    """在单个调仓日执行完整的选股流程。

    这是 forward_simulation 和单周实盘模式的共享核心。
    包含: 数据切片 → 情感加载 → 因子构建 → 打分 → 行业中性化 → 集中度风控

    Args:
        rebalance_date:    调仓日期 "YYYY-MM-DD"
        price_data_full:   全量价格数据 {code: DataFrame}
        name_to_code:      {名称: 代码}
        code_to_name:      {代码: 名称}
        code_to_sector:    {代码: 行业}
        config:            策略配置
        concentration_filter: 集中度过滤器实例 (可选, 传 None 则跳过滤)
        top_n:             目标持仓数
        min_sqlite_days:   SQLite 最少天数阈值
        sentiment_weight_override: 覆盖 config.scoring.sentiment_weight (grid search用)
        max_per_sector_override:   覆盖 config.sector_neutral.max_per_sector (grid search用)

    Returns:
        {"picks": [...], "ranked": DataFrame, "entry_prices": {...},
         "score_details": {...}, "backup_pool": [...]}
        或 None (数据不足无法选股)
    """
    date_dt = pd.Timestamp(rebalance_date)
    adjusted_factors = config.factors.for_timeframe(config.timeframe)

    # ---- 1. 截取 <= 调仓日的价格数据 ----
    price_snapshot: dict[str, pd.DataFrame] = {}
    for code, df in price_data_full.items():
        if eligible_codes is not None and code not in eligible_codes:
            continue
        df_slice = df[df.index <= date_dt].copy()
        if len(df_slice) >= 90:
            price_snapshot[code] = df_slice

    if len(price_snapshot) < top_n:
        logger.warning(
            f"[{rebalance_date}] 可用股票不足 {top_n} 只 "
            f"(仅有 {len(price_snapshot)}), 跳过"
        )
        return None

    scoring = copy.deepcopy(config.scoring)
    if sentiment_weight_override is not None:
        scoring.sentiment_weight = sentiment_weight_override

    # ---- 2. 加载情感数据（纯价格基准完全跳过） ----
    sentiment_data: dict[str, pd.Series] = {}
    if scoring.sentiment_weight != 0:
        active_name_to_code = {
            name: code for name, code in name_to_code.items()
            if code in price_snapshot
        }
        sentiment_data = load_sentiment_for_pool(
            active_name_to_code,
            "2020-01-01",
            rebalance_date,
            price_data=price_snapshot,
            min_sqlite_days=max(
                min_sqlite_days, config.sentiment_data.min_sqlite_days
            ),
            min_pool_coverage=config.sentiment_data.min_pool_coverage,
            allow_price_proxy=config.sentiment_data.allow_price_proxy,
        )

    # ---- 3. 构建因子 ----
    rb_dates = pd.DatetimeIndex([date_dt])
    price_factors = build_price_factors(price_snapshot, rb_dates, adjusted_factors)

    sentiment_factors = None
    if sentiment_data:
        sentiment_factors = build_sentiment_factors(
            sentiment_data, rb_dates, price_snapshot, adjusted_factors,
            rebalance_freq="weekly",
        )

    combined = build_combined_factors(price_factors, sentiment_factors, scoring)

    if len(combined) == 0:
        logger.warning(f"[{rebalance_date}] 无有效因子数据, 跳过")
        return None

    # ---- 4. 行业中性化选股 (初步) ----
    ranked = combined.sort_values("score", ascending=False)

    _max_per_sector = (
        max_per_sector_override
        if max_per_sector_override is not None
        else config.sector_neutral.max_per_sector
    )
    _rank_buffer = (
        rank_buffer_override
        if rank_buffer_override is not None
        else config.portfolio.rank_buffer
    )
    picks = select_with_rank_buffer(
        ranked=ranked,
        top_n=top_n,
        sectors=code_to_sector,
        max_per_sector=_max_per_sector,
        previous_picks=previous_picks,
        rank_buffer=_rank_buffer,
        sector_enabled=config.sector_neutral.enabled,
    )

    logger.info(
        f"[{rebalance_date}] 初选结果: "
        f"{[f'{c}({code_to_name.get(c,c)})' for c in picks]}"
    )

    # ---- 5. ★ 集中度风控过滤 (集成模块 1) ----
    if concentration_filter is not None:
        picks_before = picks.copy()
        picks = concentration_filter.apply(
            current_picks=picks,
            ranked_df=ranked,
            top_n=top_n,
            sectors=code_to_sector,
            code_to_name=code_to_name,
            rebalance_date=rebalance_date,
        )
        if picks != picks_before:
            removed = set(picks_before) - set(picks)
            added = set(picks) - set(picks_before)
            if removed:
                logger.info(
                    f"  [风控] 集中度过滤: 排除 {[code_to_name.get(c,c) for c in removed]}, "
                    f"补入 {[code_to_name.get(c,c) for c in added]}"
                )
    else:
        logger.info(f"  [风控] 未启用集中度过滤 (concentration_filter=None)")

    # ---- 6. 获取分数详情 ----
    score_details = {}
    for _, row in ranked.iterrows():
        if row["code"] in picks:
            score_details[row["code"]] = {
                "score": round(row["score"], 4),
                "score_price": round(row.get("score_price", 0), 4),
                "score_sentiment": round(row.get("score_sentiment", 0), 4),
                "momentum": round(row.get("momentum", 0), 4),
                "pullback": round(row.get("pullback", 0), 4),
                "sent_momentum": round(row.get("sent_momentum", 0), 4),
            }

    # ---- 7. 信号日参考价 ----
    # 这里只记录周五收盘时可见的价格，不能称为实际入场价。历史模拟统一
    # 交给 execution_engine 在下一交易日开盘成交。
    signal_prices = {}
    signal_data_dates = {}
    for code in picks:
        if code in price_snapshot:
            signal_prices[code] = round(
                float(price_snapshot[code]["close"].iloc[-1]), 2
            )
            signal_data_dates[code] = str(
                pd.Timestamp(price_snapshot[code].index[-1]).date()
            )

    # ---- 8. 备选池 (用于周中熔断替换) ----
    # 从排名中取接下来 3 只未选中的股票作为替补
    backup_pool = []
    for _, row in ranked.iterrows():
        code = row["code"]
        if code not in picks:
            backup_pool.append(code)
        if len(backup_pool) >= 3:
            break

    logger.info(
        f"[{rebalance_date}] 最终持仓: "
        f"{[f'{c}({code_to_name.get(c,c)})' for c in picks]}"
    )
    target_weights = compute_target_weights(
        target_codes=picks,
        price_data=price_snapshot,
        as_of=date_dt,
        method=config.portfolio.weighting,
        volatility_window=config.portfolio.risk_weight_window,
        max_weight=config.portfolio.max_position_weight,
    )

    return {
        "picks": picks,
        "target_weights": target_weights,
        "ranked": ranked,
        "signal_prices": signal_prices,
        "signal_data_dates": signal_data_dates,
        # 兼容旧调用方；语义已改为信号日参考价。
        "entry_prices": signal_prices,
        "score_details": score_details,
        "backup_pool": backup_pool,
    }


# ======================================================================
# 前向模拟 (回测模式)
# ======================================================================

def load_universe_context(
    config: StrategyConfig,
    start_date: str,
    end_date: str,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    Optional[PointInTimeUniverse],
]:
    """Load either a point-in-time HS300 universe or the legacy static pool."""
    if config.universe.source == "hs300_pit":
        snapshot_dir = Path(config.universe.snapshot_dir)
        if not snapshot_dir.is_absolute():
            snapshot_dir = PROJECT_ROOT / snapshot_dir
        store = PointInTimeUniverse(snapshot_dir)
        members = store.union_for_range(start_date, end_date)
        code_to_name = dict(zip(members["code"], members["name"]))
        code_to_sector = dict(zip(members["code"], members["sector"]))
        name_to_code = {name: code for code, name in code_to_name.items()}
        logger.info(
            f"历史沪深300股票池: {len(code_to_name)} 只区间并集，"
            f"逐调仓日按快照过滤"
        )
        return code_to_name, code_to_sector, name_to_code, store

    code_to_name, code_to_sector = load_stock_pool()
    _, name_to_code, _ = get_stock_mapping()
    return code_to_name, code_to_sector, name_to_code, None


def run_forward_simulation(
    start_date: str = "2026-05-20",
    end_date: str = "2026-07-13",
    top_n: int = 5,
    initial_capital: float = 100_000,
    min_sqlite_days: int = 3,
    enable_concentration: bool = True,
    lookback_weeks: int = 4,
    max_selects: int = 2,
    profile: str = "",
) -> Optional[dict]:
    """执行周线前向模拟。

    信号只使用信号日收盘前的数据，实际成交和资金曲线统一由
    :mod:`src.execution_engine` 按下一交易日开盘价计算。回测状态全部
    保存在内存中，不写入实盘持仓或集中度文件。
    """
    config = StrategyConfig.from_env(profile=profile or None)

    # ── 股票映射 ──
    code_to_name, code_to_sector, name_to_code, universe_store = (
        load_universe_context(config, start_date, end_date)
    )
    codes = list(code_to_name.keys())

    # ── 加载完整区间的价格数据 ──
    logger.info(
        f"加载价格数据 (含历史lookback): "
        f"{len(codes)} 只股票, 2020-01-01 ~ {end_date}"
    )
    price_data_full = load_pool_data(codes, "2020-01-01", end_date)

    # ── 生成每周最后一个真实交易日 ──
    weekly_dates = generate_rebalance_dates(
        price_data_full, start_date, end_date, frequency="weekly"
    )

    if len(weekly_dates) == 0:
        logger.error("无有效调仓日期！")
        return None

    logger.info(
        f"调仓日期 ({len(weekly_dates)} 周): "
        f"{weekly_dates[0].date()} ~ {weekly_dates[-1].date()}"
    )

    # ── 初始化纯内存集中度过滤器 ──
    cf = None
    if enable_concentration:
        cf = ConcentrationFilter(
            lookback_weeks=lookback_weeks,
            max_selects=max_selects,
            mode="exclude",
        )
        logger.info(
            f"[风控] 集中度过滤已启用: "
            f"回溯{lookback_weeks}周, 最多{max_selects}次, "
            "模式=exclude, 状态=回测隔离"
        )

    # ── 先生成完整信号轨迹 ──
    selection_track: list[dict] = []
    previous_picks: list[str] = []
    for rebalance_date in weekly_dates:
        date_str = rebalance_date.strftime("%Y-%m-%d")
        date_snapshot = (
            universe_store.snapshot_for(date_str)
            if universe_store is not None else None
        )
        result = select_weekly_picks(
            rebalance_date=date_str,
            price_data_full=price_data_full,
            name_to_code=(
                {name: code for code, name in date_snapshot.code_to_name.items()}
                if date_snapshot is not None else name_to_code
            ),
            code_to_name=(
                date_snapshot.code_to_name
                if date_snapshot is not None else code_to_name
            ),
            code_to_sector=(
                date_snapshot.code_to_sector
                if date_snapshot is not None else code_to_sector
            ),
            config=config,
            concentration_filter=cf,
            top_n=top_n,
            min_sqlite_days=min_sqlite_days,
            eligible_codes=(
                date_snapshot.codes if date_snapshot is not None else None
            ),
            previous_picks=previous_picks,
        )

        if result is None:
            continue

        if cf is not None:
            cf.save_history(result["picks"], date_str, persist=False)

        selection_track.append({
            "date": date_str,
            "picks": result["picks"],
            "target_weights": result["target_weights"],
            "signal_prices": result["signal_prices"],
            "score_details": result["score_details"],
            "backup_pool": result["backup_pool"],
        })
        previous_picks = list(result["picks"])

    if not selection_track:
        logger.error("无有效调仓周！")
        return None

    # ── 唯一执行口径：下一交易日开盘、真实股数/现金/手续费、日频净值 ──
    simulation = simulate_portfolio(
        selection_track=selection_track,
        price_data=price_data_full,
        initial_capital=initial_capital,
        commission=config.execution.commission,
        min_lot=config.execution.min_lot,
        min_commission=config.execution.min_commission,
        transfer_fee_rate=config.execution.transfer_fee_rate,
        stamp_duty_rate=config.execution.stamp_duty_rate,
        historical_stamp_duty_rate=(
            config.execution.historical_stamp_duty_rate
        ),
        stamp_duty_cutover_date=(
            config.execution.stamp_duty_cutover_date
        ),
        slippage_bps=config.execution.slippage_bps,
        max_volume_participation=(
            config.execution.max_volume_participation
        ),
        volume_lookback=config.execution.volume_lookback,
        enforce_price_limits=config.execution.enforce_price_limits,
        end_date=end_date,
    )
    daily_equity = simulation["equity_curve"]
    if not daily_equity:
        logger.error("无可执行交易日；最后一个信号之后可能缺少下一交易日行情")
        return None

    weekly_returns = periodic_returns(daily_equity, "weekly")
    capital = float(daily_equity[-1]["equity"])
    total_return = (capital / initial_capital - 1) * 100
    win_weeks = sum(ret > 0 for ret in weekly_returns)

    rebalance_by_signal = {
        log["signal_date"].strftime("%Y-%m-%d"): log
        for log in simulation["rebalance_log"]
    }
    weekly_records = []
    for i, signal in enumerate(selection_track, start=1):
        execution = rebalance_by_signal.get(signal["date"])
        weekly_records.append({
            "week": i,
            **signal,
            "execution_date": (
                execution["date"].strftime("%Y-%m-%d") if execution else None
            ),
            "executed_positions": {
                trade["code"]: trade["shares"]
                for trade in (execution or {}).get("buys", [])
            },
        })

    # ── 汇总报告 ──
    print(f"\n{'='*70}")
    print(f"  前向模拟汇总报告")
    print(f"{'='*70}")
    print(f"  模拟区间: {start_date} ~ {end_date}")
    print(f"  信号/成交: 每周最后交易日收盘后 / 下一交易日开盘")
    print(f"  持仓数:   Top {top_n} (行业中性化)")
    print(f"  集中度风控: {'启用' if enable_concentration else '关闭'}")
    print(f"  初始资金: {initial_capital:,.0f}")
    print(f"  最终权益: {capital:,.0f}")
    print(f"  累计收益: {total_return:+.2f}%")
    n_weeks = len(weekly_returns)
    win_rate = win_weeks / n_weeks * 100 if n_weeks else 0.0
    print(f"  胜率:     {win_weeks}/{n_weeks} ({win_rate:.1f}%)")
    if weekly_returns:
        print(f"  周均收益: {np.mean(weekly_returns):+.2f}%")
        print(f"  周收益波动: {np.std(weekly_returns):.2f}%")
    if weekly_returns and np.std(weekly_returns) > 0:
        weekly_sharpe = np.mean(weekly_returns) / np.std(weekly_returns) * np.sqrt(52)
        print(f"  年化夏普: {weekly_sharpe:.2f}")
        print(f"  最佳周:   {max(weekly_returns):+.2f}%")
        print(f"  最差周:   {min(weekly_returns):+.2f}%")

    # 选股频率统计
    print(f"\n  [被选中次数统计]")
    pick_counter = {}
    for r in weekly_records:
        for code in r["picks"]:
            pick_counter[code] = pick_counter.get(code, 0) + 1
    for code, count in sorted(pick_counter.items(), key=lambda x: -x[1]):
        name = code_to_name.get(code, code)
        sector = code_to_sector.get(code, "?")
        bar = "#" * count
        print(f"  {code} {name:6s} [{sector:6s}]: {count}次 {bar}")

    return {
        "weekly_records": weekly_records,
        "equity_curve": daily_equity,
        "trade_log": simulation["trade_log"],
        "total_return_pct": round(total_return, 2),
        "final_equity": round(capital, 2),
        "win_rate": round(win_rate, 1),
        "weekly_returns": weekly_returns,
        "pick_counter": pick_counter,
    }


# ======================================================================
# 单周实盘模式 (供每周五 Cron 调用)
# ======================================================================

def run_single_week(
    top_n: int = 5,
    initial_capital: float = 100_000,
    enable_concentration: bool = True,
    lookback_weeks: int = 4,
    max_selects: int = 2,
    profile: str = "",
) -> Optional[dict]:
    """仅生成本周选股信号 (实盘模式)。

    用于每周五 15:30 Cron 调用，输出本周信号并写入
    pending_portfolio_state.json；不会把估算价格写成已成交持仓。
    不跟踪前向收益（因为未来未发生）。

    Returns:
        {"picks": [...], "entry_prices": {...}, "positions_data": [...],
         "backup_pool": [...], "rebalance_date": "YYYY-MM-DD"}
    """
    config = StrategyConfig.from_env(profile=profile or None)

    # 计算本周五日期 (如果今天是周五就用今天, 否则用最近过去的周五)
    today = datetime.now()
    days_since_friday = (today.weekday() - 4) % 7
    if days_since_friday == 0:
        rebalance_date = today
    else:
        rebalance_date = today - timedelta(days=days_since_friday)
    date_str = rebalance_date.strftime("%Y-%m-%d")

    logger.info(f"[实盘模式] 调仓日: {date_str} (今天={today.strftime('%Y-%m-%d')})")

    # ── 股票映射 ──
    code_to_name, code_to_sector, name_to_code, universe_store = (
        load_universe_context(config, date_str, date_str)
    )
    codes = list(code_to_name.keys())

    # ── 加载价格数据 ──
    logger.info(f"加载价格数据: {len(codes)} 只股票, 2020-01-01 ~ {date_str}")
    price_data_full = load_pool_data(codes, "2020-01-01", date_str)
    market_data_as_of = max(
        (
            pd.Timestamp(frame.index.max())
            for frame in price_data_full.values()
            if not frame.empty
        ),
        default=None,
    )
    if market_data_as_of is None or market_data_as_of.date() != pd.Timestamp(
        date_str
    ).date():
        actual = (
            str(market_data_as_of.date())
            if market_data_as_of is not None else "无"
        )
        logger.error(
            "[实盘模式] 行情尚未覆盖信号日，拒绝生成正式计划: "
            f"信号日={date_str}, 行情截止={actual}"
        )
        return None

    # ── 初始化集中度过滤器 ──
    cf = None
    if enable_concentration:
        cf = ConcentrationFilter(
            lookback_weeks=lookback_weeks,
            max_selects=max_selects,
            mode="exclude",
        )
        cf.load_history()
        logger.info(
            f"[风控] 集中度过滤: 回溯{lookback_weeks}周, 最多{max_selects}次"
        )

    # ── 调用共享选股核心 ──
    date_snapshot = (
        universe_store.snapshot_for(date_str)
        if universe_store is not None else None
    )
    previous_picks: list[str] = []
    state_path = PROJECT_ROOT / "data" / "portfolio_state.json"
    if config.portfolio.rank_buffer > 0 and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            previous_picks = [
                str(position["code"])
                for position in state.get("positions", [])
                if position.get("code")
            ]
        except (OSError, ValueError, TypeError):
            logger.warning("无法读取现有持仓，排名缓冲将从空状态开始")
    result = select_weekly_picks(
        rebalance_date=date_str,
        price_data_full=price_data_full,
        name_to_code=(
            {name: code for code, name in date_snapshot.code_to_name.items()}
            if date_snapshot is not None else name_to_code
        ),
        code_to_name=(
            date_snapshot.code_to_name
            if date_snapshot is not None else code_to_name
        ),
        code_to_sector=(
            date_snapshot.code_to_sector
            if date_snapshot is not None else code_to_sector
        ),
        config=config,
        concentration_filter=cf,
        top_n=top_n,
        eligible_codes=(
            date_snapshot.codes if date_snapshot is not None else None
        ),
        previous_picks=previous_picks,
    )

    if result is None:
        logger.error("[实盘模式] 选股失败, 无法生成本周持仓")
        return None

    picks = result["picks"]
    target_weights = result["target_weights"]
    signal_prices = result["signal_prices"]
    signal_data_dates = result["signal_data_dates"]
    score_details = result["score_details"]
    backup_pool = result["backup_pool"]

    # ── 计算仓位 ──
    prices_for_sizing = {
        c: signal_prices[c] for c in picks if c in signal_prices
    }
    positions_dict, deployed, leftover = calculate_position_sizes(
        picks,
        prices_for_sizing,
        initial_capital,
        min_lot=config.execution.min_lot,
        commission=config.execution.commission,
        min_commission=config.execution.min_commission,
        transfer_fee_rate=config.execution.transfer_fee_rate,
        slippage_bps=config.execution.slippage_bps,
        target_weights=target_weights,
    )

    # 这里只能生成预计订单数量；真正的成交价和股数要在下一交易日确认。
    expected_execution_date = (
        pd.Timestamp(date_str) + pd.offsets.BDay(1)
    ).strftime("%Y-%m-%d")
    positions_data = []
    for code in picks:
        pos = positions_dict.get(code, {})
        positions_data.append({
            "code": code,
            "name": code_to_name.get(code, code),
            "sector": code_to_sector.get(code, "?"),
            "entry_date": expected_execution_date,
            "entry_price": signal_prices.get(code, 0),
            "signal_price": signal_prices.get(code, 0),
            "estimated_execution_price": pos.get(
                "price", signal_prices.get(code, 0)
            ),
            "shares": pos.get("shares", 0),
            "estimated_commission": pos.get("commission", 0),
            "estimated_transfer_fee": pos.get("transfer_fee", 0),
            "cost": pos.get("cost", 0),
            "status": "pending",
        })

    # 待执行计划与真实持仓分开保存，避免日度风控把参考收盘价当成交价。
    pending_path = _save_pending_order_plan(
        signal_date=date_str,
        expected_execution_date=expected_execution_date,
        positions_data=positions_data,
        backup_pool=backup_pool,
        initial_capital=initial_capital,
        estimated_deployed=deployed,
        estimated_leftover=leftover,
        market_data_as_of=str(market_data_as_of.date()),
        target_weights=target_weights,
        score_details=score_details,
        signal_data_dates=signal_data_dates,
    )

    # 如果正式模拟盘已初始化，周末选股完成后自动追加信号；这里只记录
    # 计划，不假设成交，也不会修改历史事件。
    try:
        from src.paper_trading import (
            DEFAULT_OUTPUT as PAPER_OUTPUT,
            capture_pending_plan,
            tracker_initialized,
        )
        if tracker_initialized(PAPER_OUTPUT):
            capture_pending_plan(pending_path, PAPER_OUTPUT)
            logger.info("[模拟盘] 本周信号已追加到账本")
    except Exception:
        logger.exception("[模拟盘] 信号追加失败；待执行计划已保留，可稍后重试")
        return None

    # ── 更新集中度历史 ──
    if cf is not None:
        cf.save_history(picks, date_str)

    # ── 终端输出 ──
    print(f"\n{'='*60}")
    print(f"  本周持仓信号 ({date_str})")
    print(f"{'='*60}")
    print(f"  {'代码':<8s} {'名称':<8s} {'行业':<8s} {'参考价':>8s} {'预计股数':>6s} {'预计成本':>10s}")
    print(f"  {'-'*52}")
    for p in positions_data:
        print(
            f"  {p['code']:<8s} {p['name']:<8s} {p['sector']:<8s} "
            f"{p['entry_price']:>8.2f} {p['shares']:>6d} {p['cost']:>10,.0f}"
        )
    print(f"  {'-'*52}")
    print(f"  部署资金: {deployed:,.0f} | 留存现金: {leftover:,.0f}")
    print(f"  备选池: {[code_to_name.get(c,c) for c in backup_pool]}")
    print(f"  待执行计划: {pending_path.relative_to(PROJECT_ROOT)}")
    print(f"  集中度记录: data/concentration_state.json")
    print(f"{'='*60}\n")

    # 打印评分详情
    print(f"  评分明细:")
    for code in picks:
        det = score_details.get(code, {})
        name = code_to_name.get(code, code)
        print(
            f"  {code} {name:6s} | "
            f"总分:{det.get('score',0):.3f} "
            f"(价格:{det.get('score_price',0):.2f} "
            f"情感:{det.get('score_sentiment',0):.2f}) | "
            f"动量:{det.get('momentum',0):.3f} "
            f"回撤:{det.get('pullback',0):.3f}"
        )

    # ── ★ 自动保存人类可读结果文件 (集成模块 3) ──
    _save_weekly_result_txt(
        rebalance_date=date_str,
        positions_data=positions_data,
        backup_pool=backup_pool,
        deployed=deployed,
        leftover=leftover,
        score_details=score_details,
        code_to_name=code_to_name,
        code_to_sector=code_to_sector,
        top_n=top_n,
        initial_capital=initial_capital,
        enable_concentration=enable_concentration,
        lookback_weeks=lookback_weeks,
        max_selects=max_selects,
        profile=profile,
    )

    return {
        "picks": picks,
        "signal_prices": signal_prices,
        "entry_prices": signal_prices,
        "positions_data": positions_data,
        "backup_pool": backup_pool,
        "rebalance_date": date_str,
        "deployed": deployed,
        "leftover": leftover,
        "expected_execution_date": expected_execution_date,
        "pending_order_path": str(pending_path),
    }


# ======================================================================
# 辅助: 保存待执行订单计划
# ======================================================================

def _save_pending_order_plan(
    *,
    signal_date: str,
    expected_execution_date: str,
    positions_data: list[dict],
    backup_pool: list[str],
    initial_capital: float,
    estimated_deployed: float,
    estimated_leftover: float,
    market_data_as_of: str,
    target_weights: dict[str, float],
    score_details: dict,
    signal_data_dates: dict[str, str],
) -> Path:
    """保存信号计划，不覆盖已经确认的实盘持仓状态。"""
    path = PROJECT_ROOT / "data" / "pending_portfolio_state.json"
    payload = {
        "status": "pending",
        "signal_date": signal_date,
        "market_data_as_of": market_data_as_of,
        "expected_execution_date": expected_execution_date,
        "pricing_note": "股数和成本按信号日收盘价估算，须以下一交易日实际成交回报为准",
        "positions": positions_data,
        "backup_pool": backup_pool,
        "target_weights": target_weights,
        "score_details": score_details,
        "signal_data_dates": signal_data_dates,
        "initial_capital": round(initial_capital, 2),
        "estimated_deployed": round(estimated_deployed, 2),
        "estimated_leftover": round(estimated_leftover, 2),
        "updated": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    logger.info(f"待执行订单计划已保存: {path}")
    return path


# ======================================================================
# 辅助: 保存人类可读结果文件
# ======================================================================

def _save_weekly_result_txt(
    rebalance_date: str,
    positions_data: list[dict],
    backup_pool: list[str],
    deployed: float,
    leftover: float,
    score_details: dict,
    code_to_name: dict[str, str],
    code_to_sector: dict[str, str],
    top_n: int = 5,
    initial_capital: float = 100_000,
    enable_concentration: bool = True,
    lookback_weeks: int = 4,
    max_selects: int = 2,
    profile: str = "",
) -> Path:
    """将本周选股结果保存为人类可读的 TXT 文件。

    写入 data/weekly_result.txt，方便周一直接用记事本打开看。
    """
    result_path = PROJECT_ROOT / "data" / "weekly_result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # 计算下周一日期
    dt = datetime.strptime(rebalance_date, "%Y-%m-%d")
    days_until_monday = (7 - dt.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = dt + timedelta(days=days_until_monday)

    lines = []
    lines.append("=" * 60)
    lines.append(f"  本周持仓信号")
    lines.append(f"  信号日期: {rebalance_date} (收盘后生成)")
    lines.append(f"  预计执行: {next_monday.strftime('%Y-%m-%d')} (下一交易日开盘，以实际成交为准)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  总资金:   {initial_capital:,.0f} 元")
    lines.append(f"  持仓数:   Top {top_n} (行业中性化)")
    lines.append(f"  集中度风控: {'启用' if enable_concentration else '关闭'} "
                 f"(回溯{lookback_weeks}周, 最多{max_selects}次)")
    lines.append("")
    lines.append(f"  {'代码':<8s} {'名称':<8s} {'行业':<8s} {'参考收盘':>8s} {'预计股数':>6s} {'预计成本':>10s} {'总分':>8s}")
    lines.append(f"  {'-'*60}")

    for p in positions_data:
        code = p["code"]
        det = score_details.get(code, {})
        lines.append(
            f"  {code:<8s} {p['name']:<8s} {p['sector']:<8s} "
            f"{p['entry_price']:>8.2f} {p['shares']:>6d} {p['cost']:>10,.0f} "
            f"{det.get('score', 0):>8.3f}"
        )

    lines.append(f"  {'-'*60}")
    lines.append(f"  合计投入: {deployed:,.0f} 元 | 留存现金: {leftover:,.0f} 元")
    lines.append("  注意: 上述价格、股数和成本均为下单估算，不是已成交持仓。")
    lines.append("")
    lines.append(f"  备选池 (若上述股票买不到, 按顺序替补):")
    for i, code in enumerate(backup_pool):
        name = code_to_name.get(code, code)
        sector = code_to_sector.get(code, "?")
        lines.append(f"    {i+1}. {code} {name} [{sector}]")
    lines.append("")
    lines.append("  [评分明细]")
    for p in positions_data:
        code = p["code"]
        det = score_details.get(code, {})
        lines.append(
            f"    {code} {p['name']:<6s} | "
            f"总分:{det.get('score',0):.3f} (价格:{det.get('score_price',0):.2f} "
            f"情感:{det.get('score_sentiment',0):.2f}) | "
            f"动量:{det.get('momentum',0):.3f} 回撤:{det.get('pullback',0):.3f}"
        )
    lines.append("")
    lines.append("=" * 60)
    lines.append("  免责声明: 本信号由量化模型自动生成, 仅供参考。")
    lines.append("  投资有风险, 入市需谨慎。")
    lines.append("=" * 60)

    content = "\n".join(lines)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"[结果保存] 人类可读结果已写入: {result_path}")
    return result_path


# ======================================================================
# CLI 入口
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="周线前向模拟 + 三大模块集成版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  [回测模式] python -m src.forward_simulation
  [实盘模式] python -m src.forward_simulation --single
  [自定义]   python -m src.forward_simulation --start 2026-06-01 --end 2026-07-13
  [无风控]   python -m src.forward_simulation --no-concentration
        """,
    )
    parser.add_argument(
        "--single", action="store_true",
        help="实盘模式: 仅生成本周选股信号 (适合每周五 Cron)",
    )
    parser.add_argument(
        "--start", type=str, default="2026-05-20",
        help="回测起始日期 (默认: 2026-05-20)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="回测结束日期 (默认: 今天)",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="持仓数 (默认: 5)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="初始资金 (默认: 100000)",
    )
    parser.add_argument(
        "--no-concentration", action="store_true",
        help="禁用集中度风控",
    )
    parser.add_argument(
        "--lookback-weeks", type=int, default=4,
        help="集中度回溯周数 (默认: 4)",
    )
    parser.add_argument(
        "--max-selects", type=int, default=2,
        help="窗口内最大允许选中次数 (默认: 2)",
    )
    parser.add_argument(
        "--min-sqlite-days", type=int, default=3,
        help="SQLite 最少天数阈值 (默认: 3)",
    )
    parser.add_argument(
        "--profile", type=str, default="",
        help="配置profile: 空=默认10万, '1M'=100万大赛策略",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    if args.single:
        # ── 实盘模式 ──
        result = run_single_week(
            top_n=args.top_n,
            initial_capital=args.capital,
            enable_concentration=not args.no_concentration,
            lookback_weeks=args.lookback_weeks,
            max_selects=args.max_selects,
            profile=args.profile,
        )
        sys.exit(0 if result else 1)
    else:
        # ── 回测模式 ──
        result = run_forward_simulation(
            start_date=args.start,
            end_date=end_date,
            top_n=args.top_n,
            initial_capital=args.capital,
            min_sqlite_days=args.min_sqlite_days,
            enable_concentration=not args.no_concentration,
            lookback_weeks=args.lookback_weeks,
            max_selects=args.max_selects,
            profile=args.profile,
        )
        sys.exit(0 if result else 1)
