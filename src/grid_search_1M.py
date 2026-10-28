"""
1M 策略 — 周度参数网格搜索
============================
基准扫描: TOP_N × MaxPerSector，固定纯价格、无个股止损
共 4 × 2 = 8 组合

设计原则:
  - 选股逻辑复用 select_weekly_picks() (与生产环境完全一致)
  - 止损模拟使用日线最低价 (保守估计，捕捉日内闪崩)
  - 选股结果缓存: STOP_LOSS 不影响选股 → 24 组选股 × 3 组止损 = 72 组合
  - 集中度风控: 每个组合独立实例，纯内存不写盘
  - 大盘择时: 网格搜索期间关闭 (避免干扰参数归因)

用法:
  python -m src.grid_search_1M --start 2020-01-01 --end 2025-05-12 --profile 1M
  python -m src.grid_search_1M --start 2024-01-01 --end 2025-05-12 --profile 1M  # 快速测试
"""

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import StrategyConfig
from src.dual_factor import get_stock_mapping
from src.price_feed import load_pool_data, load_stock_pool
from src.forward_simulation import select_weekly_picks, load_universe_context
from src.concentration_filter import ConcentrationFilter
from src.indicators import compute_performance
from src.execution_engine import (
    generate_rebalance_dates,
    periodic_returns,
    simulate_portfolio,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"

# ======================================================================
# 网格维度定义
# ======================================================================

TOP_N_VALUES = [5, 6, 7, 8]
MAX_PER_SECTOR_VALUES = [1, 2]
# 基准不让止损和伪情感参与参数优化，避免把过拟合后的风控误当 alpha。
STOP_LOSS_PAIRS = [(0.0, 0.0)]
SENTIMENT_WEIGHT_VALUES = [0.0]

# ======================================================================
# 共享数据加载
# ======================================================================

def load_shared_data(
    start_date: str,
    end_date: str,
    profile: str = "1M",
):
    """加载价格数据和股票映射 (所有 combo 共享)。

    Returns:
        price_data_full: {code: DataFrame} 日线OHLCV
        code_to_name, code_to_sector, name_to_code: 股票映射
        config: StrategyConfig (用作基线)
    """
    config = StrategyConfig.from_env(profile=profile or None)

    code_to_name, code_to_sector, name_to_code, universe_store = (
        load_universe_context(config, start_date, end_date)
    )
    codes = list(code_to_name.keys())

    logger.info(f"加载价格数据: {len(codes)} 只, {start_date} ~ {end_date}")
    price_data_full = load_pool_data(codes, start_date, end_date)

    return (
        price_data_full, code_to_name, code_to_sector, name_to_code,
        config, universe_store,
    )


def generate_weekly_dates(
    price_data_full: dict,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    """生成周度调仓日期 (周五)。

    从第一只股票的日历来对齐交易日。
    """
    return generate_rebalance_dates(
        price_data_full, start_date, end_date, frequency="weekly"
    )


# ======================================================================
# 选股轨迹生成 (缓存层)
# ======================================================================

def generate_selection_track(
    weekly_dates: pd.DatetimeIndex,
    price_data_full: dict,
    name_to_code: dict,
    code_to_name: dict,
    code_to_sector: dict,
    config: StrategyConfig,
    top_n: int,
    max_per_sector: int,
    sentiment_weight: float,
    universe_store=None,
    min_sqlite_days: int = 3,
) -> list[dict]:
    """对一组参数生成完整的选股轨迹。

    逐周调用 select_weekly_picks()，使用独立的内存集中度过滤器。

    Returns:
        [{date, picks, entry_prices, ranked_df, score_details}, ...]
    """
    import tempfile, uuid

    # 独立 ConcentationFilter: 每个 combo 用独立的临时文件 (避免交叉污染)
    tmp_dir = Path(tempfile.gettempdir()) / "quant_gs_1M"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"cf_{uuid.uuid4().hex[:8]}.json"

    cf = ConcentrationFilter(
        lookback_weeks=4,
        max_selects=2,
        mode="exclude",
        state_path=tmp_path,
    )

    track = []
    previous_picks: list[str] = []
    for i, rebalance_date in enumerate(weekly_dates):
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
            sentiment_weight_override=sentiment_weight,
            max_per_sector_override=max_per_sector,
            eligible_codes=(
                date_snapshot.codes if date_snapshot is not None else None
            ),
            previous_picks=previous_picks,
        )

        if result is None:
            continue

        # 更新集中度历史 (写入独立文件，不污染其他 combo)
        cf.save_history(result["picks"], date_str, persist=False)

        track.append({
            "date": date_str,
            "picks": result["picks"],
            "target_weights": result["target_weights"],
            "entry_prices": result["entry_prices"],
            "ranked_df": result["ranked"],
            "score_details": result["score_details"],
        })
        previous_picks = list(result["picks"])

    # 清理临时文件
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    return track


# ======================================================================
# 日线止损模拟
# ======================================================================

def simulate_daily_stops(
    selection_track: list[dict],
    price_data_full: dict,
    hard_stop_pct: float,
    trailing_stop_pct: float,
    initial_capital: float,
    commission: float = 0.0005,
    min_lot: int = 100,
    end_date: Optional[str] = None,
    execution_config=None,
) -> dict:
    """使用全项目统一执行器模拟下一交易日开盘成交和日内止损。"""
    realistic_execution = {}
    if execution_config is not None:
        realistic_execution = {
            "min_commission": execution_config.min_commission,
            "transfer_fee_rate": execution_config.transfer_fee_rate,
            "stamp_duty_rate": execution_config.stamp_duty_rate,
            "historical_stamp_duty_rate": (
                execution_config.historical_stamp_duty_rate
            ),
            "stamp_duty_cutover_date": (
                execution_config.stamp_duty_cutover_date
            ),
            "slippage_bps": execution_config.slippage_bps,
            "max_volume_participation": (
                execution_config.max_volume_participation
            ),
            "volume_lookback": execution_config.volume_lookback,
            "enforce_price_limits": execution_config.enforce_price_limits,
        }
    result = simulate_portfolio(
        selection_track=selection_track,
        price_data=price_data_full,
        initial_capital=initial_capital,
        commission=commission,
        min_lot=min_lot,
        end_date=end_date,
        hard_stop_pct=hard_stop_pct,
        trailing_stop_pct=trailing_stop_pct,
        **realistic_execution,
    )
    result["weekly_returns"] = periodic_returns(
        result["equity_curve"], frequency="weekly"
    )
    return result


# ======================================================================
# 绩效计算
# ======================================================================

def compute_simulation_performance(
    equity_curve: list[dict],
    weekly_returns: list[float],
    initial_capital: float,
    stop_loss_events: int,
) -> dict:
    """从权益曲线计算绩效指标。"""
    if len(equity_curve) < 2:
        return {
            "total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_drawdown_pct": 0.0, "calmar": 0.0,
            "win_rate_pct": 0.0, "stop_loss_events": stop_loss_events,
            "n_weeks": len(equity_curve), "final_equity": initial_capital,
        }

    eq_series = pd.Series(
        [e["equity"] for e in equity_curve],
        index=[e["date"] for e in equity_curve],
    )

    # 权益曲线统一为日频，因此 Sharpe 和最大回撤也按日频计算。
    perf = compute_performance(eq_series, periods_per_year=252)

    # Sortino ratio (手动计算)
    returns = pd.Series(weekly_returns) / 100.0  # 转为小数
    rf_weekly = (1 + 0.02) ** (1 / 52) - 1
    excess = returns - rf_weekly
    downside = excess[excess < 0]
    sortino = (
        excess.mean() / downside.std() * np.sqrt(52)
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )

    n_weeks = len(weekly_returns)
    win_weeks = sum(1 for r in weekly_returns if r > 0)
    win_rate = win_weeks / n_weeks * 100 if n_weeks > 0 else 0

    final_equity = eq_series.iloc[-1]

    return {
        "total_return_pct": perf.get("total_return", 0.0),
        "cagr_pct": perf.get("cagr", 0.0),
        "sharpe": perf.get("sharpe", 0.0),
        "sortino": round(sortino, 4),
        "max_drawdown_pct": perf.get("max_drawdown", 0.0),
        "calmar": perf.get("calmar", 0.0),
        "win_rate_pct": round(win_rate, 1),
        "stop_loss_events": stop_loss_events,
        "n_weeks": n_weeks,
        "final_equity": round(final_equity, 2),
    }


# ======================================================================
# 单次模拟 (选股 + 止损)
# ======================================================================

def run_single_simulation(
    weekly_dates: pd.DatetimeIndex,
    price_data_full: dict,
    name_to_code: dict,
    code_to_name: dict,
    code_to_sector: dict,
    config: StrategyConfig,
    top_n: int,
    max_per_sector: int,
    sentiment_weight: float,
    hard_stop_pct: float,
    trailing_stop_pct: float,
    initial_capital: float,
    commission: float,
    selection_track_cache: dict,
    universe_store=None,
    end_date: Optional[str] = None,
) -> dict:
    """运行单组参数的完整模拟。

    选股轨迹使用缓存，止损模拟每次都重新跑。
    """
    cache_key = (top_n, max_per_sector, sentiment_weight)

    # ---- 选股 (缓存) ----
    if cache_key not in selection_track_cache:
        logger.info(
            f"  生成选股轨迹: TOP_N={top_n} MaxSec={max_per_sector} SentW={sentiment_weight}"
        )
        track = generate_selection_track(
            weekly_dates=weekly_dates,
            price_data_full=price_data_full,
            name_to_code=name_to_code,
            code_to_name=code_to_name,
            code_to_sector=code_to_sector,
            config=config,
            top_n=top_n,
            max_per_sector=max_per_sector,
            sentiment_weight=sentiment_weight,
            universe_store=universe_store,
        )
        selection_track_cache[cache_key] = track
        logger.info(f"    完成: {len(track)} 周选股记录")
    else:
        track = selection_track_cache[cache_key]

    # ---- 止损模拟 ----
    sim_result = simulate_daily_stops(
        selection_track=track,
        price_data_full=price_data_full,
        hard_stop_pct=hard_stop_pct,
        trailing_stop_pct=trailing_stop_pct,
        initial_capital=initial_capital,
        commission=commission,
        end_date=end_date,
        execution_config=config.execution,
    )

    # ---- 绩效 ----
    perf = compute_simulation_performance(
        equity_curve=sim_result["equity_curve"],
        weekly_returns=sim_result["weekly_returns"],
        initial_capital=initial_capital,
        stop_loss_events=sim_result["stop_loss_events"],
    )

    return perf


# ======================================================================
# 主网格搜索
# ======================================================================

def run_grid_search(
    start_date: str = "2020-01-01",
    end_date: str = "2025-05-12",
    profile: str = "1M",
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """执行 1M 策略周度网格搜索。

    扫描空间: 4 × 2 × 3 × 3 = 72 组合
    优化: 选股轨迹按 (TOP_N, MaxPerSector, SENTIMENT_WEIGHT) 缓存
    """
    # ---- 加载共享数据 ----
    (
        price_data_full, code_to_name, code_to_sector, name_to_code,
        config, universe_store,
    ) = load_shared_data(start_date, end_date, profile)

    # 大盘择时: 网格搜索期间关闭 (避免干扰参数归因)
    config.market_timing.enabled = False

    commission = config.execution.commission

    # ---- 生成周度调仓日期 ----
    weekly_dates = generate_weekly_dates(price_data_full, start_date, end_date)
    logger.info(f"周度调仓日期: {len(weekly_dates)} 周 ({weekly_dates[0].date()} ~ {weekly_dates[-1].date()})")

    total_combos = len(TOP_N_VALUES) * len(MAX_PER_SECTOR_VALUES) * len(SENTIMENT_WEIGHT_VALUES) * len(STOP_LOSS_PAIRS)
    logger.info(f"扫描空间: {len(TOP_N_VALUES)} TOP_N × {len(MAX_PER_SECTOR_VALUES)} MaxSec "
                f"× {len(SENTIMENT_WEIGHT_VALUES)} SentW × {len(STOP_LOSS_PAIRS)} StopPairs = {total_combos} 组合")

    # ---- 选股轨迹缓存 ----
    selection_track_cache: dict[tuple, list[dict]] = {}

    # ---- 逐组合模拟 ----
    results = []
    progress = 0
    best_sharpe = -999
    best_combo = ""

    for top_n in TOP_N_VALUES:
        for max_per_sector in MAX_PER_SECTOR_VALUES:
            for sentiment_weight in SENTIMENT_WEIGHT_VALUES:
                for hard_stop, trailing_stop in STOP_LOSS_PAIRS:
                    progress += 1

                    perf = run_single_simulation(
                        weekly_dates=weekly_dates,
                        price_data_full=price_data_full,
                        name_to_code=name_to_code,
                        code_to_name=code_to_name,
                        code_to_sector=code_to_sector,
                        config=config,
                        top_n=top_n,
                        max_per_sector=max_per_sector,
                        sentiment_weight=sentiment_weight,
                        hard_stop_pct=hard_stop,
                        trailing_stop_pct=trailing_stop,
                        initial_capital=initial_capital,
                        commission=commission,
                        selection_track_cache=selection_track_cache,
                        universe_store=universe_store,
                        end_date=end_date,
                    )

                    row = {
                        "top_n": top_n,
                        "max_per_sector": max_per_sector,
                        "sentiment_weight": sentiment_weight,
                        "hard_stop_pct": hard_stop,
                        "trailing_stop_pct": trailing_stop,
                        **perf,
                    }
                    results.append(row)

                    # 更新最佳
                    if perf["sharpe"] > best_sharpe:
                        best_sharpe = perf["sharpe"]
                        best_combo = (
                            f"TOP_N={top_n} MaxSec={max_per_sector} "
                            f"SentW={sentiment_weight} Stop={hard_stop*100:.0f}/{trailing_stop*100:.0f}%"
                        )

                    # 进度输出
                    print(
                        f"  [{progress:3d}/{total_combos}] "
                        f"TOP_N={top_n} MaxSec={max_per_sector} "
                        f"SentW={sentiment_weight} Stop={hard_stop*100:.0f}/{trailing_stop*100:.0f}% | "
                        f"Sharpe={perf['sharpe']:.3f} CAGR={perf['cagr_pct']:.1f}% "
                        f"MaxDD={perf['max_drawdown_pct']:.1f}% Calmar={perf['calmar']:.3f} "
                        f"Wins={perf['win_rate_pct']:.0f}% Stops={perf['stop_loss_events']}"
                        f"{' ★BEST' if perf['sharpe'] == best_sharpe else ''}"
                    )

    logger.info(f"最佳组合: {best_combo} (Sharpe={best_sharpe:.3f})")

    df = pd.DataFrame(results)
    return df


# ======================================================================
# 输出
# ======================================================================

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def print_best_results(df: pd.DataFrame, top_n: int = 15) -> None:
    """打印最佳结果排名表。"""
    by = "sharpe"
    best = df.nlargest(top_n, by).copy()

    W = 110
    print()
    print("=" * W)
    print(f"  1M 策略周度网格搜索 — Top {top_n} 最优 (按 Sharpe)")
    print("=" * W)
    header = (
        f"  {'TOP_N':>5s} {'MaxSec':>6s} {'SentW':>5s} "
        f"{'硬止%':>5s} {'拖尾%':>5s} "
        f"{'Sharpe':>7s} {'CAGR%':>7s} {'MaxDD%':>7s} {'Calmar':>7s} "
        f"{'Sortino':>7s} {'Win%':>6s} {'StopEvt':>7s} {'收益%':>8s} {'N周':>5s}"
    )
    print(header)
    print("-" * W)

    for _, row in best.iterrows():
        sharpe = row["sharpe"]
        color = GREEN if sharpe > 1.0 else (YELLOW if sharpe > 0.5 else RED)
        dd = row["max_drawdown_pct"]
        dd_color = GREEN if abs(dd) < 10 else (YELLOW if abs(dd) < 15 else RED)
        print(
            f"  {int(row['top_n']):>5d} {int(row['max_per_sector']):>6d} {row['sentiment_weight']:>5.1f} "
            f"{row['hard_stop_pct']*100:>5.0f} {row['trailing_stop_pct']*100:>5.0f} "
            f"{color}{sharpe:>7.3f}{RESET} "
            f"{row['cagr_pct']:>7.1f} {dd_color}{dd:>7.1f}{RESET} "
            f"{row['calmar']:>7.3f} {row['sortino']:>7.3f} "
            f"{row['win_rate_pct']:>6.1f} {int(row['stop_loss_events']):>7d} "
            f"{row['total_return_pct']:>8.1f} {int(row['n_weeks']):>5d}"
        )

    print("=" * W)

    # 按 TOP_N 分组统计
    print(f"\n  各 TOP_N 均值 (按 Sharpe):")
    grouped = df.groupby("top_n")["sharpe"].agg(["mean", "max", "count"]).sort_values("mean", ascending=False)
    for top_n_val, grp in grouped.iterrows():
        print(f"  TOP_N={int(top_n_val)}:  Mean Sharpe={grp['mean']:.3f}  Best={grp['max']:.3f}  N={int(grp['count'])}")

    # 按 MaxPerSector 分组
    print(f"\n  各 MaxPerSector 均值 (按 Sharpe):")
    grouped2 = df.groupby("max_per_sector")["sharpe"].agg(["mean", "max"]).sort_values("mean", ascending=False)
    for mps, grp in grouped2.iterrows():
        print(f"  MaxPerSector={int(mps)}:  Mean Sharpe={grp['mean']:.3f}  Best={grp['max']:.3f}")

    # 按 SENTIMENT_WEIGHT 分组
    print(f"\n  各 Sentiment Weight 均值 (按 Sharpe):")
    grouped3 = df.groupby("sentiment_weight")["sharpe"].agg(["mean", "max"]).sort_values("mean", ascending=False)
    for sw, grp in grouped3.iterrows():
        print(f"  SentW={sw:.1f}:  Mean Sharpe={grp['mean']:.3f}  Best={grp['max']:.3f}")

    # 按止损对分组
    print(f"\n  各止损对均值 (按 Sharpe):")
    df["stop_label"] = df.apply(
        lambda r: f"硬{r['hard_stop_pct']*100:.0f}%/拖{r['trailing_stop_pct']*100:.0f}%", axis=1
    )
    grouped4 = df.groupby("stop_label")["sharpe"].agg(["mean", "max"]).sort_values("mean", ascending=False)
    for sl, grp in grouped4.iterrows():
        print(f"  {sl}:  Mean Sharpe={grp['mean']:.3f}  Best={grp['max']:.3f}")

    print()


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="1M 策略周度参数网格搜索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.grid_search_1M --start 2020-01-01 --end 2025-05-12 --profile 1M
  python -m src.grid_search_1M --start 2024-01-01 --end 2025-05-12  # 快速测试
  python -m src.grid_search_1M --start 2020-01-01 --end 2025-05-12 --export
        """,
    )
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2025-05-12")
    parser.add_argument("--profile", type=str, default="1M")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--export", action="store_true", help="导出CSV到 data/backtest_results/")
    parser.add_argument("--show", type=int, default=15, help="显示 Top N")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    df = run_grid_search(
        start_date=args.start,
        end_date=args.end,
        profile=args.profile,
        initial_capital=args.capital,
    )

    print_best_results(df, top_n=args.show)

    # 导出
    if args.export:
        export_dir = RESULTS_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = export_dir / f"grid_search_1M_{ts}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  完整结果已导出: {csv_path} ({len(df)} 行)")

    # 推荐最优
    best = df.nlargest(1, "sharpe").iloc[0]
    print(f"\n  >>> 推荐 1M 策略配置:")
    print(f"      TOP_N:           {int(best['top_n'])}")
    print(f"      MAX_PER_SECTOR:  {int(best['max_per_sector'])}")
    print(f"      SENTIMENT_WEIGHT: {best['sentiment_weight']:.1f}")
    print(f"      STOP_LOSS_PCT:   {best['hard_stop_pct']*100:.0f}%")
    print(f"      TRAILING_STOP_PCT: {best['trailing_stop_pct']*100:.0f}%")
    print(f"      ---")
    print(f"      Sharpe:  {best['sharpe']:.3f}")
    print(f"      CAGR:    {best['cagr_pct']:.1f}%")
    print(f"      MaxDD:   {best['max_drawdown_pct']:.1f}%")
    print(f"      Calmar:  {best['calmar']:.3f}")
    print(f"      WinRate: {best['win_rate_pct']:.0f}%")
    print(f"      Returns: {best['total_return_pct']:.1f}%")


if __name__ == "__main__":
    main()
