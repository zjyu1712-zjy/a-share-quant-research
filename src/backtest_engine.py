"""
统一回测引擎
- 支持 price_only / sentiment_only / hybrid 三种模式
- 月频/周频调仓，含大盘择时、止损、时间止损、行业中性化
- 输出权益曲线、绩效指标、交易日志、图表导出
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.config import StrategyConfig
from src.indicators import (
    zscore, calculate_position_sizes, compute_performance,
    compute_target_weights, select_with_rank_buffer,
)
from src.price_feed import (
    load_price_data, load_index_data, load_pool_data,
    load_stock_pool as load_sp,
)
from src.dual_factor import (
    build_price_factors, build_sentiment_factors, build_combined_factors,
    load_sentiment_for_pool, get_stock_mapping,
)
from src.execution_engine import generate_rebalance_dates, simulate_portfolio
from src.universe import PointInTimeUniverse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MY_QUANT_DIR = PROJECT_ROOT / "MyQuantAgent"

BacktestMode = Literal["price_only", "sentiment_only", "hybrid"]


def _generate_rebalance_dates(
    start: str,
    end: str,
    frequency: str,
    price_data: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """生成调仓日期序列 — 对齐实际交易日历。

    使用 resample 获取每月/每周最后一个交易日，而非日历日。
    复现 legacy 的 resample('ME').last() 语义。
    """
    dates = generate_rebalance_dates(
        price_data, start, end, frequency=frequency
    )

    if len(dates) == 0:
        raise ValueError(f"日期范围内无调仓日: {start} ~ {end}")
    return dates


class BacktestEngine:
    """统一回测引擎。"""

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig.from_env()
        self.price_data: dict[str, pd.DataFrame] = {}
        self.sentiment_data: dict[str, pd.Series] = {}
        self.index_data: pd.DataFrame = pd.DataFrame()

        # 股票映射
        self.code_to_name: dict[str, str] = {}
        self.name_to_code: dict[str, str] = {}
        self.code_to_sector: dict[str, str] = {}
        self.universe_store: Optional[PointInTimeUniverse] = None

        # 回测结果
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.trade_log: list[dict] = []
        self.results: dict = {}
        self.stats: dict = {}
        self.run_metadata: dict = {}

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_price_data(
        self,
        start: str,
        end: str,
        codes: Optional[list[str]] = None,
    ) -> "BacktestEngine":
        """加载全池价格数据 + 沪深300指数。"""
        point_in_time_sources = {"hs300_pit", "sse_composite_a_pit"}
        if self.config.universe.source in point_in_time_sources:
            snapshot_dir = Path(self.config.universe.snapshot_dir)
            if not snapshot_dir.is_absolute():
                snapshot_dir = PROJECT_ROOT / snapshot_dir
            self.universe_store = PointInTimeUniverse(snapshot_dir)
            members = self.universe_store.union_for_range(start, end)
            self.code_to_name = dict(zip(members["code"], members["name"]))
            self.code_to_sector = dict(zip(members["code"], members["sector"]))
            self.name_to_code = {
                name: code for code, name in self.code_to_name.items()
            }
            if codes is None:
                codes = members["code"].tolist()
            label = (
                "上证综指可交易A股"
                if self.config.universe.source == "sse_composite_a_pit"
                else "沪深300成分"
            )
            logger.info(
                f"使用历史{label}快照: {members['code'].nunique()} 只区间并集"
            )
        else:
            self.code_to_name, self.code_to_sector = load_sp()
            _, self.name_to_code, _ = get_stock_mapping()
            if codes is None:
                codes = list(self.code_to_name.keys())

        logger.info(f"加载价格数据: {len(codes)} 只股票 + CSI 300...")
        self.price_data = load_pool_data(
            codes,
            start,
            end,
            require_amount=(
                self.config.universe.source == "sse_composite_a_pit"
            ),
        )

        # 加载沪深300指数
        if self.config.market_timing.enabled:
            idx_code = self.config.market_timing.index
            try:
                self.index_data = load_index_data(idx_code, start, end)
                logger.info(f"CSI 300: {len(self.index_data)} 条")
            except Exception:
                logger.warning("CSI 300 数据加载失败，关闭大盘择时")
                self.config.market_timing.enabled = False

        return self

    def load_sentiment_data(
        self,
        start: str,
        end: str,
    ) -> "BacktestEngine":
        """加载真实新闻情感；覆盖不足时禁用情感因子。"""
        logger.info(f"加载情感数据: {start} ~ {end}")
        self.sentiment_data = load_sentiment_for_pool(
            self.name_to_code, start, end,
            price_data=self.price_data,
            min_sqlite_days=self.config.sentiment_data.min_sqlite_days,
            min_pool_coverage=self.config.sentiment_data.min_pool_coverage,
            allow_price_proxy=self.config.sentiment_data.allow_price_proxy,
        )
        return self

    def build_price_factor_frame(
        self,
        rebalance_dates: pd.DatetimeIndex,
        adjusted_factors,
    ) -> pd.DataFrame:
        """构建价格因子；研究验证器可覆写，成交和绩效口径保持统一。"""
        return build_price_factors(
            self.price_data,
            rebalance_dates,
            adjusted_factors,
        )

    def filter_factor_frame_for_date(
        self,
        factor_date: str | pd.Timestamp,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply the configured point-in-time membership to one factor date."""
        if self.universe_store is None:
            return frame.copy()
        eligible = self.universe_store.snapshot_for(factor_date).codes
        return frame[
            frame["code"].astype(str).str.zfill(6).isin(eligible)
        ].copy()

    def build_drawdown_policy(self):
        """Return an optional stateful policy; v1 intentionally has none."""
        return None

    # ------------------------------------------------------------------
    # 回测核心
    # ------------------------------------------------------------------

    def run(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        mode: Optional[BacktestMode] = None,
    ) -> dict:
        """执行完整回测。

        Returns dict with: equity_curve, trade_log, performance, stats
        """
        cfg = self.config
        start = start or cfg.backtest.start_date
        end = end or cfg.backtest.end_date
        mode = mode or cfg.backtest.mode  # type: ignore[assignment]

        frequency = cfg.portfolio.rebalance_frequency

        # 确保数据已加载
        if not self.price_data:
            self.load_price_data(start, end)

        rebalance_dates = _generate_rebalance_dates(start, end, frequency, self.price_data)
        if not self.sentiment_data and mode != "price_only":
            self.load_sentiment_data(start, end)

        top_n = cfg.portfolio.top_n
        sectors = self.code_to_sector
        sector_enabled = cfg.sector_neutral.enabled
        trailing_stop_pct = cfg.execution.trailing_stop_pct
        hard_stop_pct = cfg.execution.stop_loss_pct
        commission = cfg.execution.commission
        min_unit = cfg.execution.min_lot
        mt_enabled = cfg.market_timing.enabled
        mt_ma_window = cfg.market_timing.ma_window
        time_stop_months = cfg.execution.time_stop_months
        total_capital = cfg.execution.initial_cash

        # --- 构建因子 ---
        adjusted_factors = cfg.factors.for_timeframe(cfg.timeframe)

        logger.info("构建价格因子...")
        price_factors = self.build_price_factor_frame(
            rebalance_dates,
            adjusted_factors,
        )

        sentiment_factors = None
        if mode != "price_only" and self.sentiment_data:
            logger.info("构建情感因子...")
            sentiment_factors = build_sentiment_factors(
                self.sentiment_data, rebalance_dates, self.price_data, adjusted_factors,
                rebalance_freq=frequency,
            )

        # 按模式调整权重 (copy to avoid mutating shared config)
        scoring = copy.deepcopy(cfg.scoring)
        if mode == "price_only":
            scoring.sentiment_weight = 0.0
        elif mode == "sentiment_only":
            scoring.momentum_weight = 0.0
            scoring.pullback_weight = 0.0
            scoring.sentiment_weight = 1.0

        logger.info("合并因子...")
        combined = build_combined_factors(price_factors, sentiment_factors, scoring)
        if self.universe_store is not None:
            filtered = []
            for factor_date, date_frame in combined.groupby("date"):
                filtered.append(self.filter_factor_frame_for_date(
                    factor_date, date_frame
                ))
            combined = (
                pd.concat(filtered, ignore_index=True)
                if filtered else combined.iloc[0:0].copy()
            )
        dates = sorted(combined["date"].unique())

        # --- 准备指数择时数据（索引使用真实交易日） ---
        index_monthly = pd.DataFrame()
        if mt_enabled and len(self.index_data) > 0:
            index_dates = generate_rebalance_dates(
                {"index": self.index_data}, start, end, frequency=frequency
            )
            index_rows = []
            for idx_date in index_dates:
                row = self.index_data.loc[idx_date].copy()
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                row.name = idx_date
                index_rows.append(row)
            if index_rows:
                index_monthly = pd.DataFrame(index_rows)
            index_monthly[f"ma{mt_ma_window}"] = (
                index_monthly["close"].rolling(mt_ma_window).mean()
            )

        # --- 生成信号轨迹 ---
        selection_track: list[dict] = []
        risk_off_dates = 0
        previous_target: list[str] = []
        dates_list = list(dates)
        for date in dates_list:
            month_data = combined[combined["date"] == date].copy()
            if len(month_data) == 0:
                continue
            ranked = month_data.sort_values("score", ascending=False)
            date_sectors = sectors
            if self.universe_store is not None:
                date_sectors = self.universe_store.snapshot_for(date).code_to_sector
            target = select_with_rank_buffer(
                ranked=ranked,
                top_n=top_n,
                sectors=date_sectors,
                max_per_sector=cfg.sector_neutral.max_per_sector,
                previous_picks=previous_target,
                rank_buffer=cfg.portfolio.rank_buffer,
                sector_enabled=sector_enabled,
            )
            risk_on = True
            index_signal = ""
            if mt_enabled and len(index_monthly) > 0:
                if date in index_monthly.index:
                    idx_row = index_monthly.loc[date]
                    if isinstance(idx_row, pd.DataFrame):
                        idx_row = idx_row.iloc[0]
                    idx_close = idx_row["close"]
                    idx_ma = idx_row.get(f"ma{mt_ma_window}", np.nan)
                    if pd.notna(idx_ma) and idx_close < idx_ma:
                        risk_on = False
                        index_signal = (
                            f" (CSI300 {idx_close:.0f} < {mt_ma_window}MA {idx_ma:.0f})"
                        )
            if not risk_on:
                target = []
                risk_off_dates += 1
            target_weights = compute_target_weights(
                target_codes=target,
                price_data=self.price_data,
                as_of=date,
                method=cfg.portfolio.weighting,
                volatility_window=cfg.portfolio.risk_weight_window,
                max_weight=cfg.portfolio.max_position_weight,
            )
            selection_track.append({
                "date": date,
                "picks": target,
                "target": target,
                "target_weights": target_weights,
                "risk_on": risk_on,
                "index_signal": index_signal,
            })
            previous_target = list(target)

        drawdown_policy = self.build_drawdown_policy()
        simulation = simulate_portfolio(
            selection_track=selection_track,
            price_data=self.price_data,
            initial_capital=total_capital,
            commission=commission,
            min_lot=min_unit,
            min_commission=cfg.execution.min_commission,
            transfer_fee_rate=cfg.execution.transfer_fee_rate,
            stamp_duty_rate=cfg.execution.stamp_duty_rate,
            historical_stamp_duty_rate=(
                cfg.execution.historical_stamp_duty_rate
            ),
            stamp_duty_cutover_date=cfg.execution.stamp_duty_cutover_date,
            slippage_bps=cfg.execution.slippage_bps,
            max_volume_participation=(
                cfg.execution.max_volume_participation
            ),
            volume_lookback=cfg.execution.volume_lookback,
            enforce_price_limits=cfg.execution.enforce_price_limits,
            end_date=end,
            hard_stop_pct=hard_stop_pct if hard_stop_pct > 0 else None,
            trailing_stop_pct=(
                trailing_stop_pct if trailing_stop_pct > 0 else None
            ),
            time_stop_days=(
                int(time_stop_months * 30.44) if time_stop_months > 0 else None
            ),
            drawdown_policy=drawdown_policy,
        )
        equity_curve = simulation["equity_curve"]
        trade_log = simulation["trade_log"]
        stop_loss_events = simulation["stop_loss_events"]
        time_stop_events = simulation["time_stop_events"]
        execution_stats = simulation["execution_stats"]
        risk_events = simulation.get("risk_events", [])

        # --- 构建权益 DataFrame ---
        eq_df = pd.DataFrame(equity_curve)
        if len(eq_df) > 0:
            eq_df.set_index("date", inplace=True)

        self.equity_curve = eq_df
        self.trade_log = trade_log

        # --- 计算绩效 ---
        # 统一执行器输出日频净值，所有入口都用日频风险口径。
        periods_per_year = 252
        perf = {"no_data": True}
        stat = {
            "stop_loss_events": stop_loss_events,
            "time_stop_events": time_stop_events,
            "risk_off_dates": risk_off_dates,
            "total_dates": len(dates_list),
            "execution": execution_stats,
            "risk_events": len(risk_events),
        }

        if len(eq_df) >= 2:
            equity_series = eq_df["equity"]
            equity_aligned = equity_series
            perf = compute_performance(equity_aligned, periods_per_year=periods_per_year)

            # --- 补充指标（复现 legacy 的精确公式） ---
            final_eq = equity_aligned.iloc[-1]
            years = max((equity_aligned.index[-1] - equity_aligned.index[0]).days / 365.25, 0.02)
            monthly_ret = equity_aligned.pct_change().dropna()

            # Sortino (legacy 公式)
            rf_annual = 0.02
            rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
            excess = monthly_ret - rf_period
            downside = excess[excess < 0]
            sortino = (
                excess.mean() / downside.std() * np.sqrt(periods_per_year)
                if len(downside) > 1 and downside.std() > 0 else 0
            )

            # Win rate (legacy 公式)
            wins = (monthly_ret > 0).sum()
            total = len(monthly_ret)
            win_rate = wins / total * 100 if total > 0 else 0

            # P/L ratio (legacy 公式)
            avg_win = monthly_ret[monthly_ret > 0].mean() if wins > 0 else 0
            avg_loss = abs(monthly_ret[monthly_ret < 0].mean()) if (monthly_ret < 0).any() else 0
            pl_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

            # Sortino
            # 换手率
            prev_set = None
            turnovers = []
            for t in selection_track:
                curr_set = set(t["picks"])
                if prev_set is not None and len(prev_set) > 0:
                    overlap = len(curr_set & prev_set)
                    turnover = (len(curr_set) + len(prev_set) - 2 * overlap) / len(prev_set)
                    turnovers.append(turnover)
                prev_set = curr_set
            avg_turnover = np.mean(turnovers) * 100 if turnovers else 0

            perf["sortino"] = round(sortino, 3)
            perf["pl_ratio"] = round(pl_ratio, 3)
            perf["win_rate"] = round(win_rate, 1)
            perf["avg_turnover_pct"] = round(avg_turnover, 1)
            perf["best_return"] = round(monthly_ret.max() * 100, 2)
            perf["worst_return"] = round(monthly_ret.min() * 100, 2)

            avg_cash_ratio = (eq_df["cash"] / eq_df["equity"]).mean() * 100
            perf["avg_deployment_pct"] = round(100 - avg_cash_ratio, 1)
            perf["no_data"] = False

            stat["final_equity"] = round(final_eq, 2)
            stat["years"] = round(years, 2)

        self.run_metadata = {
            "start_date": start,
            "end_date": end,
            "mode": mode,
            "universe_source": cfg.universe.source,
            "universe_snapshot_count": (
                len(self.universe_store.available_dates)
                if self.universe_store is not None else 0
            ),
            "loaded_stock_count": len(self.price_data),
            "sentiment_weight": scoring.sentiment_weight,
            "weighting": cfg.portfolio.weighting,
            "risk_weight_window": cfg.portfolio.risk_weight_window,
            "max_position_weight": cfg.portfolio.max_position_weight,
            "rank_buffer": cfg.portfolio.rank_buffer,
            "sector_neutral": cfg.sector_neutral.enabled,
            "max_per_sector": cfg.sector_neutral.max_per_sector,
            "hard_stop_pct": hard_stop_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "time_stop_months": time_stop_months,
            "market_timing": mt_enabled,
            "commission": commission,
            "min_commission": cfg.execution.min_commission,
            "transfer_fee_rate": cfg.execution.transfer_fee_rate,
            "stamp_duty_rate": cfg.execution.stamp_duty_rate,
            "historical_stamp_duty_rate": (
                cfg.execution.historical_stamp_duty_rate
            ),
            "stamp_duty_cutover_date": (
                cfg.execution.stamp_duty_cutover_date
            ),
            "slippage_bps": cfg.execution.slippage_bps,
            "max_volume_participation": (
                cfg.execution.max_volume_participation
            ),
            "volume_lookback": cfg.execution.volume_lookback,
            "enforce_price_limits": cfg.execution.enforce_price_limits,
        }
        self.results = {
            "equity_curve": eq_df,
            "trade_log": trade_log,
            "performance": perf,
            "stats": stat,
            "metadata": self.run_metadata,
            "risk_events": risk_events,
        }

        logger.info(
            f"回测完成: {len(dates_list)} 调仓期, "
            f"止损{stop_loss_events}次, 时间止损{time_stop_events}次, "
            f"避险{risk_off_dates}期"
        )
        return self.results

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def performance_summary(self) -> dict:
        """返回绩效摘要。需在 run() 之后调用。"""
        perf = self.results.get("performance", {})
        stats = self.results.get("stats", {})
        return {**perf, **stats}

    def print_report(self) -> None:
        """打印格式化回测报告。"""
        cfg = self.config
        perf = self.results.get("performance", {})
        stats = self.results.get("stats", {})

        if perf.get("no_data"):
            print("无回测数据。请先执行 run()。")
            return

        # 兼容 compute_performance 的字段名
        total_ret = perf.get("total_return", 0)
        cagr = perf.get("cagr", 0)
        sharpe = perf.get("sharpe", 0)
        max_dd = perf.get("max_drawdown", 0)
        sortino = perf.get("sortino", 0)
        calmar = perf.get("calmar", 0)
        win_rate = perf.get("win_rate", 0)
        pl_ratio = perf.get("pl_ratio", 0)
        avg_turnover = perf.get("avg_turnover_pct", 0)
        best_month = perf.get("best_return", 0)
        worst_month = perf.get("worst_return", 0)
        n_periods = perf.get("n_periods", 0)
        ann_vol = perf.get("ann_volatility", 0)
        avg_deploy = perf.get("avg_deployment_pct", 0)

        final_eq = stats.get("final_equity", 0)
        years = stats.get("years", 0)
        stop_events = stats.get("stop_loss_events", 0)
        time_stop_ev = stats.get("time_stop_events", 0)
        risk_off = stats.get("risk_off_dates", 0)
        total_dates = stats.get("total_dates", 0)
        execution_stats = stats.get("execution", {})

        mt_enabled = cfg.market_timing.enabled
        mt_ma = cfg.market_timing.ma_window
        time_stop_m = cfg.execution.time_stop_months
        total_capital = cfg.execution.initial_cash

        W = 60
        print()
        print("=" * W)
        print(f"  统一回测报告 — {cfg.strategy_name} v{cfg.version}")
        print(f"  调仓频率: {cfg.rebalance_label}  |  初始资金: {total_capital:,.0f}")
        print(f"  持仓: Top {cfg.portfolio.top_n}  |  费率: {cfg.execution.commission*100:.2f}%")
        trailing_label = (
            f"-{cfg.execution.trailing_stop_pct*100:.0f}%"
            if cfg.execution.trailing_stop_pct > 0 else "关闭"
        )
        hard_label = (
            f"-{cfg.execution.stop_loss_pct*100:.0f}%"
            if cfg.execution.stop_loss_pct > 0 else "关闭"
        )
        time_label = f"{time_stop_m}月" if time_stop_m > 0 else "关闭"
        print(f"  大盘择时: {'沪深300 ' + str(mt_ma) + 'MA' if mt_enabled else '关闭'}"
              f"  |  拖尾: {trailing_label}"
              f"  |  硬止损: {hard_label}"
              f"  |  时间止损: {time_label}")
        print("=" * W)

        print(f"\n  {'─'*50}")
        print(f"  【收益指标】")
        print(f"  {'─'*50}")
        print(f"  {'最终资产:':<20s} {final_eq:>14,.2f} 元")
        print(f"  {'总收益率:':<20s} {total_ret:>14.2f}%")
        print(f"  {'年化收益 (CAGR):':<20s} {cagr:>14.2f}%")
        print(f"  {'回测年数:':<20s} {years:>14.2f}")
        print(f"  {'最佳单期:':<20s} {best_month:>14.2f}%")
        print(f"  {'最差单期:':<20s} {worst_month:>14.2f}%")

        print(f"\n  {'─'*50}")
        print(f"  【风险指标】")
        print(f"  {'─'*50}")
        print(f"  {'指标':<18s} {'数值':>12s}  {'评价':<20s}")

        def _label(v, thresholds):
            for t, lbl in thresholds:
                if v > t:
                    return lbl
            return thresholds[-1][1]

        sharpe_l = _label(sharpe, [(2, "优秀"), (1, "良好"), (0.5, "一般")])
        sortino_l = _label(sortino, [(2, "优秀"), (1, "良好"), (0.5, "一般")])
        calmar_l = _label(calmar, [(1, "优秀"), (0.5, "良好"), (0.2, "一般")])
        wr_l = _label(win_rate, [(60, "优秀"), (50, "良好")])
        pl_l = _label(pl_ratio, [(2, "优秀"), (1.2, "良好")])

        print(f"  {'夏普比率':<18s} {sharpe:>12.3f}  {sharpe_l:<20s}")
        print(f"  {'索提诺':<18s} {sortino:>12.3f}  {sortino_l:<20s}")
        print(f"  {'卡玛比率':<18s} {calmar:>12.3f}  {calmar_l:<20s}")
        print(f"  {'最大回撤':<18s} {max_dd:>11.2f}%")
        print(f"  {'年化波动率':<18s} {ann_vol:>11.2f}%")
        print(f"  {'胜率':<18s} {win_rate:>11.1f}%  ({n_periods}期)")
        print(f"  {'盈亏比':<18s} {pl_ratio:>12.3f}")
        print(f"  {'平均换手率':<18s} {avg_turnover:>11.1f}%")
        print(f"  {'资金利用率':<18s} {avg_deploy:>11.1f}%")
        print(f"  {'止损触发次数':<18s} {stop_events:>12}")
        print(f"  {'交易费用':<18s} {execution_stats.get('total_fees', 0):>12,.2f} 元")
        print(f"  {'滑点成本':<18s} {execution_stats.get('slippage_cost', 0):>12,.2f} 元")
        print(f"  {'成交受限记录':<18s} {execution_stats.get('constrained_orders', 0):>12}")
        if mt_enabled:
            pct = risk_off / total_dates * 100 if total_dates > 0 else 0
            print(f"  {'避险期数':<18s} {risk_off:>12} ({pct:.0f}%)")
        if time_stop_m:
            print(f"  {'时间止损次数':<18s} {time_stop_ev:>12}")

        # 综合评分
        score = sum([sharpe > 1, sortino > 1, calmar > 0.3, win_rate > 50, pl_ratio > 1.2])
        verdicts = {5: "五星", 4: "四星", 3: "三星", 2: "二星", 1: "一星", 0: "需改进"}
        print(f"  {'─'*50}")
        print(f"  综合评分: {score}/5 — {verdicts.get(score, '?')}")
        print("=" * W)

        # 前3期调仓详情
        stocks = self.code_to_name
        sectors = self.code_to_sector
        print(f"\n{'='*W}")
        print(f"  前 3 期调仓详情")
        print(f"{'='*W}")

        for t in self.trade_log[:3]:
            d = t["date"]
            print(f"\n  ┌─ {d.date()} ─────────────────────────────")
            print(f"  │ 推荐: {t['target']}")

            if t["sells"]:
                print(f"  │ 卖出:")
                for s in t["sells"]:
                    name = stocks.get(s["code"], "?")
                    note = f"  ← {s['note']}" if s.get("note") else ""
                    print(f"  │   [SELL] {s['code']} {name:<6s}  "
                          f"{s['shares']}股 @ {s['price']:.2f}  +{s['proceeds']:,.0f}元{note}")

            print(f"  │ 调仓:")
            for b in t["buys"]:
                name = stocks.get(b["code"], "?")
                sec = sectors.get(b["code"], "—")
                action = b.get("action", "BUY")
                if action == "SKIP":
                    print(f"  │   [SKIP] {b['code']} {name:<6s} [{sec}]  {b.get('reason', '')}")
                elif action == "TRIM":
                    print(f"  │   [TRIM] {b['code']} {name:<6s} [{sec}]  "
                          f"卖{b['shares']}股 → {b['total_shares']}股  +{b['cost']:,.0f}元")
                else:
                    print(f"  │   [BUY]  {b['code']} {name:<6s} [{sec}]  "
                          f"@{b['price']:.2f}  买{b['shares']:>5}股 "
                          f"→ {b.get('total_shares', b['shares'])}股 "
                          f"({b.get('lots', b['shares'] // cfg.execution.min_lot)}手)  "
                          f"成本{b['cost']:>12,.0f}元")

            print(f"  │ 现金: {t['cash_after']:,.0f}  |  市值: {t['stock_value']:,.0f}  |  权益: {t['equity']:,.0f}")
            print(f"  └{'─'*46}")

        # 最后一期持仓
        if self.trade_log:
            last_t = self.trade_log[-1]
            print(f"\n  >> 最后一期持仓 ({last_t['date'].date()}):")
            print(f"  {'─'*50}")
            eq_df = self.equity_curve
            if len(eq_df) > 0 and last_t["date"] in eq_df.index:
                # 从交易日志中获取最后持仓
                hold_codes = set()
                for t in reversed(self.trade_log):
                    for b in t["buys"]:
                        if b.get("action") in ("BUY", "TRIM"):
                            hold_codes.add(b["code"])
                    if hold_codes:
                        break
                # 从 last trade log 获取 target
                for code in last_t["target"]:
                    name = stocks.get(code, "?")
                    sec = sectors.get(code, "—")
                    # 从 price data 取最新价格
                    last_price = "N/A"
                    if code in self.price_data:
                        last_price = f"{self.price_data[code]['close'].iloc[-1]:.2f}"
                    print(f"    {code} {name:<6s} [{sec}]  @ {last_price}")
            print(f"  现金: {last_t['cash_after']:,.0f}  |  总权益: {last_t['equity']:,.0f}")

    def plot(self, show: bool = False, save_path: Optional[str | Path] = None) -> None:
        """绘制权益曲线 + 回撤图。"""
        if len(self.equity_curve) < 2:
            print("无足够数据绘图。")
            return

        import matplotlib
        matplotlib.use("Agg")  # 非交互式后端
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        # 尝试设置中文字体
        try:
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        eq = self.equity_curve["equity"]
        cummax = eq.cummax()
        drawdowns = (eq - cummax) / cummax * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # 权益曲线
        ax1.plot(eq.index, eq.values, color="#1f77b4", linewidth=1.2, label="Equity")
        ax1.plot(cummax.index, cummax.values, color="gray", linestyle="--", alpha=0.5,
                 label="High-water mark")
        ax1.set_ylabel("Portfolio Value (CNY)")
        ax1.set_title(f"{self.config.strategy_name} — Equity Curve")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x/1e3:.0f}K"))

        # 回撤
        ax2.fill_between(drawdowns.index, drawdowns.values, 0,
                          color="#d62728", alpha=0.3, label="Drawdown")
        ax2.plot(drawdowns.index, drawdowns.values, color="#d62728", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        max_dd = drawdowns.min()
        ax2.axhline(y=max_dd, color="#d62728", linestyle="--", alpha=0.5,
                     label=f"Max DD: {max_dd:.1f}%")
        ax2.legend(loc="lower left")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def export_results(self, output_dir: str | Path, prefix: str = "backtest") -> None:
        """导出结果到文件。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        eq = self.equity_curve
        if len(eq) > 0:
            eq.to_csv(out / f"{prefix}_equity.csv", encoding="utf-8-sig")

        if self.trade_log:
            trade_df = pd.DataFrame([
                {"date": t["date"], "target": ",".join(t["target"]),
                 "n_sells": len(t["sells"]), "n_buys": len(t["buys"]),
                 "risk_on": t.get("risk_on", True), "equity": t["equity"]}
                for t in self.trade_log
            ])
            trade_df.to_csv(out / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")

            order_records = []
            constraint_records = []
            for log in self.trade_log:
                for side_key, side in (("buys", "buy"), ("sells", "sell")):
                    for order in log.get(side_key, []):
                        fees = order.get("fees", {})
                        order_records.append({
                            "date": log.get("date"),
                            "signal_date": log.get("signal_date"),
                            "side": side,
                            "code": order.get("code"),
                            "shares": order.get("shares", 0),
                            "reference_price": order.get("reference_price"),
                            "execution_price": order.get("price"),
                            "gross_value": order.get("gross_value", 0),
                            "commission": fees.get("commission", 0),
                            "transfer_fee": fees.get("transfer_fee", 0),
                            "stamp_duty": fees.get("stamp_duty", 0),
                            "total_fees": fees.get("total", 0),
                            "slippage_cost": order.get("slippage_cost", 0),
                            "note": order.get("note", ""),
                        })
                for item in log.get("constraints", []):
                    constraint_records.append({
                        "date": log.get("date"),
                        "signal_date": log.get("signal_date"),
                        **item,
                    })
            if order_records:
                pd.DataFrame(order_records).to_csv(
                    out / f"{prefix}_orders.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            if constraint_records:
                pd.DataFrame(constraint_records).to_csv(
                    out / f"{prefix}_constraints.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        perf = self.results.get("performance", {})
        stats = self.results.get("stats", {})
        metadata = self.results.get("metadata", {})
        with open(out / f"{prefix}_performance.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "performance": perf, "stats": stats}, f, ensure_ascii=False, indent=2,
                      default=str)

        # 文本报告
        with open(out / f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            self.print_report()
            report = sys.stdout.getvalue()
            sys.stdout = old_stdout
            f.write(report)

        print(f"结果已导出至: {out}/")


# ======================================================================
# CLI 入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="统一回测引擎")
    parser.add_argument("--start", type=str, default="2020-01-01", help="回测起始 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-05-12", help="回测结束 YYYY-MM-DD")
    parser.add_argument("--mode", choices=["price_only", "sentiment_only", "hybrid"],
                        default=None, help="回测模式（默认读取配置，基准为 price_only）")
    parser.add_argument("--capital", type=float, default=None, help="初始资金")
    parser.add_argument("--top-n", type=int, default=None, help="持仓数")
    parser.add_argument(
        "--weighting",
        choices=["equal_weight", "inverse_volatility"],
        default=None,
        help="仓位权重：等权或逆波动率",
    )
    parser.add_argument(
        "--risk-window", type=int, default=None,
        help="逆波动率计算窗口（日）",
    )
    parser.add_argument(
        "--max-position-weight", type=float, default=None,
        help="单股目标权重上限（如 0.35）",
    )
    parser.add_argument(
        "--rank-buffer", type=int, default=None,
        help="旧持仓排名缓冲名额（0=关闭）",
    )
    parser.add_argument(
        "--max-per-sector", type=int, default=None,
        help="每行业最多持仓数",
    )
    parser.add_argument(
        "--no-sector-neutral", action="store_true",
        help="关闭行业持仓数约束",
    )
    parser.add_argument("--export", type=str, default=None, help="导出目录")
    parser.add_argument("--plot", type=str, default=None, help="保存图表路径")
    parser.add_argument("--no-report", action="store_true", help="不打印报告")
    parser.add_argument(
        "--frictionless",
        action="store_true",
        help="诊断对照：关闭费用、滑点、容量和涨跌停限制",
    )
    args = parser.parse_args()

    config = StrategyConfig.from_env()
    if args.capital:
        config.execution.initial_cash = args.capital
    if args.top_n:
        config.portfolio.top_n = args.top_n
    if args.weighting:
        config.portfolio.weighting = args.weighting
    if args.risk_window is not None:
        config.portfolio.risk_weight_window = max(args.risk_window, 2)
    if args.max_position_weight is not None:
        config.portfolio.max_position_weight = max(
            args.max_position_weight, 0.0
        )
    if args.rank_buffer is not None:
        config.portfolio.rank_buffer = max(args.rank_buffer, 0)
    if args.max_per_sector is not None:
        config.sector_neutral.max_per_sector = max(args.max_per_sector, 1)
    if args.no_sector_neutral:
        config.sector_neutral.enabled = False
    if args.frictionless:
        config.execution.commission = 0.0
        config.execution.min_commission = 0.0
        config.execution.transfer_fee_rate = 0.0
        config.execution.stamp_duty_rate = 0.0
        config.execution.historical_stamp_duty_rate = 0.0
        config.execution.slippage_bps = 0.0
        config.execution.max_volume_participation = 0.0
        config.execution.enforce_price_limits = False

    selected_mode = args.mode or config.backtest.mode
    mode_str = {
        "price_only": "纯价格因子",
        "sentiment_only": "纯情感因子",
        "hybrid": "混合因子 (价格+情感)",
    }
    print(f"  量化周度系统 — 统一回测引擎")
    print(f"  模式: {mode_str[selected_mode]}")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"  资金: {config.execution.initial_cash:,.0f}  |  持仓: Top {config.portfolio.top_n}")
    print(
        f"  权重: {config.portfolio.weighting}  |  "
        f"排名缓冲: {config.portfolio.rank_buffer}  |  "
        f"行业上限: "
        f"{config.sector_neutral.max_per_sector if config.sector_neutral.enabled else '关闭'}"
    )
    print()

    engine = BacktestEngine(config)
    engine.load_price_data(args.start, args.end)
    if selected_mode != "price_only":
        engine.load_sentiment_data(args.start, args.end)
    engine.run(args.start, args.end, mode=selected_mode)  # type: ignore[arg-type]

    if not args.no_report:
        engine.print_report()

    if args.plot:
        engine.plot(save_path=args.plot)

    if args.export:
        engine.export_results(args.export)


if __name__ == "__main__":
    main()
