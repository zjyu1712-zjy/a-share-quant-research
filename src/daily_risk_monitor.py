"""
日度风险自测与周中熔断模块
============================
设计目标: 在非调仓日 (周一到周四) 监控持仓股表现，
当单只股票周中跌幅超过阈值时触发预警或强制平仓信号。

工程解耦设计:
  - 与周线主策略通过 JSON 状态文件 (portfolio_state.json) 交互
  - 主策略在每周五调仓后写入 state，本模块每日读取
  - 完全独立运行，不修改主策略的任何代码
  - 也可读取同一个 SQLite 数据库获取新闻情感背景

集成方式:
  1. 周线策略调仓后调用 save_portfolio_state() 写入持仓
  2. 每日 cron 运行 DailyRiskMonitor.check() 做风险扫描
  3. 触发熔断时生成 midweek_alert.json，供次日开盘执行

用法:
  python -m src.daily_risk_monitor          # 单次扫描
  python -m src.daily_risk_monitor --check  # 同

输出:
  - 控制台日志 (带颜色标记)
  - data/daily_risk_report_YYYYMMDD.json (每日报告)
  - data/midweek_alert.json (仅触发熔断时生成)
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

STATE_FILE = DATA_DIR / "portfolio_state.json"
ALERT_FILE = DATA_DIR / "midweek_alert.json"
REPORT_DIR = DATA_DIR / "risk_reports"

# ── 默认阈值 (可通过 portfolio_state.json 覆盖) ──
DEFAULT_THRESHOLDS = {
    "weekly_drawdown_pct": -8.0,       # 周中累计跌幅阈值 (负数)
    "daily_drop_pct": -5.0,            # 单日跌幅阈值 (负数)
    "daily_drop_with_ma_break": True,  # 单日大跌 + 跌破 MA 才触发
    "ma_windows": [20, 60],            # 参考均线
    "relative_index_drop_pct": -3.0,   # 相对沪深300超额跌幅
    "circuit_breaker_action": "cash",  # "cash"(转现金) / "replace"(换备选)
    "index_code": "000300",            # 基准指数
}


# ======================================================================
# 数据类
# ======================================================================

@dataclass
class PositionRecord:
    """单只持仓记录"""
    code: str
    name: str
    sector: str
    entry_date: str               # 建仓日期 (本周一)
    entry_price: float            # 入场价
    shares: int                   # 持仓股数
    cost: float                   # 持仓成本 (含手续费)
    current_price: float = 0.0    # 最新价 (每日更新)
    highest_price: float = 0.0    # 本周最高价
    weekly_return_pct: float = 0.0
    daily_return_pct: float = 0.0
    vs_index_pct: float = 0.0     # 相对沪深300的超额
    below_ma20: bool = False
    below_ma60: bool = False
    alert_level: str = "normal"   # normal / warning / danger


@dataclass
class PortfolioState:
    """组合状态"""
    rebalance_date: str = ""
    positions: list[PositionRecord] = field(default_factory=list)
    cash: float = 0.0
    total_equity: float = 0.0
    top_n: int = 5
    backup_pool: list[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    updated: str = ""


# ======================================================================
# 核心监控类
# ======================================================================

class DailyRiskMonitor:
    """轻量级每日风险扫描器。

    特性:
      - 零依赖主策略: 仅读取 JSON state 文件
      - 自动获取行情 (AKShare fallback → 本地缓存)
      - 三级警报: normal → warning → danger
      - danger 触发时自动生成 midweek_alert.json
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        thresholds: Optional[dict] = None,
    ):
        self.state_path = state_path or STATE_FILE
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        self.state: Optional[PortfolioState] = None
        self.price_data: dict[str, pd.DataFrame] = {}
        self.index_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Step 1: 加载持仓状态
    # ------------------------------------------------------------------

    def load_state(self) -> bool:
        """从 portfolio_state.json 加载当前持仓。

        Returns:
            True if state loaded successfully, False if no state file
            (表示本周尚未调仓或未运行主策略)
        """
        if not self.state_path.exists():
            logger.warning(
                f"持仓状态文件不存在: {self.state_path}\n"
                f"  可能原因: 本周尚未运行周线选股, 请先执行周末调仓。"
            )
            return False

        with open(self.state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 检查状态是否过期 (> 7 天未更新)
        updated = raw.get("updated", "")
        if updated:
            try:
                dt = datetime.fromisoformat(updated)
                if (datetime.now() - dt).days > 7:
                    logger.warning(
                        f"持仓状态已过期 ({updated}, "
                        f"距今 {(datetime.now() - dt).days} 天), 请重新调仓"
                    )
            except ValueError:
                pass

        # 解析持仓
        positions = []
        for p in raw.get("positions", []):
            positions.append(PositionRecord(
                code=p.get("code", ""),
                name=p.get("name", ""),
                sector=p.get("sector", ""),
                entry_date=p.get("entry_date", ""),
                entry_price=float(p.get("entry_price", 0)),
                shares=int(p.get("shares", 0)),
                cost=float(p.get("cost", 0)),
            ))

        self.state = PortfolioState(
            rebalance_date=raw.get("rebalance_date", ""),
            positions=positions,
            cash=float(raw.get("cash", 0)),
            total_equity=float(raw.get("total_equity", 0)),
            top_n=int(raw.get("top_n", 5)),
            backup_pool=raw.get("backup_pool", []),
            thresholds=raw.get("thresholds", self.thresholds),
            updated=updated,
        )

        # 合并阈值 (state 中的优先)
        if raw.get("thresholds"):
            self.thresholds.update(raw["thresholds"])

        logger.info(
            f"持仓加载成功: {len(positions)} 只股票, "
            f"调仓日 {self.state.rebalance_date}, "
            f"总权益 {self.state.total_equity:,.0f}"
        )
        return True

    # ------------------------------------------------------------------
    # Step 2: 获取行情数据
    # ------------------------------------------------------------------

    def fetch_prices(self) -> None:
        """获取持仓股最新行情 + 基准指数。

        优先使用本地缓存 (data/price_cache/), 缓存缺失时调用 AKShare。
        """
        if not self.state or not self.state.positions:
            return

        codes = [p.code for p in self.state.positions]
        end_date = datetime.now().strftime("%Y-%m-%d")
        # 取最近 120 天数据用于计算均线
        start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

        try:
            from src.price_feed import load_pool_data, load_index_data

            self.price_data = load_pool_data(codes, start_date, end_date)

            # 基准指数
            idx_code = self.thresholds.get("index_code", "000300")
            try:
                self.index_data = load_index_data(idx_code, start_date, end_date)
                logger.info(f"基准指数 {idx_code} 加载成功: {len(self.index_data)} 条")
            except Exception as e:
                logger.warning(f"基准指数加载失败: {e}, 跳过相对强弱计算")

        except ImportError as e:
            logger.error(f"无法导入 price_feed: {e}")
            raise
        except Exception as e:
            logger.error(f"行情获取失败: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 3: 风险扫描
    # ------------------------------------------------------------------

    def scan(self) -> list[PositionRecord]:
        """逐只股票扫描风险指标。

        Returns:
            更新后的持仓列表 (含当前行情和风险标记)
        """
        if not self.state:
            raise RuntimeError("请先调用 load_state()")

        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        week_start = self._get_week_start(today)

        alerts = []

        for pos in self.state.positions:
            code = pos.code
            df = self.price_data.get(code)

            if df is None or len(df) == 0:
                logger.warning(f"  [{code} {pos.name}] 无行情数据, 跳过")
                pos.alert_level = "warning"
                pos.current_price = pos.entry_price
                continue

            # 确保索引是 datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # ── 最新价 ──
            latest = df.iloc[-1]
            pos.current_price = float(latest["close"])

            # ── 本周最高价 ──
            week_mask = df.index >= pd.Timestamp(week_start)
            week_data = df[week_mask]
            if len(week_data) > 0:
                pos.highest_price = float(week_data["high"].max())
            else:
                pos.highest_price = pos.entry_price

            # ── 周中累计跌幅 (从入场价计算) ──
            if pos.entry_price > 0:
                pos.weekly_return_pct = round(
                    (pos.current_price / pos.entry_price - 1) * 100, 2
                )

            # ── 今日跌幅 ──
            if len(df) >= 2:
                prev_close = float(df.iloc[-2]["close"])
                if prev_close > 0:
                    pos.daily_return_pct = round(
                        (pos.current_price / prev_close - 1) * 100, 2
                    )
            else:
                pos.daily_return_pct = 0.0

            # ── 相对指数表现 (同期) ──
            if self.index_data is not None and len(self.index_data) > 0:
                pos.vs_index_pct = self._calc_relative_return(
                    pos.entry_price, pos.current_price,
                    pos.entry_date, today_str,
                )

            # ── 均线位置 ──
            for ma_w in self.thresholds.get("ma_windows", [20, 60]):
                if len(df) >= ma_w:
                    ma_val = float(df["close"].rolling(ma_w).mean().iloc[-1])
                    if ma_val > 0:
                        if ma_w == 20:
                            pos.below_ma20 = pos.current_price < ma_val
                        elif ma_w == 60:
                            pos.below_ma60 = pos.current_price < ma_val

            # ── 判定警报等级 ──
            pos.alert_level = self._assess_alert(pos)

            if pos.alert_level != "normal":
                alerts.append(pos)

        self.state.positions = [
            p for p in self.state.positions
        ]  # 保持引用, pos 已就地更新

        if alerts:
            logger.warning(
                f"风险扫描完成: {len(alerts)}/{len(self.state.positions)} 只触发警报"
            )
        else:
            logger.info(
                f"风险扫描完成: 全部 {len(self.state.positions)} 只正常"
            )

        return alerts

    # ------------------------------------------------------------------
    # Step 4: 生成报告 & 熔断指令
    # ------------------------------------------------------------------

    def generate_report(
        self,
        alerts: list[PositionRecord],
    ) -> dict:
        """生成日度风险自测报告。

        Returns:
            报告 dict (同时写入 JSON 文件)
        """
        now = datetime.now()
        report_date = now.strftime("%Y%m%d")

        # 汇总统计
        total_weekly_pnl = 0.0
        for pos in (self.state.positions if self.state else []):
            total_weekly_pnl += pos.weekly_return_pct * (pos.cost / self.state.total_equity)

        danger_codes = [p.code for p in alerts if p.alert_level == "danger"]
        warning_codes = [p.code for p in alerts if p.alert_level == "warning"]

        report = {
            "report_date": now.strftime("%Y-%m-%d"),
            "report_time": now.isoformat(),
            "rebalance_date": self.state.rebalance_date if self.state else "",
            "days_since_rebalance": (
                (now - datetime.fromisoformat(self.state.rebalance_date)).days
                if self.state and self.state.rebalance_date else 0
            ),
            "total_positions": len(self.state.positions) if self.state else 0,
            "total_equity": self.state.total_equity if self.state else 0,
            "weekly_pnl_weighted_pct": round(total_weekly_pnl, 2),
            "thresholds_used": self.thresholds,
            "alerts_summary": {
                "danger_count": len(danger_codes),
                "warning_count": len(warning_codes),
                "danger_codes": danger_codes,
                "warning_codes": warning_codes,
            },
            "positions": [
                {
                    "code": p.code,
                    "name": p.name,
                    "sector": p.sector,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "weekly_return_pct": p.weekly_return_pct,
                    "daily_return_pct": p.daily_return_pct,
                    "vs_index_pct": p.vs_index_pct,
                    "below_ma20": p.below_ma20,
                    "below_ma60": p.below_ma60,
                    "alert_level": p.alert_level,
                }
                for p in (self.state.positions if self.state else [])
            ],
        }

        # 写入报告
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"daily_risk_report_{report_date}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"风险报告已保存: {report_path}")

        return report

    def generate_alert(
        self,
        alerts: list[PositionRecord],
    ) -> Optional[dict]:
        """如果触发 danger 级别警报, 生成周中强制调仓指令。

        Returns:
            midweek_alert dict, 或 None (无 danger 则不生成)
        """
        danger_alerts = [p for p in alerts if p.alert_level == "danger"]
        if not danger_alerts:
            logger.info("无 danger 级别警报, 不生成熔断指令")
            # 删除旧警报文件 (如果存在)
            if ALERT_FILE.exists():
                ALERT_FILE.unlink()
                logger.info("已清除旧熔断指令")
            return None

        action = self.thresholds.get("circuit_breaker_action", "cash")
        now = datetime.now().isoformat()

        alert_doc = {
            "generated_at": now,
            "action_required": action,
            "reason": f"{len(danger_alerts)} 只股票触发 danger 级别警报",
            "sell_list": [],
            "buy_list": [],
            "instructions": [],
        }

        for pos in danger_alerts:
            sell_instruction = (
                f"卖出 [{pos.code} {pos.name}] {pos.shares}股 "
                f"@市价 {pos.current_price:.2f} | "
                f"周跌幅: {pos.weekly_return_pct:+.1f}% | "
                f"日跌幅: {pos.daily_return_pct:+.1f}% | "
                f"今日跌破MA20" if pos.below_ma20 else ""
            )
            alert_doc["sell_list"].append({
                "code": pos.code,
                "name": pos.name,
                "shares": pos.shares,
                "current_price": pos.current_price,
                "weekly_return_pct": pos.weekly_return_pct,
                "daily_return_pct": pos.daily_return_pct,
                "reason": sell_instruction,
            })

        if action == "replace" and self.state and self.state.backup_pool:
            # 从备选池中选择替换标的
            backup = self.state.backup_pool[:len(danger_alerts)]
            for i, code in enumerate(backup):
                alert_doc["buy_list"].append({
                    "code": code,
                    "reason": f"替换被熔断的 {danger_alerts[i].code}",
                })
            alert_doc["instructions"].append(
                f"先卖出 {len(danger_alerts)} 只 danger 标的, "
                f"再买入备选池: {backup}"
            )
        elif action == "cash":
            alert_doc["instructions"].append(
                f"卖出 {len(danger_alerts)} 只 danger 标的, 转为现金持有"
            )

        # 写入文件
        ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERT_FILE, "w", encoding="utf-8") as f:
            json.dump(alert_doc, f, ensure_ascii=False, indent=2)

        logger.warning(
            f"!!! 周中熔断触发 !!!\n"
            f"  涉及标的: {[p.code for p in danger_alerts]}\n"
            f"  执行动作: {action}\n"
            f"  指令文件: {ALERT_FILE}"
        )
        return alert_doc

    # ------------------------------------------------------------------
    # 一站式入口
    # ------------------------------------------------------------------

    def check(self) -> dict:
        """一站式每日风险扫描: 加载→获取行情→扫描→报告→熔断。

        Returns:
            {"status": "ok"|"no_state"|"error",
             "report": {...},
             "alert": {...} | None}
        """
        # 1. 加载持仓
        if not self.load_state():
            return {"status": "no_state", "report": None, "alert": None}

        # 2. 获取行情
        try:
            self.fetch_prices()
        except Exception as e:
            logger.error(f"行情获取失败: {e}")
            return {"status": "error", "report": None, "alert": None,
                    "error": str(e)}

        # 3. 扫描
        alerts = self.scan()

        # 4. 报告
        report = self.generate_report(alerts)

        # 5. 熔断 (如有)
        alert = self.generate_alert(alerts)

        # ── 终端彩色输出 ──
        self._print_summary(report, alert)

        return {"status": "ok", "report": report, "alert": alert}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _assess_alert(self, pos: PositionRecord) -> str:
        """根据阈值判定 alert_level。

        danger:  单周累计跌 > 8% 或 (单日跌 > 5% 且跌破 MA20)
        warning: 单周累计跌 > 5% 或 日跌 > 3%
        normal:  其余
        """
        w_dd = self.thresholds.get("weekly_drawdown_pct", -8.0)
        d_drop = self.thresholds.get("daily_drop_pct", -5.0)
        ma_break = self.thresholds.get("daily_drop_with_ma_break", True)
        rel_drop = self.thresholds.get("relative_index_drop_pct", -3.0)

        weekly_hit = pos.weekly_return_pct <= w_dd
        daily_hit = pos.daily_return_pct <= d_drop
        daily_ma_hit = daily_hit and pos.below_ma20
        relative_hit = pos.vs_index_pct <= rel_drop

        # danger: 单周累计跌幅超阈值
        if weekly_hit:
            return "danger"

        # danger: 单日跌幅超阈值 + 跌破MA20
        if ma_break and daily_ma_hit:
            return "danger"

        # danger: 相对指数大幅跑输 + 单日大跌
        if relative_hit and daily_hit:
            return "danger"

        # warning: 单日跌幅 > 3% 但 < 5%
        if pos.daily_return_pct <= -3.0:
            return "warning"

        # warning: 周累计跌幅 > 5% 但 < 8%
        if pos.weekly_return_pct <= -5.0:
            return "warning"

        return "normal"

    def _calc_relative_return(
        self,
        entry_price: float,
        current_price: float,
        entry_date: str,
        today: str,
    ) -> float:
        """计算个股相对沪深300的超额收益 (同期)。"""
        if self.index_data is None or len(self.index_data) == 0:
            return 0.0

        stock_ret = (current_price / entry_price - 1) * 100

        try:
            idx_df = self.index_data
            entry_dt = pd.Timestamp(entry_date)
            today_dt = pd.Timestamp(today)

            # 取最接近的指数收盘价
            idx_before = idx_df[idx_df.index <= entry_dt]
            idx_after = idx_df[idx_df.index <= today_dt]

            if len(idx_before) > 0 and len(idx_after) > 0:
                idx_entry = float(idx_before.iloc[-1]["close"])
                idx_current = float(idx_after.iloc[-1]["close"])
                if idx_entry > 0:
                    idx_ret = (idx_current / idx_entry - 1) * 100
                    return round(stock_ret - idx_ret, 2)
        except Exception:
            pass

        return 0.0

    @staticmethod
    def _get_week_start(dt: datetime) -> str:
        """返回本周一的日期字符串。"""
        monday = dt - timedelta(days=dt.weekday())
        return monday.strftime("%Y-%m-%d")

    def _print_summary(self, report: dict, alert: Optional[dict]) -> None:
        """终端彩色摘要输出。"""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  每日风险自测报告 — {report['report_date']}")
        print(f"{sep}")
        print(f"  调仓日: {report['rebalance_date']} "
              f"(距今 {report['days_since_rebalance']} 天)")
        print(f"  总权益: {report['total_equity']:,.0f} | "
              f"本周加权盈亏: {report['weekly_pnl_weighted_pct']:+.2f}%")
        print(f"  警报: {report['alerts_summary']['danger_count']} danger / "
              f"{report['alerts_summary']['warning_count']} warning")
        print(f"\n  {'代码':<8s} {'名称':<8s} {'入场价':>8s} {'最新价':>8s} "
              f"{'周收益':>8s} {'日收益':>8s} {'警报':>6s}")
        print(f"  {'-'*54}")

        for p in report["positions"]:
            alert_tag = {
                "normal": "  OK  ",
                "warning": "WARN! ",
                "danger": "DANGER",
            }.get(p["alert_level"], "  ?  ")
            print(
                f"  {p['code']:<8s} {p['name']:<8s} "
                f"{p['entry_price']:>8.2f} {p['current_price']:>8.2f} "
                f"{p['weekly_return_pct']:>+7.2f}% "
                f"{p['daily_return_pct']:>+7.2f}% "
                f"{alert_tag:>6s}"
            )

        if alert:
            print(f"\n  !!! 熔断触发 !!!")
            for inst in alert.get("instructions", []):
                print(f"  -> {inst}")
            print(f"  指令文件: {ALERT_FILE}")
        else:
            print(f"\n  状态: 一切正常, 无需干预")

        print(f"{sep}\n")


# ======================================================================
# 组合状态管理 (供周线策略调用)
# ======================================================================

def save_portfolio_state(
    rebalance_date: str,
    positions_data: list[dict],
    cash: float,
    total_equity: float,
    top_n: int = 5,
    backup_pool: Optional[list[str]] = None,
    thresholds: Optional[dict] = None,
    state_path: Optional[Path] = None,
) -> Path:
    """周线策略调仓后, 保存持仓状态供日度监控读取。

    应在 backtest_engine.run() 或 forward_simulation 的每周调仓后调用。

    Args:
        rebalance_date:  调仓日期 "YYYY-MM-DD"
        positions_data:  持仓列表, 每项含:
                         {code, name, sector, entry_price, shares, cost}
        cash:            剩余现金
        total_equity:    总权益
        top_n:           目标持仓数
        backup_pool:     备选池 (供熔断时替换)
        thresholds:       自定义阈值
        state_path:       状态文件路径

    Returns:
        state 文件的 Path
    """
    path = Path(state_path) if state_path else STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "rebalance_date": rebalance_date,
        "positions": positions_data,
        "cash": round(cash, 2),
        "total_equity": round(total_equity, 2),
        "top_n": top_n,
        "backup_pool": backup_pool or [],
        "thresholds": thresholds or DEFAULT_THRESHOLDS.copy(),
        "updated": datetime.now().isoformat(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    logger.info(
        f"持仓状态已保存: {path}\n"
        f"  调仓日: {rebalance_date}, "
        f"{len(positions_data)} 只持仓, "
        f"总权益 {total_equity:,.0f}"
    )
    return path


# ======================================================================
# CLI 入口
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="日度风险自测")
    parser.add_argument(
        "--check", action="store_true", default=True,
        help="执行每日风险扫描"
    )
    parser.add_argument(
        "--state", type=str, default=None,
        help="自定义持仓状态文件路径"
    )
    parser.add_argument(
        "--threshold-weekly", type=float, default=-8.0,
        help="周中累计跌幅阈值 (负数, 默认 -8%%)"
    )
    parser.add_argument(
        "--threshold-daily", type=float, default=-5.0,
        help="单日跌幅阈值 (负数, 默认 -5%%)"
    )
    args = parser.parse_args()

    # 自定义阈值
    custom_thresholds = DEFAULT_THRESHOLDS.copy()
    custom_thresholds["weekly_drawdown_pct"] = args.threshold_weekly
    custom_thresholds["daily_drop_pct"] = args.threshold_daily

    # 初始化
    monitor = DailyRiskMonitor(
        state_path=Path(args.state) if args.state else None,
        thresholds=custom_thresholds,
    )

    # 执行
    result = monitor.check()

    # 退出码
    if result["status"] == "ok":
        if result["alert"]:
            sys.exit(2)  # 有熔断
        sys.exit(0)      # 正常
    else:
        sys.exit(1)      # 异常
