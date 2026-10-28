"""
100万模拟盘大赛策略 — 双重回撤控制机制
========================================

两层防护:
  第一层 — 优化器硬约束 (数学层面)
    在组合优化阶段施加不等式约束，确保历史回测期:
    - 最大回撤 ≤ 9.5% (预留0.5%安全边际)
    - 单日 VaR(95%) ≤ 1.8%
    - 行业偏离度 ≤ ±3%

  第二层 — 动态风险预算 (运行时保护)
    实时监控组合权益曲线，按回撤深度自动调整风险敞口:
    - 回撤 ≥ 7%:  目标波动率下调30%, 股票仓位上限40%
    - 回撤 ≥ 9%:  熔断触发, 股票仓位强制降至20%, 仅保留高流动性标的
    - 回撤 ≤ 5%:  分3个交易日逐步恢复正常风险预算

所有风控阈值均从 config.RISK_CONTROL_CONFIG 读取，禁止硬编码。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RISK_CONTROL_CONFIG, RiskControlConfig


# ======================================================================
# 第一层: 优化器硬约束
# ======================================================================

@dataclass
class PortfolioOptimizer:
    """带硬约束的组合优化器。

    使用 scipy SLSQP 求解带不等式约束的权重优化问题。
    目标: 最大化 Sharpe Ratio
    约束:
      1. sum(w) = 1
      2. w_i >= 0 (禁止做空)
      3. 历史最大回撤 >= -9.5%  (即 max_dd >= -limit)
      4. 单日 VaR(95%) 绝对值 <= 1.8%
      5. 各行业权重偏离基准 <= ±3%
    """

    returns: pd.DataFrame           # 历史日收益, columns = asset codes
    sectors: dict[str, str]         # {asset_code: sector_name}
    benchmark_sector_weights: Optional[dict[str, float]] = None  # 基准行业权重
    risk_config: Optional[RiskControlConfig] = None

    def __post_init__(self):
        if self.risk_config is None:
            self.risk_config = RiskControlConfig.from_risk_config()
        if self.benchmark_sector_weights is None:
            # 默认等权基准
            unique_sectors = set(self.sectors.values())
            n_sec = len(unique_sectors)
            self.benchmark_sector_weights = {s: 1.0 / n_sec for s in unique_sectors}

        self.asset_list = list(self.returns.columns)
        self.n_assets = len(self.asset_list)
        self._sector_matrix = self._build_sector_matrix()

    def _build_sector_matrix(self) -> dict[str, np.ndarray]:
        """构建行业哑变量矩阵: sector → (n_assets,) 0/1 向量。"""
        unique = set(self.sectors.values())
        matrix = {}
        for sec in unique:
            mask = np.array([
                1.0 if self.sectors.get(a, "未知") == sec else 0.0
                for a in self.asset_list
            ])
            matrix[sec] = mask
        return matrix

    # ── 目标函数 ──
    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """负夏普比率 (最小化 → 最大化夏普)"""
        port_returns = self.returns.values @ weights
        mean_ret = port_returns.mean()
        std_ret = port_returns.std()
        if std_ret < 1e-10:
            return 1e6  # 惩罚零波动
        rf_daily = 0.02 / 252
        return -(mean_ret - rf_daily) / std_ret

    # ── 约束函数 ──
    def _max_drawdown_constraint(self, weights: np.ndarray) -> float:
        """不等式约束: max_drawdown + limit >= 0  (max_dd >= -limit)"""
        port_returns = self.returns.values @ weights
        cumulative = (1 + port_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = drawdowns.min()  # 负数, 如 -0.10
        return max_dd + self.risk_config.max_drawdown_hard_limit  # >= 0 则满足

    def _var_constraint(self, weights: np.ndarray) -> float:
        """不等式约束: VaR_limit - abs(VaR_95) >= 0"""
        port_returns = self.returns.values @ weights
        var_95 = np.percentile(port_returns, 5)  # 5%分位数, 负数
        return self.risk_config.daily_var_95_limit - abs(var_95)

    def _sector_deviation_constraints(self, weights: np.ndarray) -> list[dict]:
        """生成每个行业的偏离度约束 (返回 scipy 格式 dict 列表)。"""
        constraints = []
        limit = self.risk_config.sector_deviation_limit

        for sector, sector_mask in self._sector_matrix.items():
            benchmark_w = self.benchmark_sector_weights.get(sector, 0.0)
            # 行业实际权重
            actual_w = np.dot(sector_mask, weights)

            # 上界约束: actual_w <= benchmark_w + limit
            constraints.append({
                "type": "ineq",
                "fun": lambda w, b=benchmark_w, l=limit, m=sector_mask: b + l - np.dot(m, w),
            })
            # 下界约束: actual_w >= benchmark_w - limit
            constraints.append({
                "type": "ineq",
                "fun": lambda w, b=benchmark_w, l=limit, m=sector_mask: np.dot(m, w) - (b - l),
            })

        return constraints

    def optimize(self, initial_weights: Optional[np.ndarray] = None) -> dict:
        """执行带约束的权重优化。

        Returns:
            {
                "success": bool,
                "weights": np.ndarray (最优权重),
                "sharpe": float,
                "max_drawdown": float,
                "var_95": float,
                "sector_weights": {sector: weight},
                "message": str,
            }
        """
        try:
            from scipy.optimize import minimize, Bounds, LinearConstraint
        except ImportError:
            return {
                "success": False,
                "weights": np.ones(self.n_assets) / self.n_assets,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "sector_weights": {},
                "message": "scipy 未安装, 返回等权配置",
            }

        # 初始权重: 等权或给定
        if initial_weights is None:
            x0 = np.ones(self.n_assets) / self.n_assets
        else:
            x0 = initial_weights

        # 边界: 0 <= w_i <= 1
        bounds = Bounds([0.0] * self.n_assets, [1.0] * self.n_assets)

        # 线性约束: sum(w) = 1
        linear_constraint = LinearConstraint(
            np.ones(self.n_assets), lb=1.0, ub=1.0,
        )

        # 非线性约束
        constraints = [
            {"type": "ineq", "fun": self._max_drawdown_constraint},
            {"type": "ineq", "fun": self._var_constraint},
        ]
        constraints += self._sector_deviation_constraints(x0)

        result = minimize(
            self._negative_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[linear_constraint] + constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        weights = result.x
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()  # 归一化

        # 计算最终指标
        port_returns = self.returns.values @ weights
        cumulative = (1 + port_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(drawdowns.min())
        var_95 = float(np.percentile(port_returns, 5))
        sharpe_val = (port_returns.mean() / port_returns.std()) * np.sqrt(252) if port_returns.std() > 0 else 0.0

        # 行业权重
        sector_weights = {}
        for sec, mask in self._sector_matrix.items():
            sector_weights[sec] = round(float(np.dot(mask, weights)), 4)

        return {
            "success": result.success,
            "weights": weights,
            "sharpe": round(sharpe_val, 4),
            "max_drawdown": round(max_dd, 4),
            "var_95": round(var_95, 4),
            "sector_weights": sector_weights,
            "message": result.message,
        }


def optimize_portfolio(
    returns: pd.DataFrame,
    sectors: dict[str, str],
    benchmark_sector_weights: Optional[dict[str, float]] = None,
    risk_config: Optional[RiskControlConfig] = None,
) -> dict:
    """优化器硬约束入口函数 (便捷封装)。

    在组合优化阶段施加不等式约束，确保:
      - 历史回测期最大回撤 ≤ 9.5% (预留0.5%安全边际)
      - 单日 VaR(95%) ≤ 1.8%
      - 行业偏离度 ≤ ±3%

    Args:
        returns:                 历史日收益 DataFrame, columns = asset codes
        sectors:                 {asset_code: sector_name}
        benchmark_sector_weights: 基准行业权重 (None = 等权)
        risk_config:             风控配置 (None = 从 RISK_CONTROL_CONFIG 读取)

    Returns:
        {"success": bool, "weights": np.ndarray, "sharpe": float,
         "max_drawdown": float, "var_95": float, "sector_weights": dict}
    """
    optimizer = PortfolioOptimizer(
        returns=returns,
        sectors=sectors,
        benchmark_sector_weights=benchmark_sector_weights,
        risk_config=risk_config,
    )
    return optimizer.optimize()


# ======================================================================
# 第二层: 动态风险预算监控
# ======================================================================

@dataclass
class RiskBudgetState:
    """风险预算状态机"""
    status: str                    # "normal" | "warning" | "circuit_breaker" | "recovery"
    current_drawdown: float        # 当前回撤 (正数, 如 0.07 = 7%)
    target_vol_ratio: float        # 目标波动率缩放比例 (1.0 = 不缩放)
    equity_cap: float              # 股票仓位上限 (1.0 = 不限制)
    recovery_day: int              # 恢复期已过天数 (0 = 尚未进入恢复)
    action: str                    # 人类可读的动作描述


def risk_budget_monitor(
    portfolio_value_series: pd.Series,
    risk_config: Optional[RiskControlConfig] = None,
    previous_state: Optional[RiskBudgetState] = None,
) -> RiskBudgetState:
    """动态风险预算监控 — 运行时保护层。

    实时监控组合权益曲线，按回撤深度自动调整风险敞口:

    ┌──────────────┬──────────────────────────────────────────┐
    │  回撤深度     │  动作                                     │
    ├──────────────┼──────────────────────────────────────────┤
    │  < 5%        │  正常: 全风险预算                          │
    │  5% ~ 7%     │  恢复中: 逐步恢复 (如果之前触发过防御)      │
    │  7% ~ 9%     │  预警: 目标波动率下调30%, 股票仓位上限40%   │
    │  >= 9%       │  熔断: 股票仓位强降至20%, 仅高流动性标的    │
    └──────────────┴──────────────────────────────────────────┘

    恢复机制: 回撤回落至 5% 以下后, 分 recovery_days 个交易日
    逐步恢复正常风险预算, 避免"刚出ICU就进KTV"的二次冲击。

    Args:
        portfolio_value_series: 组合权益序列 (至少2个点)
        risk_config:            风控配置 (None = 从 RISK_CONTROL_CONFIG 读取)
        previous_state:         上一期的风险预算状态 (用于恢复期计数)

    Returns:
        RiskBudgetState: 当前风险预算状态
    """
    if risk_config is None:
        risk_config = RiskControlConfig.from_risk_config()

    # ── 计算当前回撤 ──
    if len(portfolio_value_series) < 2:
        return RiskBudgetState(
            status="normal",
            current_drawdown=0.0,
            target_vol_ratio=1.0,
            equity_cap=1.0,
            recovery_day=0,
            action="NORMAL: 数据不足, 默认全风险预算",
        )

    values = portfolio_value_series.values
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    current_dd = abs(float(drawdowns[-1]))  # 正数, 如 0.073 = 7.3%

    # ── 状态判定 ──
    budget = risk_config  # RiskControlConfig 已从 RISK_CONTROL_CONFIG 提取

    if current_dd >= budget.circuit_breaker_dd_threshold:
        # 熔断: 回撤 >= 9%
        return RiskBudgetState(
            status="circuit_breaker",
            current_drawdown=round(current_dd, 4),
            target_vol_ratio=budget.warning_vol_target_ratio,
            equity_cap=budget.circuit_breaker_equity_cap,
            recovery_day=0,
            action=(
                f"MELT_DOWN: 回撤 {current_dd*100:.1f}% >= {budget.circuit_breaker_dd_threshold*100:.0f}% "
                f"→ 股票仓位强降至 {budget.circuit_breaker_equity_cap*100:.0f}%, "
                f"仅保留高流动性标的 (日均成交>{budget.liquidity_min_daily_volume/1e8:.1f}亿)"
            ),
        )

    elif current_dd >= budget.warning_dd_threshold:
        # 预警: 回撤 >= 7%
        return RiskBudgetState(
            status="warning",
            current_drawdown=round(current_dd, 4),
            target_vol_ratio=budget.warning_vol_target_ratio,
            equity_cap=budget.warning_equity_cap,
            recovery_day=0,
            action=(
                f"DEFEND: 回撤 {current_dd*100:.1f}% >= {budget.warning_dd_threshold*100:.0f}% "
                f"→ 目标波动率 ×{budget.warning_vol_target_ratio:.0%}, "
                f"股票仓位上限 {budget.warning_equity_cap*100:.0f}%"
            ),
        )

    elif current_dd <= budget.recovery_dd_threshold:
        # 正常 / 恢复完成
        if previous_state and previous_state.status in ("warning", "circuit_breaker", "recovery"):
            # 从防御中恢复, 检查是否完成
            if previous_state.recovery_day >= budget.recovery_days:
                return RiskBudgetState(
                    status="normal",
                    current_drawdown=round(current_dd, 4),
                    target_vol_ratio=1.0,
                    equity_cap=1.0,
                    recovery_day=0,
                    action=(
                        f"NORMAL: 回撤已恢复至 {current_dd*100:.1f}% "
                        f"(< {budget.recovery_dd_threshold*100:.0f}%), "
                        f"恢复期完成, 全风险预算"
                    ),
                )
            else:
                # 仍在恢复期
                recovery_day = previous_state.recovery_day + 1
                progress = recovery_day / budget.recovery_days
                vol_ratio = budget.warning_vol_target_ratio + (1.0 - budget.warning_vol_target_ratio) * progress
                eq_cap = budget.warning_equity_cap + (1.0 - budget.warning_equity_cap) * progress
                return RiskBudgetState(
                    status="recovery",
                    current_drawdown=round(current_dd, 4),
                    target_vol_ratio=round(vol_ratio, 4),
                    equity_cap=round(eq_cap, 4),
                    recovery_day=recovery_day,
                    action=(
                        f"RECOVERY: 第 {recovery_day}/{budget.recovery_days} 天 "
                        f"→ 波动率 ×{vol_ratio:.0%}, 股票仓位上限 {eq_cap*100:.0f}%"
                    ),
                )

        # 完全正常
        return RiskBudgetState(
            status="normal",
            current_drawdown=round(current_dd, 4),
            target_vol_ratio=1.0,
            equity_cap=1.0,
            recovery_day=0,
            action=f"NORMAL: 回撤 {current_dd*100:.1f}%, 全风险预算",
        )

    else:
        # 5% < 回撤 < 7%: 灰色地带
        if previous_state and previous_state.status in ("warning", "circuit_breaker", "recovery"):
            # 之前触发过防御, 进入恢复模式
            recovery_day = previous_state.recovery_day + 1
            progress = min(recovery_day / budget.recovery_days, 1.0)
            vol_ratio = budget.warning_vol_target_ratio + (1.0 - budget.warning_vol_target_ratio) * progress
            eq_cap = budget.warning_equity_cap + (1.0 - budget.warning_equity_cap) * progress
            return RiskBudgetState(
                status="recovery",
                current_drawdown=round(current_dd, 4),
                target_vol_ratio=round(vol_ratio, 4),
                equity_cap=round(eq_cap, 4),
                recovery_day=recovery_day,
                action=(
                    f"RECOVERY: 第 {recovery_day}/{budget.recovery_days} 天 "
                    f"→ 波动率 ×{vol_ratio:.0%}, 股票仓位上限 {eq_cap*100:.0f}%"
                ),
            )

        # 从未触发防御, 保持正常
        return RiskBudgetState(
            status="normal",
            current_drawdown=round(current_dd, 4),
            target_vol_ratio=1.0,
            equity_cap=1.0,
            recovery_day=0,
            action=f"NORMAL: 回撤 {current_dd*100:.1f}%, 全风险预算",
        )


# ======================================================================
# 辅助: 流动性筛选
# ======================================================================

def filter_liquid_stocks(
    daily_volume: dict[str, float],
    turnover_rate: dict[str, float],
    risk_config: Optional[RiskControlConfig] = None,
) -> list[str]:
    """筛选高流动性标的 (熔断阶段使用)。

    Args:
        daily_volume:   {code: 日均成交额}
        turnover_rate:  {code: 换手率}
        risk_config:    风控配置

    Returns:
        符合流动性条件的股票代码列表
    """
    if risk_config is None:
        risk_config = RiskControlConfig.from_risk_config()

    liquid = []
    for code in daily_volume:
        vol = daily_volume.get(code, 0)
        turnover = turnover_rate.get(code, 0)
        if vol >= risk_config.liquidity_min_daily_volume and turnover >= risk_config.liquidity_min_turnover_rate:
            liquid.append(code)

    return liquid


# ======================================================================
# 自测入口
# ======================================================================

if __name__ == "__main__":
    # 生成模拟数据验证两层机制
    print("=" * 60)
    print("  双重回撤控制机制 — 自测")
    print("=" * 60)

    # ── 模拟日收益数据 (24只"股票", 1000天) ──
    np.random.seed(42)
    n_assets = 24
    n_days = 1000
    asset_codes = [f"stock_{i:02d}" for i in range(n_assets)]
    returns_data = np.random.randn(n_days, n_assets) * 0.02 + 0.0005  # 日均+0.05%，波动2%
    returns_df = pd.DataFrame(returns_data, columns=asset_codes)

    # 模拟行业映射
    sector_list = ["银行", "食品饮料", "电力设备", "医药生物", "电子",
                   "非银金融", "家用电器", "公用事业", "汽车", "农林牧渔",
                   "计算机", "商贸零售"]
    sectors = {}
    for i, code in enumerate(asset_codes):
        sectors[code] = sector_list[i % len(sector_list)]

    # ── 测试第一层: 优化器硬约束 ──
    print("\n[第一层] 优化器硬约束测试...")
    result = optimize_portfolio(returns_df, sectors)

    print(f"  求解成功: {result['success']}")
    print(f"  最优 Sharpe: {result['sharpe']:.4f}")
    print(f"  历史最大回撤: {result['max_drawdown']:.4f}  "
          f"(上限: ≤{RISK_CONTROL_CONFIG['max_drawdown_hard_limit']:.1%})")
    print(f"  日 VaR(95%):   {result['var_95']:.4f}  "
          f"(上限: ≤{RISK_CONTROL_CONFIG['daily_var_95_limit']:.1%})")
    print(f"  行业权重: ")
    for sec, w in sorted(result['sector_weights'].items()):
        bar = "█" * int(w * 50)
        print(f"    {sec:6s}: {w:.2%} {bar}")

    dd_ok = abs(result['max_drawdown']) <= RISK_CONTROL_CONFIG['max_drawdown_hard_limit']
    var_ok = abs(result['var_95']) <= RISK_CONTROL_CONFIG['daily_var_95_limit']
    print(f"\n  Hard Constraint Check: MaxDD {'PASS' if dd_ok else 'FAIL'} | VaR {'PASS' if var_ok else 'FAIL'}")

    # ── 测试第二层: 动态风险预算 ──
    print("\n[第二层] 动态风险预算监控测试...")

    # 模拟权益曲线: 正常→下跌→暴跌→恢复
    equity_scenarios = {
        "正常市场": [100, 101, 102, 103, 102.5, 104, 105],
        "预警 (7%)": [100, 98, 96, 95, 94.5, 93.5, 93],     # -7%
        "熔断 (9%)": [100, 97, 94, 92, 91, 90.5, 90.8],      # -9.2%
        "恢复中 (5%)": [100, 95, 93, 93.5, 94, 94.5, 94.8],  # 从-7%恢复到-5.2%
        "恢复完成":    [100, 95, 96, 97, 97.5, 98, 98.5],     # -1.5%
    }

    for label, equity in equity_scenarios.items():
        state = risk_budget_monitor(pd.Series(equity))
        print(f"  {label:12s}: dd={state.current_drawdown:.1%} | "
              f"status={state.status:15s} | vol×{state.target_vol_ratio:.0%} | "
              f"eq_cap={state.equity_cap:.0%}")
        print(f"    → {state.action}")

    # ── 测试恢复期天数计数 ──
    print("\n[恢复期计数测试] 模拟回撤从 9% → 4% 的3天渐进恢复...")

    # 制造一个从9%回撤恢复的场景
    equity_path = [100, 95, 93, 91, 91.5, 92.5, 94, 95.5, 97, 98, 99, 100]
    prev_state = None
    for i in range(3, len(equity_path)):
        series = pd.Series(equity_path[:i + 1])
        state = risk_budget_monitor(series, previous_state=prev_state)
        print(f"  Day {i}: equity={series.iloc[-1]:.0f} | dd={state.current_drawdown:.1%} | "
              f"{state.status} | recovery_day={state.recovery_day}")
        prev_state = state

    print("\n" + "=" * 60)
    print("  自测完成")
    print("=" * 60)
