"""
多周期配置管理
- 从老项目 strategy_config.json 提取参数，统一管理
- 支持 TIMEFRAME = DAILY / WEEKLY 切换
- 通读 .env 覆盖关键参数
- 支持多 profile: --profile 1M → 加载 .env.1M
- 包含大赛策略专用风控配置 (RISK_CONTROL_CONFIG)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

Timeframe = Literal["DAILY", "WEEKLY"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# 大赛策略专用风控配置 (所有阈值集中管理，禁止硬编码)
# ======================================================================

RISK_CONTROL_CONFIG: dict = {
    # ── 优化器硬约束 (数学层面保障) ──
    "max_drawdown_hard_limit": 0.095,       # 历史回测期最大回撤 ≤ 9.5% (预留0.5%安全边际)
    "daily_var_95_limit": 0.018,            # 单日 VaR(95%) ≤ 1.8%
    "sector_deviation_limit": 0.03,         # 行业偏离度 ≤ ±3%

    # ── 动态风险预算 (运行时保护) ──
    "risk_budget": {
        "warning_dd_threshold": 0.07,            # 回撤达7%: 触发预警
        "circuit_breaker_dd_threshold": 0.09,    # 回撤达9%: 触发熔断
        "recovery_dd_threshold": 0.05,           # 回撤恢复至5%以下: 解除防御
        "warning_vol_target_ratio": 0.70,        # 预警阶段: 目标波动率降至原来的70%
        "warning_equity_cap": 0.40,              # 预警阶段: 股票仓位上限40%
        "circuit_breaker_equity_cap": 0.20,      # 熔断阶段: 股票仓位强制降至20%
        "recovery_days": 3,                      # 恢复期: 分3个交易日逐步恢复正常风险预算
    },

    # ── 流动性筛选 (熔断阶段仅保留高流动性标的) ──
    "liquidity_min_daily_volume": 50_000_000,    # 最低日均成交额 (5000万)
    "liquidity_min_turnover_rate": 0.005,        # 最低换手率 (0.5%)
}


# ======================================================================
# 策略配置 Dataclasses
# ======================================================================

@dataclass
class FactorConfig:
    """因子参数"""
    momentum_window: int = 40       # 动量窗口（日）
    pullback_window: int = 90       # 回撤窗口（日）
    sentiment_ma_window: int = 20   # 情感均线窗口

    # 每周聚合时窗口自动缩放 (WEEKLY: 窗口/5)
    def for_timeframe(self, tf: Timeframe) -> "FactorConfig":
        if tf == "WEEKLY":
            return FactorConfig(
                momentum_window=max(4, self.momentum_window // 5),
                pullback_window=max(12, self.pullback_window // 5),
                sentiment_ma_window=max(4, self.sentiment_ma_window // 5),
            )
        return self


@dataclass
class ScoringConfig:
    """评分方法"""
    method: str = "zscore_sum"
    momentum_weight: float = 1.0
    pullback_weight: float = 1.0
    sentiment_weight: float = 0.0  # 基准默认纯价格；新闻因子需显式启用


@dataclass
class SentimentDataConfig:
    """情感数据质量门槛。

    价格代理情感默认禁用，避免把同一份价格动量重复包装为“独立情感”。
    """
    allow_price_proxy: bool = False
    min_sqlite_days: int = 30
    min_pool_coverage: float = 0.60


@dataclass
class SectorNeutralConfig:
    """行业中性化"""
    enabled: bool = True
    max_per_sector: int = 1


@dataclass
class PortfolioConfig:
    """持仓参数"""
    top_n: int = 5
    weighting: str = "equal_weight"
    risk_weight_window: int = 60
    max_position_weight: float = 0.35
    rank_buffer: int = 0
    rebalance_frequency: str = "monthly"  # daily / weekly / monthly


@dataclass
class ExecutionConfig:
    """执行参数"""
    initial_cash: float = 100_000
    commission: float = 0.0003
    min_commission: float = 5.0
    transfer_fee_rate: float = 0.00001
    stamp_duty_rate: float = 0.0005
    historical_stamp_duty_rate: float = 0.001
    stamp_duty_cutover_date: str = "2023-08-28"
    slippage_bps: float = 5.0
    max_volume_participation: float = 0.05
    volume_lookback: int = 20
    enforce_price_limits: bool = True
    min_lot: int = 100
    stop_loss_pct: float = 0.0          # 0 = 禁用；基准不使用个股止损
    trailing_stop_pct: float = 0.0      # 0 = 禁用
    time_stop_months: int = 0           # 0 = 禁用


@dataclass
class RiskControlConfig:
    """风控参数 (从 RISK_CONTROL_CONFIG 字典初始化)"""
    max_drawdown_hard_limit: float = 0.095
    daily_var_95_limit: float = 0.018
    sector_deviation_limit: float = 0.03
    warning_dd_threshold: float = 0.07
    circuit_breaker_dd_threshold: float = 0.09
    recovery_dd_threshold: float = 0.05
    warning_vol_target_ratio: float = 0.70
    warning_equity_cap: float = 0.40
    circuit_breaker_equity_cap: float = 0.20
    recovery_days: int = 3
    liquidity_min_daily_volume: float = 50_000_000
    liquidity_min_turnover_rate: float = 0.005

    @classmethod
    def from_risk_config(cls, config: dict | None = None) -> "RiskControlConfig":
        """从 RISK_CONTROL_CONFIG 字典创建实例，允许 env 覆盖。"""
        cfg = config or RISK_CONTROL_CONFIG
        budget = cfg.get("risk_budget", {})
        return cls(
            max_drawdown_hard_limit=float(
                os.getenv("RC_MAX_DRAWDOWN", cfg.get("max_drawdown_hard_limit", 0.095))
            ),
            daily_var_95_limit=float(
                os.getenv("RC_DAILY_VAR", cfg.get("daily_var_95_limit", 0.018))
            ),
            sector_deviation_limit=float(
                os.getenv("RC_SECTOR_DEVIATION", cfg.get("sector_deviation_limit", 0.03))
            ),
            warning_dd_threshold=float(
                os.getenv("RC_WARNING_DD", budget.get("warning_dd_threshold", 0.07))
            ),
            circuit_breaker_dd_threshold=float(
                os.getenv("RC_BREAKER_DD", budget.get("circuit_breaker_dd_threshold", 0.09))
            ),
            recovery_dd_threshold=float(
                os.getenv("RC_RECOVERY_DD", budget.get("recovery_dd_threshold", 0.05))
            ),
            warning_vol_target_ratio=float(
                os.getenv("RC_WARNING_VOL", budget.get("warning_vol_target_ratio", 0.70))
            ),
            warning_equity_cap=float(
                os.getenv("RC_WARNING_EQ_CAP", budget.get("warning_equity_cap", 0.40))
            ),
            circuit_breaker_equity_cap=float(
                os.getenv("RC_BREAKER_EQ_CAP", budget.get("circuit_breaker_equity_cap", 0.20))
            ),
            recovery_days=int(
                os.getenv("RC_RECOVERY_DAYS", budget.get("recovery_days", 3))
            ),
            liquidity_min_daily_volume=float(
                os.getenv("RC_LIQ_MIN_VOL", cfg.get("liquidity_min_daily_volume", 50_000_000))
            ),
            liquidity_min_turnover_rate=float(
                os.getenv("RC_LIQ_MIN_TURNOVER", cfg.get("liquidity_min_turnover_rate", 0.005))
            ),
        )


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str = "2020-01-01"
    end_date: str = "2025-05-12"
    mode: str = "price_only"  # price_only / sentiment_only / hybrid
    export_dir: str = "data/backtest_results"


@dataclass
class UniverseConfig:
    """股票池配置。"""
    source: str = "hs300_pit"  # hs300_pit / sse_composite_a_pit / legacy
    snapshot_dir: str = "data/universe/hs300"
    strict_point_in_time: bool = True


@dataclass
class MarketTimingConfig:
    """大盘择时"""
    enabled: bool = False
    index: str = "000300"         # 沪深300
    ma_window: int = 20


@dataclass
class StrategyConfig:
    """总配置"""
    strategy_name: str = "情感因子多周期策略"
    version: str = "1.0"
    timeframe: Timeframe = "DAILY"
    profile: str = ""             # 当前 profile 名称 (空 = 默认)
    baseline_mode: bool = True     # 纯价格、无个股止损的可复现基准
    factors: FactorConfig = field(default_factory=FactorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sentiment_data: SentimentDataConfig = field(default_factory=SentimentDataConfig)
    sector_neutral: SectorNeutralConfig = field(default_factory=SectorNeutralConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk_control: RiskControlConfig = field(default_factory=RiskControlConfig)
    market_timing: MarketTimingConfig = field(default_factory=MarketTimingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)

    @classmethod
    def from_env(cls, profile: Optional[str] = None) -> "StrategyConfig":
        """从环境变量加载配置，.env 优先。

        支持多 profile:
          - profile=None:  使用默认 .env
          - profile="1M":  先加载 .env.1M (覆盖项), 再回退 .env (公共项)
          - 也可通过环境变量 PROFILE=1M 指定

        TIMEFRAME 同时驱动：
        - 因子窗口缩放 (DAILY 不缩放, WEEKLY / 5)
        - 回测调仓频率 (DAILY → monthly, WEEKLY → weekly)
        - 年化周期数 (DAILY → 252, WEEKLY → 52)
        """
        # ── 确定 profile ──
        resolved_profile = profile or os.getenv("PROFILE", "")

        # ── 加载 .env 文件 ──
        env_dir = PROJECT_ROOT
        if resolved_profile:
            # 先加载 profile 专用配置 (高优先级)
            profile_env = env_dir / f".env.{resolved_profile}"
            if profile_env.exists():
                load_dotenv(profile_env, override=True)
            # 再加载默认 .env (低优先级, 不覆盖已有的)
            default_env = env_dir / ".env"
            if default_env.exists():
                load_dotenv(default_env, override=False)
        else:
            # 仅加载默认 .env
            default_env = env_dir / ".env"
            if default_env.exists():
                load_dotenv(default_env)

        # ── TIMEFRAME ──
        tf_raw = os.getenv("TIMEFRAME", "DAILY").upper()
        if tf_raw not in ("DAILY", "WEEKLY"):
            raise ValueError(f"TIMEFRAME 必须是 DAILY 或 WEEKLY，当前: {tf_raw}")

        # TIMEFRAME → 调仓频率自动推导 (可被显式 REBALANCE_FREQ 覆盖)
        default_freq = {"DAILY": "monthly", "WEEKLY": "weekly"}[tf_raw]
        rebalance_freq = os.getenv("REBALANCE_FREQ", default_freq)
        baseline_mode = os.getenv("BASELINE_MODE", "true").lower() == "true"

        # 旧 .env 中可能还保留混合因子和止损参数。基准模式优先于这些旧值，
        # 只有显式设置 BASELINE_MODE=false 才恢复实验配置。
        sentiment_weight = (
            0.0 if baseline_mode else float(os.getenv("SENTIMENT_WEIGHT", "0.3"))
        )
        stop_loss_pct = (
            0.0 if baseline_mode else float(os.getenv("STOP_LOSS_PCT", "0.12"))
        )
        trailing_stop_pct = (
            0.0 if baseline_mode else float(os.getenv("TRAILING_STOP_PCT", "0.10"))
        )
        time_stop_months = (
            0 if baseline_mode else int(os.getenv("TIME_STOP_MONTHS", "2"))
        )
        backtest_mode = (
            "price_only" if baseline_mode else os.getenv("BACKTEST_MODE", "hybrid")
        )

        return cls(
            strategy_name=os.getenv("STRATEGY_NAME", "价格因子多周期策略"),
            timeframe=tf_raw,  # type: ignore[arg-type]
            profile=resolved_profile,
            baseline_mode=baseline_mode,
            factors=FactorConfig(
                momentum_window=int(os.getenv("MOMENTUM_WINDOW", "40")),
                pullback_window=int(os.getenv("PULLBACK_WINDOW", "90")),
                sentiment_ma_window=int(os.getenv("SENTIMENT_MA_WINDOW", "20")),
            ),
            scoring=ScoringConfig(
                sentiment_weight=sentiment_weight,
            ),
            sentiment_data=SentimentDataConfig(
                allow_price_proxy=(
                    os.getenv("ALLOW_PRICE_SENTIMENT_PROXY", "false").lower() == "true"
                ),
                min_sqlite_days=int(os.getenv("MIN_SQLITE_DAYS", "30")),
                min_pool_coverage=float(os.getenv("MIN_NEWS_COVERAGE", "0.60")),
            ),
            sector_neutral=SectorNeutralConfig(
                max_per_sector=int(os.getenv("MAX_PER_SECTOR", "1")),
            ),
            portfolio=PortfolioConfig(
                top_n=int(os.getenv("TOP_N", "5")),
                weighting=os.getenv("WEIGHTING", "equal_weight"),
                risk_weight_window=int(os.getenv("RISK_WEIGHT_WINDOW", "60")),
                max_position_weight=float(
                    os.getenv("MAX_POSITION_WEIGHT", "0.35")
                ),
                rank_buffer=int(os.getenv("RANK_BUFFER", "0")),
                rebalance_frequency=rebalance_freq,
            ),
            execution=ExecutionConfig(
                initial_cash=float(os.getenv("INITIAL_CASH", "100000")),
                commission=float(os.getenv("COMMISSION", "0.0003")),
                min_commission=float(os.getenv("MIN_COMMISSION", "5")),
                transfer_fee_rate=float(
                    os.getenv("TRANSFER_FEE_RATE", "0.00001")
                ),
                stamp_duty_rate=float(
                    os.getenv("STAMP_DUTY_RATE", "0.0005")
                ),
                historical_stamp_duty_rate=float(
                    os.getenv("HISTORICAL_STAMP_DUTY_RATE", "0.001")
                ),
                stamp_duty_cutover_date=os.getenv(
                    "STAMP_DUTY_CUTOVER_DATE", "2023-08-28"
                ),
                slippage_bps=float(os.getenv("SLIPPAGE_BPS", "5")),
                max_volume_participation=float(
                    os.getenv("MAX_VOLUME_PARTICIPATION", "0.05")
                ),
                volume_lookback=int(os.getenv("VOLUME_LOOKBACK", "20")),
                enforce_price_limits=(
                    os.getenv("ENFORCE_PRICE_LIMITS", "true").lower() == "true"
                ),
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
                time_stop_months=time_stop_months,
            ),
            risk_control=RiskControlConfig.from_risk_config(),
            market_timing=MarketTimingConfig(
                enabled=(
                    False if baseline_mode
                    else os.getenv("MARKET_TIMING", "false").lower() == "true"
                ),
            ),
            backtest=BacktestConfig(
                start_date=os.getenv("BACKTEST_START", "2020-01-01"),
                end_date=os.getenv("BACKTEST_END", "2025-05-12"),
                mode=backtest_mode,
            ),
            universe=UniverseConfig(
                source=os.getenv("UNIVERSE_SOURCE", "hs300_pit"),
                snapshot_dir=os.getenv(
                    "UNIVERSE_SNAPSHOT_DIR", "data/universe/hs300"
                ),
                strict_point_in_time=(
                    os.getenv("UNIVERSE_STRICT_PIT", "true").lower() == "true"
                ),
            ),
        )

    @property
    def periods_per_year(self) -> int:
        """年化周期数（对齐调仓频率）"""
        freq = self.portfolio.rebalance_frequency
        return {"daily": 252, "weekly": 52, "monthly": 12}[freq]

    @property
    def rebalance_label(self) -> str:
        """调仓频率中文标签"""
        freq = self.portfolio.rebalance_frequency
        return {"daily": "日度调仓", "weekly": "周度调仓", "monthly": "月度调仓"}[freq]


# 全局默认配置实例
config = StrategyConfig.from_env()
