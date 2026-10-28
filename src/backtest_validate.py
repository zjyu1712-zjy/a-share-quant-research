"""
情感因子回测验证 — 三模式对比
- sentiment_only: 纯情感因子选股
- price_only: 纯价格因子选股 (动量+回撤)
- hybrid: 混合因子 (价格+情感)

对比维度: 收益、风险、换手率、情感覆盖率、相关性
输出: 控制台报告 + JSON + 对比图表 + CSV
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.backtest_engine import BacktestEngine
from src.config import StrategyConfig, Timeframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"


def _run_one(config: StrategyConfig, mode: str, start: str, end: str) -> dict:
    """运行单个模式回测，返回结果摘要。"""
    label = {
        "sentiment_only": "纯情感因子",
        "price_only": "纯价格因子",
        "hybrid": "混合因子",
    }[mode]

    logger.info(f"开始回测: {label}")
    engine = BacktestEngine(config)
    engine.load_price_data(start, end)
    engine.load_sentiment_data(start, end)
    results = engine.run(start, end, mode=mode)  # type: ignore[arg-type]

    perf = results.get("performance", {})
    stats = results.get("stats", {})

    return {
        "mode": mode,
        "label": label,
        "perf": perf,
        "stats": stats,
        "equity_curve": results.get("equity_curve", pd.DataFrame()),
        "engine": engine,
    }


def _extract_metrics(result: dict) -> dict:
    """从回测结果提取核心指标。"""
    perf = result["perf"]
    stats = result["stats"]
    return {
        "mode": result["mode"],
        "label": result["label"],
        "final_equity": stats.get("final_equity", 0),
        "total_return_pct": perf.get("total_return", 0),
        "cagr_pct": perf.get("cagr", 0),
        "sharpe": perf.get("sharpe", 0),
        "sortino": perf.get("sortino", 0),
        "max_drawdown_pct": perf.get("max_drawdown", 0),
        "calmar": perf.get("calmar", 0),
        "ann_volatility_pct": perf.get("ann_volatility", 0),
        "win_rate_pct": perf.get("win_rate", 0),
        "pl_ratio": perf.get("pl_ratio", 0),
        "avg_turnover_pct": perf.get("avg_turnover_pct", 0),
        "best_return_pct": perf.get("best_return", 0),
        "worst_return_pct": perf.get("worst_return", 0),
        "avg_deployment_pct": perf.get("avg_deployment_pct", 0),
        "stop_loss_events": stats.get("stop_loss_events", 0),
        "time_stop_events": stats.get("time_stop_events", 0),
        "risk_off_dates": stats.get("risk_off_dates", 0),
        "total_dates": stats.get("total_dates", 0),
        "years": stats.get("years", 0),
    }


def compare(results: list[dict], timeframe: str = "monthly") -> pd.DataFrame:
    """生成三模式对比 DataFrame。"""
    rows = [_extract_metrics(r) for r in results]
    df = pd.DataFrame(rows)
    df.set_index("mode", inplace=True)
    return df


def print_comparison(df: pd.DataFrame) -> None:
    """打印格式化对比报告。"""
    W = 72
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print()
    print("=" * W)
    print(f"  情感因子回测验证 — 三模式对比报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * W)

    modes = df.index.tolist()
    metrics_display = [
        ("总收益率 (%)", "total_return_pct", "{:.2f}", "higher"),
        ("年化收益 CAGR (%)", "cagr_pct", "{:.2f}", "higher"),
        ("夏普比率", "sharpe", "{:.3f}", "higher"),
        ("索提诺比率", "sortino", "{:.3f}", "higher"),
        ("卡玛比率", "calmar", "{:.3f}", "higher"),
        ("最大回撤 (%)", "max_drawdown_pct", "{:.2f}", "lower"),
        ("年化波动率 (%)", "ann_volatility_pct", "{:.2f}", "lower"),
        ("胜率 (%)", "win_rate_pct", "{:.1f}", "higher"),
        ("盈亏比", "pl_ratio", "{:.3f}", "higher"),
        ("平均换手率 (%)", "avg_turnover_pct", "{:.1f}", "lower"),
        ("资金利用率 (%)", "avg_deployment_pct", "{:.1f}", "higher"),
        ("最佳单期 (%)", "best_return_pct", "{:.2f}", "higher"),
        ("最差单期 (%)", "worst_return_pct", "{:.2f}", "higher"),
        ("止损触发次数", "stop_loss_events", "{}", "lower"),
    ]

    def _best(row, col, direction):
        vals = df[col]
        return vals.idxmax() if direction == "higher" else vals.idxmin()

    print(f"\n  {'─' * 68}")
    print(f"  {'指标':<22s}", end="")
    for m in modes:
        print(f"  {str(m)[:16]:>16s}", end="")
    print(f"  {'最佳':>8s}")
    print(f"  {'-' * 68}")

    for label, col, fmt, direction in metrics_display:
        vals = df[col]
        best_mode = _best(vals, col, direction)
        print(f"  {label:<22s}", end="")
        for m in modes:
            v = vals.get(m, 0)
            mark = " *" if m == best_mode and len(modes) > 1 else "  "
            text = f"{fmt.format(v)}{mark}"
            print(f"  {text:>16s}", end="")
        print(f"  {best_mode:>8s}")

    print(f"  {'-' * 68}")

    # 综合评分
    def _composite_score(row: pd.Series) -> int:
        score = 0
        score += row["sharpe"] > 1
        score += row["sortino"] > 1
        score += row["calmar"] > 0.3
        score += row["win_rate_pct"] > 50
        score += row["pl_ratio"] > 1.2
        return score

    print(f"\n  {'模式':<22s}  {'综合评分':>8s}  {'评价':>12s}")
    verdicts = {5: "*****", 4: "****", 3: "***", 2: "**", 1: "*", 0: "需改进"}
    for m in modes:
        s = _composite_score(df.loc[m])
        print(f"  {m:<22s}  {s:>6d}/5    {verdicts[s]:>12s}")

    # 情感覆盖说明
    print(f"\n  {YELLOW}说明:{RESET}")
    print(f"  - sentiment_only: 仅用情感因子 (sent_momentum + volatility + positive_ratio)")
    print(f"  - price_only: 仅用传统因子 (momentum + pullback)")
    print(f"  - hybrid: 价格因子 + 情感因子 × 0.5 权重")
    print()
    print("=" * W)
    print()


def plot_comparison(results: list[dict], save_path: Optional[Path] = None) -> None:
    """绘制三模式权益曲线 + 回撤对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    colors = {"sentiment_only": "#e74c3c", "price_only": "#3498db", "hybrid": "#2ecc71"}
    labels_zh = {"sentiment_only": "纯情感", "price_only": "纯价格", "hybrid": "混合"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) 权益曲线
    ax1 = axes[0, 0]
    for r in results:
        eq = r.get("equity_curve", pd.DataFrame())
        if len(eq) < 2:
            continue
        eq_series = eq["equity"]
        # 全部归一化到 100
        normalized = eq_series / eq_series.iloc[0] * 100
        ax1.plot(normalized.index, normalized.values,
                 color=colors[r["mode"]], linewidth=1.2,
                 label=labels_zh[r["mode"]], alpha=0.85)
    ax1.set_title("Normalized Equity (Base=100)")
    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

    # (0,1) 回撤
    ax2 = axes[0, 1]
    for r in results:
        eq = r.get("equity_curve", pd.DataFrame())
        if len(eq) < 2:
            continue
        eq_series = eq["equity"]
        cummax = eq_series.cummax()
        dd = (eq_series - cummax) / cummax * 100
        ax2.fill_between(dd.index, dd.values, 0,
                         color=colors[r["mode"]], alpha=0.15)
        ax2.plot(dd.index, dd.values, color=colors[r["mode"]],
                 linewidth=0.8, label=labels_zh[r["mode"]], alpha=0.7)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # (1,0) 年化收益-CAGR vs MaxDD 散点
    ax3 = axes[1, 0]
    metrics = [_extract_metrics(r) for r in results]
    for i, m in enumerate(metrics):
        ax3.scatter(m["max_drawdown_pct"], m["cagr_pct"],
                     color=colors[m["mode"]], s=120, zorder=5,
                     label=labels_zh[m["mode"]])
        ax3.annotate(labels_zh[m["mode"]],
                     (m["max_drawdown_pct"], m["cagr_pct"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9)
    ax3.set_xlabel("Max Drawdown (%)")
    ax3.set_ylabel("CAGR (%)")
    ax3.set_title("Risk-Return Profile")
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.axvline(x=0, color="gray", linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    # (1,1) 关键指标条形图
    ax4 = axes[1, 1]
    bar_metrics = ["sharpe", "sortino", "calmar", "win_rate_pct"]
    bar_labels = ["Sharpe", "Sortino", "Calmar", "WinRate%"]
    x_pos = np.arange(len(bar_labels))
    width = 0.25

    for i, m in enumerate(metrics):
        values = [m.get(k, 0) for k in bar_metrics]
        # normalize for bar chart (win_rate scaled)
        values_scaled = [v / (10 if k == "win_rate_pct" else 1) for v, k in
                         zip(values, bar_metrics)]
        offset = (i - 1) * width
        bars = ax4.bar(x_pos + offset, values_scaled, width,
                       color=colors[m["mode"]], label=labels_zh[m["mode"]], alpha=0.75)
        for bar, orig_val, k in zip(bars, values, bar_metrics):
            fmt = "{:.1f}" if k == "win_rate_pct" else "{:.2f}"
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     fmt.format(orig_val), ha="center", va="bottom", fontsize=6)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bar_labels)
    ax4.set_title("Key Metrics Comparison")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  对比图表已保存: {save_path}")
    else:
        plt.close(fig)


def export_results(results: list[dict], df: pd.DataFrame, output_dir: Path) -> None:
    """导出结果到文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 对比表
    df.to_csv(output_dir / f"comparison_{timestamp}.csv", encoding="utf-8-sig")
    print(f"  对比表: {output_dir / f'comparison_{timestamp}.csv'}")

    # 各模式权益曲线
    for r in results:
        eq = r.get("equity_curve", pd.DataFrame())
        if len(eq) > 0:
            eq.to_csv(output_dir / f"{r['mode']}_equity_{timestamp}.csv",
                      encoding="utf-8-sig")

    # 完整 JSON
    json_data = {
        "timestamp": timestamp,
        "comparison": df.reset_index().to_dict(orient="records"),
        "details": {
            r["mode"]: {
                "stats": {k: v for k, v in r["stats"].items()
                          if not isinstance(v, pd.DataFrame)},
            }
            for r in results
        },
    }
    with open(output_dir / f"report_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2,
                  default=lambda x: str(x) if isinstance(x, pd.Timestamp) else None)

    print(f"  报告: {output_dir / f'report_{timestamp}.json'}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="情感因子回测验证 — 三模式对比")
    parser.add_argument("--start", type=str, default=None, help="回测起始 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="回测结束 YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=None, help="初始资金")
    parser.add_argument("--top-n", type=int, default=None, help="持仓数")
    parser.add_argument("--timeframe", choices=["DAILY", "WEEKLY"], default=None,
                        help="时间周期 (默认读取 .env)")
    parser.add_argument("--export", type=str, default=None, help="导出目录")
    parser.add_argument("--plot", type=str, default=None, help="图表保存路径")
    parser.add_argument("--no-report", action="store_true", help="不打印报告")
    args = parser.parse_args()

    config = StrategyConfig.from_env()

    # CLI 覆盖
    timeframe = args.timeframe or config.timeframe
    config.timeframe = timeframe  # type: ignore[assignment]
    config.backtest.mode = "hybrid"  # 总是三种模式各跑一次

    if args.capital:
        config.execution.initial_cash = args.capital
    if args.top_n:
        config.portfolio.top_n = args.top_n

    start = args.start or config.backtest.start_date
    end = args.end or config.backtest.end_date

    print(f"\n  {'='*62}")
    print(f"  |  情感因子回测验证 — 周线量化系统{' ' * 22}|")
    print(f"  {'='*62}")
    print(f"  |  时间周期: {timeframe:<6s}  调仓: {config.rebalance_label:<8s}           |")
    print(f"  |  区间: {start} ~ {end}            |")
    print(f"  |  资金: {config.execution.initial_cash:,.0f}  持仓: Top {config.portfolio.top_n}  费率: {config.execution.commission*100:.2f}%     |")
    print(f"  {'='*62}")
    print()

    # 三模式回测
    modes = ["sentiment_only", "price_only", "hybrid"]
    results = []
    mode_labels = {
        "sentiment_only": "纯情感因子",
        "price_only": "纯价格因子",
        "hybrid": "混合因子 (价格+情感)",
    }

    for mode in modes:
        print(f"  >> 运行 {mode_labels[mode]} ...")
        try:
            r = _run_one(config, mode, start, end)
            results.append(r)
            eq = r.get("equity_curve", pd.DataFrame())
            n_periods = len(eq) if len(eq) > 0 else 0
            perf = r["perf"]
            print(f"     [OK] 完成: {n_periods} 期, "
                  f"CAGR {perf.get('cagr', 0):.1f}%, "
                  f"Sharpe {perf.get('sharpe', 0):.2f}, "
                  f"MaxDD {perf.get('max_drawdown', 0):.1f}%")
        except Exception as e:
            logger.exception(f"{mode} 回测失败: {e}")
            print(f"     [FAIL] 失败: {e}")

    if len(results) < 2:
        print("\n  回测结果不足 (至少需要 2 个模式成功)，无法对比。")
        sys.exit(1)

    # 生成对比
    freq = config.portfolio.rebalance_frequency
    df = compare(results, freq)

    if not args.no_report:
        print_comparison(df)

    # 导出
    export_dir = Path(args.export) if args.export else RESULTS_DIR
    export_results(results, df, export_dir)

    # 绘图
    plot_path = Path(args.plot) if args.plot else (export_dir / f"comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plot_comparison(results, plot_path)

    print(f"\n  验证完成! {'=' * 50}")


if __name__ == "__main__":
    main()
