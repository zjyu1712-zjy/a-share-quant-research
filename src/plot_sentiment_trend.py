"""
情感趋势可视化 — 读取 SQLite 新闻数据，绘制情感均值折线 + 利好/利空柱状图。

用法:
    python -m src.plot_sentiment_trend              # 默认 30 天
    python -m src.plot_sentiment_trend --days 7     # 近 7 天
    python -m src.plot_sentiment_trend --save       # 保存为 PNG
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── 后端：无显示器时自动用 Agg ───────────────────────────
try:
    import os as _os
    if _os.name == "nt":
        import ctypes
        if not ctypes.windll.user32.GetSystemMetrics(0):
            matplotlib.use("Agg")
    elif sys.platform.startswith("linux") and "DISPLAY" not in _os.environ:
        matplotlib.use("Agg")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "financial_news.db"

# ── 中文字体 ──────────────────────────────────────────────
for font in [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "SimHei",
]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def load_data(db_path: Path, days: int) -> pd.DataFrame:
    """按日聚合情感数据。"""
    if not db_path.exists():
        print(f"[错误] 数据库不存在: {db_path}")
        sys.exit(1)

    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """
        SELECT fetch_date,
               COUNT(*)          AS total,
               ROUND(AVG(sentiment_score), 4) AS avg_score,
               SUM(CASE WHEN sentiment_label = '利好' THEN 1 ELSE 0 END) AS positive,
               SUM(CASE WHEN sentiment_label = '利空' THEN 1 ELSE 0 END) AS negative,
               SUM(CASE WHEN sentiment_label = '中性' THEN 1 ELSE 0 END) AS neutral
        FROM news_archive
        WHERE fetch_date >= ?
          AND sentiment_score IS NOT NULL
          AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
          AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
        GROUP BY fetch_date
        ORDER BY fetch_date ASC
        """,
        conn,
        params=(start_date,),
        parse_dates=["fetch_date"],
    )
    conn.close()
    return df


def plot(df: pd.DataFrame, days: int, save: bool = False):
    """绘制情感趋势图。"""
    if df.empty:
        print("[提示] 没有数据可绘制。")
        return

    dates = df["fetch_date"]
    n = len(dates)

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.6), 5.5))
    fig.patch.set_facecolor("#fafafa")
    ax1.set_facecolor("#fafafa")

    # ── 柱状图：利好/利空/中性 ──────────────────────────
    bar_width = 0.55
    x = np.arange(n)

    bars_pos = ax1.bar(x, df["positive"], bar_width,
                       color="#2ecc71", alpha=0.85, label="利好", zorder=2)
    bars_neg = ax1.bar(x, df["negative"], bar_width,
                       bottom=df["neutral"] + df["positive"],
                       color="#e74c3c", alpha=0.85, label="利空", zorder=2)
    bars_neu = ax1.bar(x, df["neutral"], bar_width,
                       bottom=df["positive"],
                       color="#bdc3c7", alpha=0.70, label="中性", zorder=2)

    # 柱上显示总数
    for i, total in enumerate(df["total"]):
        ax1.text(i, df["positive"].iloc[i] + df["neutral"].iloc[i] + df["negative"].iloc[i] + 1.5,
                 str(total), ha="center", fontsize=7.5, color="#555")

    ax1.set_ylabel("新闻数量", fontsize=11, color="#333")
    ax1.set_ylim(0, max(df["total"]) * 1.25)
    ax1.set_xlim(-0.6, n - 0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.strftime("%m-%d") for d in dates], fontsize=9)
    ax1.set_xlabel("日期", fontsize=11, color="#333")
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)

    # ── 折线图：情感均值 ─────────────────────────────────
    ax2 = ax1.twinx()
    line = ax2.plot(x, df["avg_score"], color="#2980b9", linewidth=2.2,
                    marker="o", markersize=8, markerfacecolor="white",
                    markeredgewidth=2, markeredgecolor="#2980b9",
                    label="情感均值", zorder=5)

    # 数据点标注
    for i, score in enumerate(df["avg_score"]):
        offset = 0.04 if score >= 0 else -0.06
        ax2.annotate(f"{score:+.3f}", (i, score),
                     textcoords="offset points", xytext=(0, 12 if score >= 0 else -16),
                     ha="center", fontsize=8.5, color="#2980b9",
                     fontweight="bold")

    ax2.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("情感均值分数", fontsize=11, color="#2980b9")
    y_max = max(abs(df["avg_score"].min()), abs(df["avg_score"].max())) * 1.5
    y_max = max(y_max, 0.2)
    ax2.set_ylim(-y_max, y_max)
    ax2.tick_params(axis="y", colors="#2980b9")

    # 图例
    bars = [bars_pos, bars_neu, bars_neg]
    labels_bar = ["利好", "中性", "利空"]
    legend1 = ax1.legend(bars, labels_bar, loc="upper left", fontsize=9,
                         framealpha=0.85, edgecolor="#ddd")
    legend2 = ax2.legend(loc="upper right", fontsize=9,
                         framealpha=0.85, edgecolor="#ddd")
    ax1.add_artist(legend1)

    # 标题
    date_range = f"{dates.iloc[0].strftime('%Y-%m-%d')} ~ {dates.iloc[-1].strftime('%Y-%m-%d')}"
    total_news = df["total"].sum()
    overall_mean = (df["avg_score"] * df["total"]).sum() / df["total"].sum()
    ax1.set_title(
        f"每日新闻情感趋势  ({date_range})\n"
        f"共 {len(df)} 天 / {total_news} 条新闻  |  整体均分 {overall_mean:+.3f}",
        fontsize=13, fontweight="bold", color="#222", pad=15,
    )

    fig.tight_layout()

    out_path = PROJECT_ROOT / "data" / "sentiment_trend.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[已保存] {out_path}")

    if not save:
        # 只在非保存模式下尝试显示
        try:
            plt.show()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="情感趋势可视化")
    parser.add_argument("--days", type=int, default=30, help="回溯天数 (默认 30)")
    parser.add_argument("--save", action="store_true", help="保存为 PNG 而非显示")
    parser.add_argument("--db", type=str, default=None, help="数据库路径 (默认 data/financial_news.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH

    print(f"数据库: {db_path}")
    df = load_data(db_path, args.days)

    if df.empty:
        print(f"[提示] 近 {args.days} 天内无数据。")
        return

    print(f"加载 {len(df)} 天数据, 共 {df['total'].sum()} 条新闻")
    plot(df, args.days, save=args.save)


if __name__ == "__main__":
    main()
