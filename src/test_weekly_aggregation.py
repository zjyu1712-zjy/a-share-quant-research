"""
临时验证脚本 — 用少量真实数据模拟 30 天积累，测试周线情感因子聚合。

1. 从现有 SQLite DB 读取所有新闻
2. 复制到临时 DB，按 30 个日期均匀分布（加随机噪声模拟真实积累）
3. 调用 build_weekly_sentiment_factors() 输出周线因子
4. 验证输出正确性后清理临时 DB
"""

import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DB = PROJECT_ROOT / "data" / "financial_news.db"
TEMP_DB = PROJECT_ROOT / "data" / "_test_30day_temp.db"

sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from src.database import init_db
from src.dual_factor import get_stock_mapping, build_weekly_sentiment_factors


def replicate_30_days() -> int:
    """复制真实数据到临时 DB，模拟 30 天积累。返回复制的总条数。"""
    if not REAL_DB.exists():
        logger.error(f"真实数据库不存在: {REAL_DB}")
        return 0

    # 1) 读取所有现有新闻
    conn = sqlite3.connect(str(REAL_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT title, source, sentiment_score, tags, "
        "mentioned_stocks, summary, url, reasoning "
        "FROM news_archive ORDER BY fetch_date"
    ).fetchall()
    conn.close()

    if not rows:
        logger.error("数据库为空")
        return 0

    logger.info(f"从真实库读取 {len(rows)} 条新闻")

    # 2) 30 个目标日期 (2026-05-01 ~ 2026-05-30)
    base = datetime(2026, 5, 1)
    target_dates = [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]

    # 3) 准备临时 DB
    if TEMP_DB.exists():
        TEMP_DB.unlink()
    init_db(TEMP_DB)

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    conn = sqlite3.connect(str(TEMP_DB))
    total = 0

    for date_idx, fetch_date in enumerate(target_dates):
        # 每目标日随机采样 70-80 条不重复标题的新闻
        n_samples = rng.randint(70, 81)
        sampled = rng.sample(rows, k=min(n_samples, len(rows)))

        seen_titles = set()
        day_count = 0
        for row in sampled:
            title = row["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            # 给 sentiment_score 加 ±0.08 随机噪声
            orig_score = row["sentiment_score"] or 0.0
            noise = float(np_rng.uniform(-0.08, 0.08))
            score = max(-1.0, min(1.0, round(orig_score + noise, 4)))

            label = "利好" if score > 0.1 else ("利空" if score < -0.1 else "中性")
            tags = (row["tags"] or "")
            stocks = (row["mentioned_stocks"] or "")
            summary = (row["summary"] or "")

            conn.execute(
                """INSERT INTO news_archive
                   (publish_date, fetch_date, title, source,
                    sentiment_score, sentiment_label, tags, mentioned_stocks,
                    summary, url, reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fetch_date, fetch_date, title,
                    row["source"] or "", score, label,
                    tags, stocks,
                    summary, (row["url"] or ""),
                    row["reasoning"] or "",
                ),
            )
            day_count += 1

        conn.commit()
        total += day_count

        if (date_idx + 1) % 10 == 0:
            logger.info(f"  已写入 {date_idx + 1}/30 天 ({total} 条累计)")

    conn.close()

    # 验证
    vconn = sqlite3.connect(str(TEMP_DB))
    days = vconn.execute("SELECT COUNT(DISTINCT fetch_date) FROM news_archive").fetchone()[0]
    count = vconn.execute("SELECT COUNT(*) FROM news_archive").fetchone()[0]
    vconn.close()

    logger.info(f"临时库就绪: {count} 条, {days} 天 → {TEMP_DB}")
    return total


def main():
    logger.info("=" * 60)
    logger.info("  周线情感因子聚合 — 30天模拟验证")
    logger.info("=" * 60)

    # --- Step 1: 生成 30 天模拟数据 ---
    logger.info("Step 1/4: 生成 30 天模拟数据...")
    total = replicate_30_days()
    if total == 0:
        logger.error("模拟数据生成失败")
        return
    logger.info(f"  共写入 {total} 条记录")

    # --- Step 2: 获取股票映射 ---
    logger.info("Step 2/4: 加载股票池...")
    _, name_to_code, _ = get_stock_mapping()
    logger.info(f"  股票池: {len(name_to_code)} 只")

    # --- Step 3: 构建周线因子 ---
    logger.info("Step 3/4: 调用 build_weekly_sentiment_factors()...")
    df = build_weekly_sentiment_factors(
        name_to_code,
        start_date="2026-05-01",
        end_date="2026-05-30",
        db_path=TEMP_DB,
    )

    # --- Step 4: 验证输出 ---
    logger.info("Step 4/4: 验证输出")
    print("\n" + "=" * 90)
    print("  周线情感因子输出")
    print("=" * 90)

    code_to_name = {v: k for k, v in name_to_code.items()}

    if df.empty:
        print("\n  [警告] 输出为空！可能原因:")
        print("    - 股票池中的股票名与新闻中提到的公司名不匹配")
        print("    - mentioned_stocks 字段中主要是英文公司名（Nvidia, Tesla 等）")
        print("    - 中文新闻较少，且中文股票名未必出现在财经新闻中")
        print()
        print("  补救: 用关键词（如 'AI', '新能源', '半导体'）测试通用情感序列...")

        # Fallback: 用关键词测试
        from src.database import get_weekly_sentiment_trend
        for kw in ["AI", "新能源", "半导体", "汽车"]:
            try:
                trend = get_weekly_sentiment_trend(kw, weeks=52, db_path=TEMP_DB)
                if not trend.empty:
                    print(f"\n  '{kw}' 周度情感走势 ({len(trend)} 周):")
                    for _, row in trend.iterrows():
                        print(
                            f"    {row['week_start']}: "
                            f"{int(row['news_count']):>4}条, "
                            f"均分{row['avg_score']:>+.3f}, "
                            f"利好{int(row['positive_count']):>3}/"
                            f"利空{int(row['negative_count']):>3}/"
                            f"中性{int(row['neutral_count']):>3}"
                        )
            except Exception as e:
                logger.warning(f"关键词 '{kw}' 查询失败: {e}")
    else:
        n_stocks = df["code"].nunique()
        n_weeks = df["week_start"].nunique()
        print(f"\n  总记录: {len(df)} 条")
        print(f"  覆盖股票: {n_stocks} / {len(name_to_code)} 只")
        print(f"  覆盖周数: {n_weeks} 周")

        # 每只股票的周度汇总
        print(f"\n  {'代码':<8} {'名称':<10} {'覆盖周':>6} {'周均分均值':>10} {'周均分波动':>10} {'利好占比均值':>12}")
        print(f"  {'-'*56}")

        for code in sorted(df["code"].unique()):
            sdf = df[df["code"] == code]
            name = code_to_name.get(code, "?")
            print(
                f"  {code:<8} {name:<10} "
                f"{len(sdf):>6} "
                f"{sdf['week_mean'].mean():>+10.4f} "
                f"{sdf['week_std'].mean():>10.4f} "
                f"{sdf['positive_ratio'].mean():>12.4f}"
            )

        # 全量统计
        print(f"\n  {'─'*56}")
        print(f"  全量统计:")
        print(f"    周均分范围:    {df['week_mean'].min():+.4f} ~ {df['week_mean'].max():+.4f}")
        print(f"    周标准差范围:   {df['week_std'].min():.4f} ~ {df['week_std'].max():.4f}")
        print(f"    利好占比范围:   {df['positive_ratio'].min():.4f} ~ {df['positive_ratio'].max():.4f}")
        print(f"    周新闻数范围:   {df['news_count'].min():.0f} ~ {df['news_count'].max():.0f}")
        avg_news_per_week = df["news_count"].sum() / n_weeks if n_weeks > 0 else 0
        print(f"    周均新闻数:     {avg_news_per_week:.1f} 条/周")

        # 按周展示
        print(f"\n  按周汇总:")
        print(f"  {'周起始':<12} {'股票数':>6} {'周均分':>8} {'周新闻总数':>10}")
        print(f"  {'-'*40}")
        weekly_summary = df.groupby("week_start").agg(
            stock_count=("code", "nunique"),
            avg_score=("week_mean", "mean"),
            total_news=("news_count", "sum"),
        )
        for week, row in weekly_summary.iterrows():
            print(
                f"  {week.strftime('%Y-%m-%d'):<12} "
                f"{int(row['stock_count']):>6} "
                f"{row['avg_score']:>+8.4f} "
                f"{int(row['total_news']):>10}"
            )

        # 验证检查
        print(f"\n  {'─'*56}")
        print(f"  验证检查:")
        checks = []
        checks.append(("C1: 周数在 4-6 之间", 4 <= n_weeks <= 6))
        checks.append(("C2: week_mean 在 [-1, 1]", bool(df["week_mean"].between(-1, 1).all())))
        checks.append(("C3: week_std >= 0", bool((df["week_std"] >= 0).all())))
        checks.append(("C4: positive_ratio 在 [0, 1]", bool(df["positive_ratio"].between(0, 1).all())))
        checks.append(("C5: news_count > 0", bool((df["news_count"] > 0).all())))

        all_pass = True
        for name, passed in checks:
            status = "[PASS]" if passed else "[FAIL]"
            if not passed:
                all_pass = False
            print(f"    {status} {name}")

        if all_pass:
            print(f"\n  >>> 全部验证通过，周线聚合逻辑正确 <<<")
        else:
            print(f"\n  >>> 部分检查未通过，请排查 <<<")

    print(f"\n{'='*90}\n")

    # --- 清理 ---
    if TEMP_DB.exists():
        TEMP_DB.unlink()
        logger.info(f"临时库已清理: {TEMP_DB}")


if __name__ == "__main__":
    main()
