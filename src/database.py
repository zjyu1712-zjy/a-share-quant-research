import hashlib
import math
import numbers
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "financial_news.db"


def _sentiment_label(score: float) -> str:
    if score > 0.1:
        return "利好"
    elif score < -0.1:
        return "利空"
    return "中性"


def _compute_content_hash(item: dict) -> str:
    """计算新闻内容哈希（URL + 摘要），用于检测 RSS 内容是否更新。

    只用 RSS 原始字段，不用 DeepSeek 分析结果（title/reasoning 可能被分析覆盖或波动）。
    """
    url = (item.get("link") or item.get("url") or "").strip()
    summary = (item.get("summary") or "").strip()
    digest = hashlib.sha256(f"{url}|{summary}".encode("utf-8")).hexdigest()
    return digest


def init_db(db_path: Optional[Path] = None) -> Path:
    """初始化数据库，创建 news_archive 表。返回数据库文件路径。"""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_archive (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            publish_date TEXT,
            fetch_date   TEXT    NOT NULL,
            title        TEXT    NOT NULL,
            source       TEXT,
            sentiment_score REAL DEFAULT 0.0,
            sentiment_label TEXT DEFAULT '中性',
            tags         TEXT,
            mentioned_stocks TEXT,
            summary      TEXT,
            url          TEXT,
            reasoning    TEXT
        )
        """
    )
    # 兼容旧表：补上可能缺失的列
    for col, col_type in [("mentioned_stocks", "TEXT"), ("content_hash", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE news_archive ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON news_archive(url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fetch_date ON news_archive(fetch_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_publish_date ON news_archive(publish_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title_source ON news_archive(title, source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON news_archive(source)")
    conn.commit()
    conn.close()
    logger.info(f"数据库已就绪: {path}")
    return path


def save_news_batch(
    news_list: list[dict],
    fetch_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """批量保存分析后的新闻到数据库，按 URL/标题去重，内容有变则更新。"""
    if not news_list:
        return 0

    path = db_path or DB_PATH
    if not path.exists():
        init_db(path)

    if fetch_date is None:
        fetch_date = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(path))
    inserted = 0
    updated = 0
    skipped = 0
    rejected = 0

    for item in news_list:
        score = item.get("sentiment_score")
        if (
            item.get("analysis_status") == "failed"
            or not isinstance(score, numbers.Real)
            or not math.isfinite(float(score))
        ):
            rejected += 1
            continue

        url = (item.get("link") or item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        if not title:
            continue

        new_hash = _compute_content_hash(item)

        # ---- Locate existing record by URL, then by title+source ----
        existing_id = None
        old_hash = None

        if url:
            cur = conn.execute(
                "SELECT id, content_hash FROM news_archive WHERE url = ? LIMIT 1",
                (url,),
            )
            row = cur.fetchone()
            if row:
                existing_id, old_hash = row

        if existing_id is None:
            source = item.get("source", "")
            cur = conn.execute(
                "SELECT id, content_hash FROM news_archive WHERE title = ? AND source = ? LIMIT 1",
                (title, source),
            )
            row = cur.fetchone()
            if row:
                existing_id, old_hash = row

        # ---- True duplicate: same content hash → skip ----
        if existing_id is not None and old_hash == new_hash:
            skipped += 1
            continue

        # ---- Build field values once ----
        score = float(score)
        label = _sentiment_label(score)
        tags = item.get("tags", [])
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        stocks = item.get("mentioned_stocks", [])
        stocks_str = ", ".join(stocks) if isinstance(stocks, list) else str(stocks)
        summary = item.get("summary", "") or item.get("reasoning", "")
        publish_date = item.get("published", "")

        if existing_id is not None:
            # Content changed → update existing record
            conn.execute(
                """
                UPDATE news_archive
                   SET publish_date = ?, fetch_date = ?, title = ?,
                       sentiment_score = ?, sentiment_label = ?,
                       tags = ?, mentioned_stocks = ?,
                       summary = ?, url = ?, reasoning = ?,
                       content_hash = ?
                 WHERE id = ?
                """,
                (
                    publish_date, fetch_date, title,
                    score, label,
                    tags_str, stocks_str,
                    summary, url, item.get("reasoning", ""),
                    new_hash,
                    existing_id,
                ),
            )
            updated += 1
        else:
            # New entry → insert
            conn.execute(
                """
                INSERT INTO news_archive
                    (publish_date, fetch_date, title, source,
                     sentiment_score, sentiment_label, tags, mentioned_stocks,
                     summary, url, reasoning, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    publish_date, fetch_date, title,
                    item.get("source", ""),
                    score, label,
                    tags_str, stocks_str,
                    summary, url,
                    item.get("reasoning", ""),
                    new_hash,
                ),
            )
            inserted += 1

    conn.commit()
    conn.close()
    logger.info(
        f"数据库写入完成: 新增 {inserted} 条, 更新 {updated} 条, "
        f"跳过重复 {skipped} 条, 拒绝未完成分析 {rejected} 条 -> {path}"
    )
    return inserted + updated


# ======================================================================
# 查询函数
# ======================================================================

def get_history(
    start_date: str,
    end_date: str,
    keyword: Optional[str] = None,
    db_path: Optional[Path] = None,
):
    """按日期范围查询新闻，可选关键词过滤。

    Args:
        start_date: 起始日期，格式 "YYYY-MM-DD"
        end_date:   结束日期，格式 "YYYY-MM-DD"
        keyword:    可选关键词，在标题/标签/股票/分析理由中模糊匹配
        db_path:    数据库路径

    Returns:
        pandas DataFrame，包含符合条件的所有新闻记录
    """
    import pandas as pd

    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))

    if keyword:
        like = f"%{keyword}%"
        sql = """
            SELECT id, publish_date, fetch_date, title, source,
                   sentiment_score, sentiment_label, tags, mentioned_stocks,
                   summary, url, reasoning
            FROM news_archive
            WHERE fetch_date >= ? AND fetch_date <= ?
              AND sentiment_score IS NOT NULL
              AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
              AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
              AND (title LIKE ? OR tags LIKE ? OR mentioned_stocks LIKE ?
                   OR reasoning LIKE ? OR summary LIKE ?)
            ORDER BY fetch_date DESC, sentiment_score DESC
        """
        params = (start_date, end_date, like, like, like, like, like)
    else:
        sql = """
            SELECT id, publish_date, fetch_date, title, source,
                   sentiment_score, sentiment_label, tags, mentioned_stocks,
                   summary, url, reasoning
            FROM news_archive
            WHERE fetch_date >= ? AND fetch_date <= ?
              AND sentiment_score IS NOT NULL
              AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
              AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
            ORDER BY fetch_date DESC, sentiment_score DESC
        """
        params = (start_date, end_date)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    logger.info(f"查询完成: {len(df)} 条记录 ({start_date} ~ {end_date})")
    return df


def get_sentiment_trend(
    keyword: str,
    days: int = 30,
    db_path: Optional[Path] = None,
):
    """查询某关键词在过去 N 天的情感走势。

    Args:
        keyword: 关键词（公司名如"英伟达"，或概念如"AI"）
        days:    回溯天数，默认 30 天
        db_path: 数据库路径

    Returns:
        pandas DataFrame，列: fetch_date, news_count, avg_score,
        positive_count, negative_count, neutral_count
    """
    import pandas as pd

    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))

    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    like = f"%{keyword}%"

    sql = """
        SELECT fetch_date,
               COUNT(*)          AS news_count,
               AVG(sentiment_score) AS avg_score,
               SUM(CASE WHEN sentiment_label = '利好' THEN 1 ELSE 0 END) AS positive_count,
               SUM(CASE WHEN sentiment_label = '利空' THEN 1 ELSE 0 END) AS negative_count,
               SUM(CASE WHEN sentiment_label = '中性' THEN 1 ELSE 0 END) AS neutral_count
        FROM news_archive
        WHERE fetch_date >= ?
          AND sentiment_score IS NOT NULL
          AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
          AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
          AND (title LIKE ? OR tags LIKE ? OR mentioned_stocks LIKE ?
               OR reasoning LIKE ? OR summary LIKE ?)
        GROUP BY fetch_date
        ORDER BY fetch_date ASC
    """
    params = (start_date, like, like, like, like, like)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    logger.info(
        f"情感走势查询完成: 关键词='{keyword}', {days}天, {len(df)} 个日期有数据"
    )
    return df


# ======================================================================
# 周度聚合查询（多周期适配层）
# ======================================================================

def get_weekly_aggregation(
    keyword: str,
    start_date: str,
    end_date: str,
    db_path: Optional[Path] = None,
):
    """将日线情感数据聚合为周线。

    每周取该周所有匹配新闻的情感分均值、总数、利好/利空数。

    Args:
        keyword:    关键词
        start_date: 起始日期 "YYYY-MM-DD"
        end_date:   结束日期 "YYYY-MM-DD"
        db_path:    数据库路径

    Returns:
        DataFrame，列: week_start, news_count, avg_score,
        positive_count, negative_count, neutral_count
    """
    import pandas as pd

    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    like = f"%{keyword}%"

    sql = """
        SELECT
            DATE(fetch_date, '-' || ((strftime('%w', fetch_date) + 6) % 7) || ' days')
                AS week_start,
            COUNT(*)            AS news_count,
            AVG(sentiment_score) AS avg_score,
            SUM(CASE WHEN sentiment_label = '利好' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN sentiment_label = '利空' THEN 1 ELSE 0 END) AS negative_count,
            SUM(CASE WHEN sentiment_label = '中性' THEN 1 ELSE 0 END) AS neutral_count
        FROM news_archive
        WHERE fetch_date >= ? AND fetch_date <= ?
          AND sentiment_score IS NOT NULL
          AND COALESCE(reasoning, '') NOT LIKE '%API 调用失败%'
          AND COALESCE(reasoning, '') NOT LIKE '%未完成分析%'
          AND (title LIKE ? OR tags LIKE ? OR mentioned_stocks LIKE ?
               OR reasoning LIKE ? OR summary LIKE ?)
        GROUP BY week_start
        ORDER BY week_start ASC
    """
    params = (start_date, end_date, like, like, like, like, like)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    logger.info(
        f"周度聚合查询: 关键词='{keyword}', {start_date}~{end_date}, "
        f"共 {len(df)} 周"
    )
    return df


def get_weekly_sentiment_trend(
    keyword: str,
    weeks: int = 52,
    db_path: Optional[Path] = None,
):
    """查询某关键词在过去 N 周的周度情感走势。

    Args:
        keyword: 关键词
        weeks:   回溯周数，默认 52 周
        db_path: 数据库路径

    Returns:
        DataFrame，列: week_start, news_count, avg_score,
        positive_count, negative_count, neutral_count
    """
    start_date = (datetime.now() - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    return get_weekly_aggregation(keyword, start_date, end_date, db_path)


# ======================================================================
# Demo / 直接运行
# ======================================================================

if __name__ == "__main__":
    import pandas as pd

    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 60)

    # 确保数据库就绪
    init_db()

    # --- Demo 1：查询过去 7 天关于 "AI" 的新闻 ---
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*80}")
    print(f"  查询: 过去7天关于 'AI' 的新闻 ({start} ~ {end})")
    print(f"{'='*80}\n")

    df_ai = get_history(start, end, keyword="AI")
    if len(df_ai) == 0:
        print("  无匹配结果。请先运行 main.py 采集数据。")
    else:
        print(df_ai[["fetch_date", "sentiment_score", "sentiment_label", "title", "source"]].to_string(index=False))

    # --- Demo 2：AI 情感走势 ---
    print(f"\n{'='*80}")
    print(f"  情感走势: 'AI' 近30天")
    print(f"{'='*80}\n")

    df_trend = get_sentiment_trend("AI", days=30)
    if len(df_trend) == 0:
        print("  无走势数据（数据库不足30天历史）。")
    else:
        print(df_trend.to_string(index=False))
        print()
        avg_all = df_trend["avg_score"].mean()
        total_news = df_trend["news_count"].sum()
        print(f"  汇总: {len(df_trend)} 个交易日, 共 {total_news} 条新闻, 整体均分 {avg_all:+.3f}")

    # --- Demo 3：周度聚合 ---
    print(f"\n{'='*80}")
    print(f"  周度聚合: 'AI' 近4周")
    print(f"{'='*80}\n")

    df_weekly = get_weekly_aggregation("AI", start, end)
    if len(df_weekly) == 0:
        print("  无周度数据。")
    else:
        print(df_weekly.to_string(index=False))
