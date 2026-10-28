"""
一次性索引迁移脚本 — 为 news_archive 表添加缺失的查询索引。

用法: python -m src.migrate_indexes

添加的索引:
  idx_title_source  — (title, source) 复合索引，加速 save_news_batch 去重查询
  idx_source        — source 单列索引，加速按来源过滤
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "financial_news.db"

MIGRATIONS = [
    (
        "idx_title_source",
        "CREATE INDEX IF NOT EXISTS idx_title_source ON news_archive(title, source)",
    ),
    (
        "idx_source",
        "CREATE INDEX IF NOT EXISTS idx_source ON news_archive(source)",
    ),
]


def run():
    if not DB_PATH.exists():
        print(f"[错误] 数据库不存在: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # 检查已有索引
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='news_archive'"
        )
    }
    print(f"现有索引 ({len(existing)}): {', '.join(sorted(existing))}")

    added = 0
    skipped = 0
    for name, sql in MIGRATIONS:
        if name in existing:
            print(f"  [跳过] {name} — 已存在")
            skipped += 1
        else:
            conn.execute(sql)
            print(f"  [创建] {name}")
            added += 1

    conn.commit()

    # 验证
    final = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='news_archive'"
        )
    }
    print(f"\n迁移后索引 ({len(final)}): {', '.join(sorted(final))}")

    # 验证索引是否被 SQLite 使用
    print("\n查询计划验证:")
    for test_sql, desc in [
        (
            "EXPLAIN QUERY PLAN SELECT 1 FROM news_archive WHERE title = 'x' AND source = 'y' LIMIT 1",
            "title+source 去重查询",
        ),
        (
            "EXPLAIN QUERY PLAN SELECT 1 FROM news_archive WHERE url = 'http://x' LIMIT 1",
            "URL 去重查询",
        ),
        (
            "EXPLAIN QUERY PLAN SELECT * FROM news_archive WHERE fetch_date >= '2026-05-01' AND fetch_date <= '2026-05-31'",
            "日期范围查询",
        ),
    ]:
        plan = conn.execute(test_sql).fetchall()
        plan_str = " | ".join(str(r) for r in plan)
        using_index = "USING INDEX" in plan_str or "COVERING INDEX" in plan_str or "SEARCH" in plan_str
        status = "[OK] 使用索引" if using_index else "[!!] 全表扫描"
        print(f"  {desc}: {status}")
        if not using_index:
            print(f"    计划: {plan_str}")

    conn.close()
    print(f"\n迁移完成: 新增 {added} 个索引, 跳过 {skipped} 个")


if __name__ == "__main__":
    run()
