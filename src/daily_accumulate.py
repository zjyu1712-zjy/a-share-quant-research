"""
每日新闻情感数据积累管道
用法:
  python -m src.daily_accumulate                # 默认运行
  python -m src.daily_accumulate --no-pause     # Cron 友好模式 (无交互)
  python -m src.daily_accumulate --log-file     # 同时写入日志文件

流程: RSS抓取 → DeepSeek分析 → JSONL存档 → SQLite入库
可作为定时任务直接调用，也可被其他模块 import。

Cron 修复说明 (2026-07-13):
  - 移除 input() 调用, 用 sys.stdin.isatty() 守卫
  - 添加 argparse 支持 --no-pause / --log-file
  - 所有路径使用绝对路径 (Path.resolve())
  - 添加日志文件输出 (适合 Cron 后台运行)
  - 添加 --json 模式输出机器可读结果
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# 确保项目根目录在 sys.path 中，兼容直接运行和 -m 两种方式
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rss_fetcher import RSSFetcher
from src.analysis_agent import AnalysisAgent
from src.database import init_db, save_news_batch

# ★ 使用绝对路径 (Cron 环境 $PWD 不一定是项目根目录)
DATA_DIR = PROJECT_ROOT.resolve() / "data"
LOG_DIR = PROJECT_ROOT.resolve() / "logs"


def setup_file_logging() -> None:
    """为 Cron 环境添加日志文件输出 (每日一个日志文件)。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"daily_accumulate_{today}.log"
    logger.add(
        str(log_path),
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"日志文件: {log_path}")


def run_accumulate() -> dict:
    """执行一次完整的日度积累，返回摘要字典。"""
    today = datetime.now().strftime("%Y%m%d")
    fetch_date = datetime.now().strftime("%Y-%m-%d")

    # ---- Step 1: RSS 抓取 ----
    logger.info("Step 1/3: RSS 抓取...")
    fetcher = RSSFetcher()
    raw_news = fetcher.fetch_all()
    if not raw_news:
        logger.warning("未抓到任何新闻，终止")
        return {"status": "empty", "raw_count": 0}

    raw_path = DATA_DIR / f"raw_news_{today}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for item in raw_news:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"  原始数据: {len(raw_news)} 条 → {raw_path.name}")

    # ---- Step 2: DeepSeek 分析 ----
    logger.info("Step 2/3: DeepSeek 情感分析...")
    try:
        agent = AnalysisAgent()
    except ValueError as e:
        logger.error(str(e))
        return {"status": "no_api_key", "raw_count": len(raw_news)}

    analyzed = agent.analyze_batch(raw_news)
    if not analyzed:
        logger.error("分析结果为空")
        return {"status": "analysis_failed", "raw_count": len(raw_news)}

    analyzed_path = DATA_DIR / f"analyzed_news_{today}.jsonl"
    with open(analyzed_path, "w", encoding="utf-8") as f:
        for item in analyzed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"  分析完成: {len(analyzed)} 条 → {analyzed_path.name}")

    # ---- Step 3: SQLite 入库 ----
    logger.info("Step 3/3: 写入数据库...")
    init_db()
    inserted = save_news_batch(analyzed, fetch_date=fetch_date)

    # ---- 汇总 ----
    sentiments = [
        it.get("sentiment_score")
        for it in analyzed
        if it.get("analysis_status") != "failed"
        and isinstance(it.get("sentiment_score"), (int, float))
    ]
    failed = len(analyzed) - len(sentiments)
    avg_score = sum(sentiments) / len(sentiments) if sentiments else 0.0
    pos = sum(1 for s in sentiments if s > 0.1)
    neg = sum(1 for s in sentiments if s < -0.1)
    neu = len(sentiments) - pos - neg

    summary = {
        "status": "ok",
        "fetch_date": fetch_date,
        "raw_count": len(raw_news),
        "analyzed_count": len(analyzed),
        "db_inserted": inserted,
        "avg_sentiment": round(avg_score, 4),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "failed": failed,
    }

    logger.info(
        f"===== 积累完成: {len(analyzed)} 条 | "
        f"入库 {inserted} 条 | "
        f"情感均值 {avg_score:+.3f} | "
        f"利好 {pos} / 利空 {neg} / 中性 {neu} / 失败 {failed} ====="
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="每日新闻情感积累管道 — RSS抓取 → DeepSeek分析 → SQLite入库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python -m src.daily_accumulate                    # 交互模式 (会 pause)
  python -m src.daily_accumulate --no-pause         # Cron 友好模式
  python -m src.daily_accumulate --log-file         # 同时写入日志文件
  python -m src.daily_accumulate --json             # 输出 JSON (机器可读)
  python -m src.daily_accumulate --no-pause --log-file --json  # Cron 推荐

Crontab 配置示例 (Linux):
  # 每个工作日 08:27 和 18:07 各运行一次
  27 08 * * 1-5 cd /path/to/quant_weekly_system && \\
    /path/to/python -m src.daily_accumulate --no-pause --log-file --json \\
    >> /var/log/daily_accumulate.log 2>&1

  07 18 * * 1-5 cd /path/to/quant_weekly_system && \\
    /path/to/python -m src.daily_accumulate --no-pause --log-file --json \\
    >> /var/log/daily_accumulate.log 2>&1
        """,
    )
    parser.add_argument(
        "--no-pause", action="store_true",
        help="跳过 '按 Enter 关闭' 提示 (Cron/自动化环境必须启用)",
    )
    parser.add_argument(
        "--log-file", action="store_true",
        help="同时将日志写入 data/logs/ 目录",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出结果 (机器可读)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式，仅输出错误信息",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # ── 文件日志 (Cron 推荐) ──
    if args.log_file:
        setup_file_logging()

    # ── 执行 ──
    summary = run_accumulate()

    # ── 输出 ──
    if args.json:
        # 机器可读 JSON 输出 (Cron 推荐)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.quiet:
        # 仅输出关键行
        if summary["status"] == "ok":
            print(
                f"[{summary['fetch_date']}] OK | "
                f"抓取{summary['raw_count']}条 | "
                f"入库{summary['db_inserted']}条 | "
                f"情感{summary['avg_sentiment']:+.3f} | "
                f"利好{summary['positive']}/利空{summary['negative']}/中性{summary['neutral']}"
            )
        else:
            print(f"[{summary.get('fetch_date', '?')}] FAIL: {summary['status']}")
    else:
        # 人类可读输出
        if summary["status"] == "ok":
            print(
                f"\n{'='*50}\n"
                f"  每日新闻情感积累完成 ({summary['fetch_date']})\n"
                f"{'='*50}\n"
                f"  抓取: {summary['raw_count']} 条\n"
                f"  入库: {summary['db_inserted']} 条\n"
                f"  情感均值: {summary['avg_sentiment']:+.3f}\n"
                f"  分布: 利好 {summary['positive']} / "
                f"利空 {summary['negative']} / "
                f"中性 {summary['neutral']}\n"
                f"{'='*50}"
            )
        else:
            print(f"\n积累失败: {summary['status']}")

    # ── ★ 关键修复: 仅在交互终端才暂停 ──
    if not args.no_pause and sys.stdin.isatty():
        try:
            input("\n按 Enter 关闭窗口...")
        except EOFError:
            # 双重保护: isatty() 返回 True 但仍可能 EOF (管道/重定向等边缘情况)
            pass

    sys.exit(0 if summary["status"] == "ok" else 1)
