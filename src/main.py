import sys
from datetime import datetime

from colorama import init, Fore, Style
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
init(autoreset=True)

from src.rss_fetcher import RSSFetcher
from src.analysis_agent import AnalysisAgent
from src.database import init_db, save_news_batch, get_weekly_sentiment_trend
from src.config import config as cfg

# ---------------------------------------------------------------------------
# 步骤 1：抓取
# ---------------------------------------------------------------------------

def step_fetch(min_news: int = 5) -> list[dict]:
    logger.info("===== [1/4] RSS 抓取开始 =====")
    fetcher = RSSFetcher()
    news = fetcher.fetch_all()

    if not news:
        print(Fore.RED + "\n  所有 RSS 源均抓取失败，流程终止。请检查网络连接。")
        logger.error("抓取结果为 0 条，终止流程")
        sys.exit(1)

    if len(news) < min_news:
        print(
            Fore.RED
            + f"\n  仅抓取到 {len(news)} 条新闻（低于最低阈值 {min_news}），"
            + "流程终止以避免浪费 API Token。"
        )
        logger.error(f"新闻量不足: {len(news)} < {min_news}，终止流程")
        sys.exit(1)

    saved = RSSFetcher.save_to_jsonl(news)
    print(f"  [1/4] 抓取完成: {len(news)} 条 -> {saved.name}")
    return news


# ---------------------------------------------------------------------------
# 步骤 2：分析
# ---------------------------------------------------------------------------

def step_analyze(news: list[dict]) -> list[dict]:
    logger.info("===== [2/4] DeepSeek 分析开始 =====")

    try:
        agent = AnalysisAgent()
    except ValueError as e:
        print(Fore.RED + f"\n  API 配置错误: {e}")
        logger.error(str(e))
        sys.exit(1)

    analyzed = agent.analyze_batch(news)
    if analyzed:
        saved = AnalysisAgent.save_analyzed(analyzed)
        print(f"  [2/4] 分析完成: {len(analyzed)} 条 -> {saved.name}")
    else:
        print(Fore.RED + "\n  分析结果为空，流程终止。")
        logger.error("分析结果为空")
        sys.exit(1)
    return analyzed


# ---------------------------------------------------------------------------
# 步骤 3：数据库存储
# ---------------------------------------------------------------------------

def step_save_db(analyzed: list[dict]) -> None:
    logger.info("===== [3/4] 数据库存储 =====")
    fetch_date = datetime.now().strftime("%Y-%m-%d")
    inserted = save_news_batch(analyzed, fetch_date)
    print(f"  [3/4] 数据库写入: 新增 {inserted} 条")


# ---------------------------------------------------------------------------
# 步骤 4：早报摘要
# ---------------------------------------------------------------------------

def step_summary(analyzed: list[dict]) -> None:
    logger.info("===== [4/4] 早报摘要 =====")

    if not analyzed:
        print(Fore.RED + "无分析数据，请先运行抓取和分析步骤")
        return

    today_str = datetime.now().strftime("%Y年%m月%d日")
    total = len(analyzed)

    scored = [
        item for item in analyzed
        if item.get("analysis_status") != "failed"
        and isinstance(item.get("sentiment_score"), (int, float))
    ]
    failed_count = total - len(scored)

    pos_items = [item for item in scored if item["sentiment_score"] > 0.1]
    neg_items = [item for item in scored if item["sentiment_score"] < -0.1]
    neutral_items = [
        item for item in scored if -0.1 <= item["sentiment_score"] <= 0.1
    ]

    pos_count = len(pos_items)
    neg_count = len(neg_items)
    neutral_count = len(neutral_items)
    avg_score = sum(s["sentiment_score"] for s in scored) / len(scored) if scored else 0.0

    pos_items.sort(key=lambda x: x["sentiment_score"], reverse=True)
    neg_items.sort(key=lambda x: x["sentiment_score"])

    def _format_card(i: int, item: dict, score_color: str) -> None:
        score = item.get("sentiment_score", 0)
        title = item.get("title", "无标题")
        tags = item.get("tags", [])
        stocks = item.get("mentioned_stocks", [])
        reasoning = item.get("reasoning", "")

        sign = "+" if score > 0 else ""
        print(
            f"  {score_color}{i}. [{sign}{score:.2f}]  {title}{Style.RESET_ALL}"
        )
        if tags:
            print(f"     {Fore.CYAN}标签{Style.RESET_ALL}  {', '.join(tags)}")
        if stocks:
            print(f"     {Fore.MAGENTA}标的{Style.RESET_ALL}  {', '.join(stocks)}")
        if reasoning:
            brief = reasoning if len(reasoning) <= 80 else reasoning[:77] + "..."
            print(f"     {Fore.WHITE}{brief}{Style.RESET_ALL}")

    # ==================================================================
    W = 72
    Y = Fore.YELLOW + Style.BRIGHT
    G = Fore.GREEN + Style.BRIGHT
    R = Fore.RED + Style.BRIGHT

    print()
    print(Y + "=" * W + Style.RESET_ALL)
    print(Y + f"  {today_str}  每日财经早报".center(W + 6) + Style.RESET_ALL)
    print(Y + "=" * W + Style.RESET_ALL)

    avg_color = Fore.GREEN if avg_score > 0 else Fore.RED if avg_score < 0 else Fore.WHITE
    print(
        f"  共 {total} 条  |  "
        f"{Fore.GREEN}利好 {pos_count}{Style.RESET_ALL}  |  "
        f"{Fore.RED}利空 {neg_count}{Style.RESET_ALL}  |  "
        f"中性 {neutral_count}  |  "
        f"分析失败 {failed_count}  |  "
        f"{avg_color}平均分 {avg_score:+.2f}{Style.RESET_ALL}"
    )
    print(Y + "-" * W + Style.RESET_ALL)

    # --- 利好 TOP 3 ---
    print(f"\n  {G}>>> 利好 TOP 3{Style.RESET_ALL}")
    if pos_items:
        for i, item in enumerate(pos_items[:3], 1):
            _format_card(i, item, Fore.GREEN)
    else:
        print("  (无利好新闻)")

    # --- 利空 TOP 3 ---
    print(f"\n  {R}>>> 利空 TOP 3{Style.RESET_ALL}")
    if neg_items:
        for i, item in enumerate(neg_items[:3], 1):
            _format_card(i, item, Fore.RED)
    else:
        print("  (无利空新闻)")

    print()
    print(Y + "=" * W + Style.RESET_ALL)
    print()


# ======================================================================
# 主入口
# ======================================================================

if __name__ == "__main__":
    # --- 回测模式 ---
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        sys.argv.pop(1)  # remove "backtest" so argparse doesn't choke
        from src.backtest_engine import main as backtest_main
        backtest_main()
        sys.exit(0)

    # --- 回测验证模式 ---
    if len(sys.argv) > 1 and sys.argv[1] == "backtest-validate":
        sys.argv.pop(1)
        from src.backtest_validate import main as validate_main
        validate_main()
        sys.exit(0)

    # --- 网格搜索模式 ---
    if len(sys.argv) > 1 and sys.argv[1] == "grid-search":
        sys.argv.pop(1)
        from src.grid_search_1M import main as grid_main
        grid_main()
        sys.exit(0)

    mode_tag = f"{cfg.timeframe} 模式 | {cfg.rebalance_label}"
    logger.info(f"========== QuantWeeklySystem 启动 ({mode_tag}) ==========")

    print(Fore.CYAN + f"  当前模式: {cfg.timeframe} ({cfg.rebalance_label})" + Style.RESET_ALL)
    print(Fore.CYAN + f"  策略: {cfg.strategy_name} v{cfg.version}" + Style.RESET_ALL)
    init_db()

    try:
        news = step_fetch()
        analyzed = step_analyze(news)
        step_save_db(analyzed)
        step_summary(analyzed)

        # WEEKLY 模式额外展示周度情感走势
        if cfg.timeframe == "WEEKLY":
            print(Fore.YELLOW + "\n  >>> 周度情感走势 (近4周, 关键词 'AI')" + Style.RESET_ALL)
            try:
                df_w = get_weekly_sentiment_trend("AI", weeks=4)
                if len(df_w) > 0:
                    print(df_w.to_string(index=False))
                else:
                    print("  (数据不足，需积累更多天数据)")
            except Exception:
                print("  (周度聚合查询失败，请检查数据库)")
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n  用户中断，流程退出。")
        sys.exit(0)

    logger.info(f"========== 工作流结束 ({mode_tag}) =========")

    if sys.stdin.isatty():
        input(Fore.CYAN + "\n  按 Enter 键退出..." + Style.RESET_ALL)
