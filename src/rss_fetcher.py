import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import feedparser
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOGS_DIR / "rss_fetcher_{time:YYYYMMDD}.log",
    rotation="1 day",
    retention="30 days",
    encoding="utf-8",
    level="INFO",
)

RSS_FEEDS: list[dict] = [
    {
        "name": "CNBC Top News",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "lang": "en",
    },
    {
        "name": "36氪",
        "url": "https://36kr.com/feed",
        "lang": "zh",
    },
    {
        "name": "雪球热门",
        "url": "https://xueqiu.com/hots/topic/rss",
        "lang": "zh",
    },
]


class RSSFetcher:
    """抓取多源财经 RSS，去重后按日期保存为 JSONL。"""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        self.seen_ids: set[str] = set()

    # ------------------------------------------------------------------
    # 抓取
    # ------------------------------------------------------------------

    def _fetch_single(self, feed_info: dict) -> list[dict]:
        """抓取单个 RSS 源，返回标准化条目列表。"""
        name = feed_info["name"]
        url = feed_info["url"]
        results: list[dict] = []

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()

            raw_content = resp.content
            content = self._ensure_decode(raw_content)

            parsed = feedparser.parse(content)
            if parsed.bozo and not parsed.entries:
                logger.warning(f"[{name}] 解析异常 (bozo): {parsed.bozo_exception}")
                return results

            for entry in parsed.entries:
                item = self._normalize_entry(entry, name)
                if item is None:
                    continue
                dedup_key = item["link"] or item.get("id", "")
                if dedup_key and dedup_key in self.seen_ids:
                    continue
                if dedup_key:
                    self.seen_ids.add(dedup_key)
                results.append(item)

            logger.info(f"[{name}] 抓取 {len(results)} 条")
        except requests.RequestException as e:
            logger.error(f"[{name}] 网络请求失败: {e}")
        except Exception:
            logger.exception(f"[{name}] 未知错误")

        return results

    # ------------------------------------------------------------------
    # 编码处理
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_decode(raw: bytes) -> str:
        """尝试将原始字节解码为 UTF-8，失败则回退到常见中文编码。"""
        for enc in ("utf-8", "gb2312", "gbk", "gb18030", "latin-1"):
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 标准化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_entry(entry, source_name: str) -> Optional[dict]:
        link = entry.get("link", "")
        title = entry.get("title", "").strip()
        if not title:
            return None

        published_raw = entry.get("published", "") or entry.get("updated", "")
        published = RSSFetcher._parse_datetime(published_raw)

        summary = entry.get("summary", "") or entry.get("description", "")
        summary = RSSFetcher._strip_html(summary)

        return {
            "title": title,
            "link": link,
            "published": published,
            "published_raw": published_raw,
            "summary": summary,
            "source": source_name,
            "id": entry.get("id", link),
        }

    @staticmethod
    def _parse_datetime(raw: str) -> Optional[str]:
        """尝试将原始时间字符串转为 ISO-8601 UTC 格式。"""
        if not raw:
            return None
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw)
            return dt.astimezone(timezone.utc).isoformat()
        except (ValueError, TypeError):
            pass
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
        return raw

    @staticmethod
    def _strip_html(text: str) -> str:
        """移除 HTML 标签，保留纯文本。"""
        import re
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def fetch_all(self) -> list[dict]:
        """遍历全部 RSS 源，返回去重后的新闻列表。"""
        self.seen_ids.clear()
        all_news: list[dict] = []

        for feed_info in RSS_FEEDS:
            entries = self._fetch_single(feed_info)
            all_news.extend(entries)

        logger.info(f"本轮共抓取 {len(all_news)} 条新闻（去重后）")
        return all_news

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    @staticmethod
    def save_to_jsonl(news_list: list[dict]) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        filepath = DATA_DIR / f"raw_news_{today}.jsonl"

        with open(filepath, "w", encoding="utf-8") as f:
            for item in news_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"已保存 {len(news_list)} 条至 {filepath}")
        return filepath


# ======================================================================
# 主程序入口
# ======================================================================

if __name__ == "__main__":
    logger.info("===== RSS 抓取开始 =====")
    fetcher = RSSFetcher()
    news = fetcher.fetch_all()
    if news:
        saved_path = RSSFetcher.save_to_jsonl(news)
        logger.info(f"===== 今日共抓取 {len(news)} 条新闻，文件: {saved_path} =====")
        print(f"今日共抓取 {len(news)} 条新闻，已保存至: {saved_path}")
    else:
        logger.warning("===== 未抓到任何新闻 =====")
        print("今日未抓到任何新闻，请检查网络或 RSS 源是否可用。")
