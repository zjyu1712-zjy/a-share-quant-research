import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = """\
你是一位资深宏观与行业分析师，拥有20年金融市场研究经验。你的任务是对给定的财经新闻列表进行专业分析。

## 分析要求

1. **情感打分 (sentiment_score)**：对每条新闻给出 -1.0 到 +1.0 的情感分数。
   - -1.0 ~ -0.3：利空（政策收紧、业绩暴雷、地缘冲突升级、监管处罚等）
   - -0.3 ~ +0.3：中性（常规报道、行业动态、人事变动等）
   - +0.3 ~ +1.0：利好（政策支持、业绩超预期、技术突破、并购整合等）

2. **标签提取 (tags)**：识别新闻涉及的具体行业和概念，从以下维度提取 1-5 个标签：
   - 行业：半导体、新能源、AI、医药、消费、金融、地产、汽车、互联网、军工、农业、能源、电信、制造
   - 概念：并购、涨价、降价、地缘冲突、政策、IPO、财报、裁员、技术突破、制裁、供应链、关税、加息、降息

3. **关联标的 (mentioned_stocks)**：提取新闻中明确提及的上市公司名称（如"英伟达"、"腾讯"、"Meta"）。只提取具体公司名，不要提取行业统称。没有明确提及则返回空数组。

4. **分析理由 (reasoning)**：用 1-2 句中文简述打分依据，点出关键变量或潜在影响。

## 输出格式
严格返回 JSON，不要包含 markdown 代码块标记，直接输出如下结构：
{"results":[{"index":<新闻编号>,"title":"<原标题>","sentiment_score":<float>,"tags":["<标签1>",...],"mentioned_stocks":["<公司名>",...],"reasoning":"<简短分析>"}]}
"""

BATCH_PROMPT_TEMPLATE = """请对以下 {count} 条财经新闻逐一进行分析，返回 JSON。

{news_items}"""


class AnalysisAgent:
    """使用 DeepSeek 对财经新闻进行批量情感分析、标签提取和标的识别。"""

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置，请在 .env 文件中配置: DEEPSEEK_API_KEY=sk-xxx"
            )
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # 模型名称可通过 .env 覆盖；使用前需向服务商确认可用模型。
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

    # ------------------------------------------------------------------
    # 读取最新原始数据
    # ------------------------------------------------------------------

    @staticmethod
    def find_latest_raw_file() -> Optional[Path]:
        files = sorted(DATA_DIR.glob("raw_news_*.jsonl"))
        if not files:
            logger.warning("data/ 目录下未找到 raw_news_*.jsonl 文件")
            return None
        return files[-1]

    @staticmethod
    def load_raw_news(filepath: Path) -> list[dict]:
        news = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    news.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info(f"从 {filepath.name} 读取 {len(news)} 条新闻")
        return news

    # ------------------------------------------------------------------
    # 构建批量分析 Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_prompt(news_list: list[dict]) -> str:
        items = []
        for i, item in enumerate(news_list, start=1):
            title = item.get("title", "")
            summary = item.get("summary", "")
            # 限制每条摘要长度，避免 token 溢出
            summary_short = summary[:300] + "..." if len(summary) > 300 else summary
            items.append(f"[{i}] 标题: {title}\n    摘要: {summary_short}")

        news_block = "\n\n".join(items)
        return BATCH_PROMPT_TEMPLATE.format(count=len(news_list), news_items=news_block)

    # ------------------------------------------------------------------
    # 调用 DeepSeek
    # ------------------------------------------------------------------

    def analyze_batch(self, news_list: list[dict], batch_size: int = 50) -> list[dict]:
        """对新闻列表批量调用 DeepSeek 进行分析，返回带分析结果的列表。"""
        results: list[dict] = []

        for start in range(0, len(news_list), batch_size):
            chunk = news_list[start:start + batch_size]
            prompt = self._build_batch_prompt(chunk)
            logger.info(f"发送分析请求: {len(chunk)} 条 (第 {start // batch_size + 1} 批)")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=8192,
                )
                raw_output = response.choices[0].message.content or ""
                parsed = self._parse_response(raw_output)

                # 严格按输入逐条对齐。缺失、越界或无有效分数的响应标记为
                # failed，不得用 0 分冒充中性新闻。
                parsed_by_index = {
                    item.get("index", 0) - 1: item
                    for item in parsed
                    if isinstance(item, dict)
                }
                success_count = 0
                for idx, raw_item in enumerate(chunk):
                    analyzed = parsed_by_index.get(idx)
                    score = (
                        analyzed.get("sentiment_score")
                        if analyzed is not None else None
                    )
                    if analyzed is not None and isinstance(score, (int, float)):
                        results.append({
                            **raw_item,
                            **analyzed,
                            "analysis_status": "success",
                        })
                        success_count += 1
                    else:
                        results.append({
                            **raw_item,
                            "sentiment_score": None,
                            "tags": [],
                            "mentioned_stocks": [],
                            "reasoning": "模型响应缺失或格式无效，未完成分析",
                            "analysis_status": "failed",
                        })

                logger.info(
                    f"成功解析 {success_count}/{len(chunk)} 条分析结果"
                )
            except Exception:
                logger.exception("DeepSeek API 调用失败")
                # 失败记录保留在 JSONL 供重试，但数据库层会拒绝入库。
                for item in chunk:
                    results.append({
                        **item,
                        "sentiment_score": None,
                        "tags": [],
                        "mentioned_stocks": [],
                        "reasoning": "API 调用失败，未完成分析",
                        "analysis_status": "failed",
                    })
        return results

    # ------------------------------------------------------------------
    # 解析响应
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> list[dict]:
        """解析 DeepSeek 返回的 JSON，容错处理。"""
        # 移除可能的 markdown 代码块标记
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            return data.get("results", [])
        except json.JSONDecodeError:
            # 尝试提取 JSON 片段
            match = re.search(r"\{[\s\S]*\"results\"[\s\S]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("results", [])
                except json.JSONDecodeError:
                    pass
            logger.error(f"无法解析 DeepSeek 响应: {raw[:500]}...")
            return []

    # ------------------------------------------------------------------
    # 保存分析结果
    # ------------------------------------------------------------------

    @staticmethod
    def save_analyzed(results: list[dict]) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        filepath = DATA_DIR / f"analyzed_news_{today}.jsonl"

        with open(filepath, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"已保存 {len(results)} 条分析结果至 {filepath}")
        return filepath


# ======================================================================
# 主程序入口
# ======================================================================

if __name__ == "__main__":
    logger.info("===== DeepSeek 分析开始 =====")

    raw_file = AnalysisAgent.find_latest_raw_file()
    if raw_file is None:
        logger.error("无原始数据文件，请先运行 rss_fetcher.py")
        exit(1)

    news = AnalysisAgent.load_raw_news(raw_file)
    if not news:
        logger.error("原始数据为空")
        exit(1)

    try:
        agent = AnalysisAgent()
    except ValueError as e:
        logger.error(str(e))
        exit(1)

    analyzed = agent.analyze_batch(news)
    if analyzed:
        saved_path = AnalysisAgent.save_analyzed(analyzed)
        sentiments = [r.get("sentiment_score") for r in analyzed]
        valid = [s for s in sentiments if isinstance(s, (int, float))]
        failed = sum(r.get("analysis_status") != "success" for r in analyzed)
        avg_score = sum(valid) / len(valid) if valid else 0.0
        print(
            f"分析完成: 共 {len(analyzed)} 条新闻, "
            f"失败: {failed} 条, "
            f"平均情感分: {avg_score:+.2f}, "
            f"已保存至: {saved_path}"
        )
    else:
        logger.error("分析结果为空")
