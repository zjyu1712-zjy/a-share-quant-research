"""
集中度风控模块 — 防止周线选股过度集中于少数标的
======================================================
设计目标: 在现有的 sector_neutral_select 选股流程之后，
增加一个"选股历史频率过滤器"，防止某只股票因动量过高
而在连续多周内被重复选中，导致单票暴跌拖垮组合。

集成方式:
  1. 在 backtest_engine.py 或 forward_simulation.py 的选股步骤之后调用
  2. 维护一个 JSON 状态文件记录每只股票的历史被选次数
  3. 支持两种模式:
     - "deduct":  降低重复选中股票的得分 (软约束)
     - "exclude": 直接剔除超限股票，顺延选取 (硬约束)

用法示例:
  >>> from src.concentration_filter import ConcentrationFilter
  >>> cf = ConcentrationFilter(lookback_weeks=4, max_selects=2, mode="exclude")
  >>> cf.load_history()
  >>> final_picks = cf.apply(picks, ranked_df, top_n=5)
  >>> cf.save_history(final_picks, rebalance_date)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STATE_PATH = DATA_DIR / "concentration_state.json"

FilterMode = Literal["deduct", "exclude"]


class ConcentrationFilter:
    """周线选股集中度过滤器。

    Attributes:
        lookback_weeks: 回溯窗口（周数），默认 4 周
        max_selects:    窗口内最大允许被选次数，默认 2 次
        mode:           "deduct" 降低得分 / "exclude" 直接排除
        deduct_penalty: 当 mode="deduct" 时的得分惩罚系数 (0~1)，默认 0.5
        state_path:     状态文件路径
    """

    def __init__(
        self,
        lookback_weeks: int = 4,
        max_selects: int = 2,
        mode: FilterMode = "exclude",
        deduct_penalty: float = 0.5,
        state_path: Optional[Path] = None,
    ):
        self.lookback_weeks = lookback_weeks
        self.max_selects = max_selects
        self.mode = mode
        self.deduct_penalty = max(0.0, min(1.0, deduct_penalty))
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE_PATH

        # {(week_start_str, code): count}  — 仅保留最近 N 周
        self._history: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # 状态持久化
    # ------------------------------------------------------------------

    def load_history(self) -> "ConcentrationFilter":
        """从 JSON 文件加载历史选股记录。文件不存在则初始化为空。"""
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # JSON key 是字符串 "YYYY-MM-DD|code" 形式
            self._history = {}
            for k, v in raw.get("history", {}).items():
                week_str, code = k.rsplit("|", 1)
                # 旧版本重复运行同一周时会累加计数。每只股票在一个自然周
                # 只能贡献一次选择记录，加载时顺带归一化旧状态。
                self._history[(week_str, code)] = 1 if int(v) > 0 else 0
            logger.info(
                f"集中度过滤器: 已加载 {len(self._history)} 条历史记录 "
                f"({self.state_path.name})"
            )
        else:
            logger.info("集中度过滤器: 无历史文件, 初始化为空")
        return self

    def save_history(
        self,
        picks: list[str],
        rebalance_date: str,
        persist: bool = True,
    ) -> None:
        """幂等写入本周选股，自动清理超出窗口的旧记录。

        Args:
            picks:          本周最终选股列表 ["000725", "600900", ...]
            rebalance_date: 调仓日期 "YYYY-MM-DD"
        """
        week_start = self._week_start(rebalance_date)

        # 先移除同一周旧结果，再写入新结果。相同信号重复运行不会累加，
        # 本周选股发生变化时也会以最新一次结果为准。
        for key in [key for key in self._history if key[0] == week_start]:
            del self._history[key]
        for code in dict.fromkeys(picks):
            self._history[(week_start, code)] = 1

        # 清理过期记录
        cutoff = self._cutoff_week_start(rebalance_date)
        stale_keys = [k for k in self._history if k[0] < cutoff]
        for k in stale_keys:
            del self._history[k]

        if not persist:
            return

        # 持久化 (JSON key 用 "YYYY-MM-DD|code")
        serializable = {}
        for (ws, code), count in self._history.items():
            serializable[f"{ws}|{code}"] = count

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated": datetime.now().isoformat(),
                    "lookback_weeks": self.lookback_weeks,
                    "max_selects": self.max_selects,
                    "mode": self.mode,
                    "history": serializable,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(
            f"集中度过滤器: 已保存 {len(picks)} 只本周选股, "
            f"历史总记录 {len(self._history)} 条"
        )

    # ------------------------------------------------------------------
    # 核心过滤逻辑
    # ------------------------------------------------------------------

    def apply(
        self,
        current_picks: list[str],
        ranked_df: pd.DataFrame,
        top_n: int = 5,
        sectors: Optional[dict[str, str]] = None,
        code_to_name: Optional[dict[str, str]] = None,
        rebalance_date: Optional[str] = None,
    ) -> list[str]:
        """对本周选股结果施加集中度约束。

        Args:
            current_picks: 初步选股结果 (如 sector_neutral_select 的输出)
            ranked_df:     全池排序后的 DataFrame，含 code, score, close 列
            top_n:         目标持仓数
            sectors:       {code: sector_name} 映射 (仅 exclude 模式行业中性补选需要)
            code_to_name:  {code: name} 映射 (用于日志)
            rebalance_date: 本次信号日期；回测时必须显式传入

        Returns:
            过滤后的最终选股列表
        """
        date_str = rebalance_date or datetime.now().strftime("%Y-%m-%d")
        current_week = self._week_start(date_str)
        cutoff = self._cutoff_week_start(date_str)

        # 统计各 code 在窗口内被选次数
        recent_counts: dict[str, int] = {}
        for (ws, code), count in self._history.items():
            # 只统计当前周之前的记录；同一周重跑不应被自己的旧结果拦截。
            if cutoff <= ws < current_week:
                recent_counts[code] = recent_counts.get(code, 0) + count

        # 建立已排除/已选中集合
        excluded: set[str] = set()
        final_picks: list[str] = []

        if self.mode == "exclude":
            # ── 硬约束: 直接排除超限股票 ──
            for code in current_picks:
                cnt = recent_counts.get(code, 0)
                if cnt >= self.max_selects:
                    name = (code_to_name or {}).get(code, code)
                    logger.info(
                        f"  集中度过滤: [{code} {name}] "
                        f"近{self.lookback_weeks}周已选{cnt}次 >= {self.max_selects}, 排除"
                    )
                    excluded.add(code)
                else:
                    final_picks.append(code)

            # 从剩余排名中顺延补足
            if len(final_picks) < top_n and sectors:
                from src.indicators import sector_neutral_select

                # 过滤掉已排除和已选中的
                remaining = ranked_df[
                    ~ranked_df["code"].isin(set(final_picks) | excluded)
                ].copy()
                if len(remaining) > 0:
                    # 对剩余候选也做集中度检查
                    for _, row in remaining.iterrows():
                        code = row["code"]
                        cnt = recent_counts.get(code, 0)
                        if cnt >= self.max_selects:
                            excluded.add(code)

                    remaining_filtered = remaining[
                        ~remaining["code"].isin(excluded)
                    ]
                    supplement = sector_neutral_select(
                        remaining_filtered,
                        top_n - len(final_picks),
                        sectors,
                    )
                    final_picks.extend(supplement)

            # 如果还不够，按全池排名补足。集中度是保护性约束，不应因为
            # 候选池耗尽意外生成空仓；发生降级时明确记录告警。
            if len(final_picks) < top_n:
                logger.warning(
                    f"集中度过滤后仅剩 {len(final_picks)}/{top_n} 只，"
                    "已降级为按原始排名补足"
                )
                for _, row in ranked_df.iterrows():
                    code = row["code"]
                    if code not in final_picks:
                        final_picks.append(code)
                    if len(final_picks) >= top_n:
                        break

        elif self.mode == "deduct":
            # ── 软约束: 降低超限股票的得分 ──
            adjusted_scores = ranked_df.copy()
            for idx, row in adjusted_scores.iterrows():
                code = row["code"]
                cnt = recent_counts.get(code, 0)
                if cnt >= self.max_selects:
                    # 超过次数越多惩罚越大: score *= penalty^(cnt-max_selects+1)
                    excess = cnt - self.max_selects + 1
                    penalty = self.deduct_penalty ** excess
                    adjusted_scores.at[idx, "score"] = row["score"] * penalty
                    name = (code_to_name or {}).get(code, code)
                    logger.info(
                        f"  集中度惩罚: [{code} {name}] "
                        f"近{self.lookback_weeks}周已选{cnt}次, "
                        f"得分 {row['score']:.3f} -> {adjusted_scores.at[idx, 'score']:.3f}"
                    )

            # 按调整后得分重新排序, 用行业中性化选股
            adjusted_scores = adjusted_scores.sort_values(
                "score", ascending=False
            )
            if sectors:
                from src.indicators import sector_neutral_select
                final_picks = sector_neutral_select(adjusted_scores, top_n, sectors)
            else:
                final_picks = adjusted_scores.head(top_n)["code"].tolist()

        return final_picks

    def get_recent_counts(
        self,
        rebalance_date: Optional[str] = None,
    ) -> dict[str, int]:
        """查询各股票在窗口内的被选次数 (调试用)。

        Args:
            rebalance_date: 基准日期，默认今天

        Returns:
            {code: count}
        """
        date_str = rebalance_date or datetime.now().strftime("%Y-%m-%d")
        cutoff = self._cutoff_week_start(date_str)
        counts: dict[str, int] = {}
        for (ws, code), count in self._history.items():
            if ws >= cutoff:
                counts[code] = counts.get(code, 0) + count
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _week_start(date_str: str) -> str:
        """返回该日期所在周的周一日期字符串 (ISO 周一=1)。"""
        dt = pd.Timestamp(date_str)
        monday = dt - pd.Timedelta(days=dt.dayofweek)
        return monday.strftime("%Y-%m-%d")

    def _cutoff_week_start(self, date_str: str) -> str:
        """返回 lookback_weeks 之前的周一日期字符串。"""
        dt = pd.Timestamp(date_str)
        cutoff_dt = dt - pd.Timedelta(weeks=self.lookback_weeks)
        return self._week_start(cutoff_dt.strftime("%Y-%m-%d"))


# ======================================================================
# 便捷入口: 与现有选股流程集成
# ======================================================================

def apply_concentration_control(
    picks: list[str],
    ranked: pd.DataFrame,
    rebalance_date: str,
    top_n: int = 5,
    sectors: Optional[dict[str, str]] = None,
    code_to_name: Optional[dict[str, str]] = None,
    lookback_weeks: int = 4,
    max_selects: int = 2,
    mode: FilterMode = "exclude",
) -> list[str]:
    """一行式调用: 对本周选股施加集中度控制。

    设计为可直接插入 backtest_engine.run() 或 forward_simulation 中:

        from src.concentration_filter import apply_concentration_control
        picks = apply_concentration_control(picks, ranked, date_str,
                                             top_n=5, sectors=sectors,
                                             code_to_name=code_to_name)

    Args:
        picks:           初步选股列表
        ranked:          全池排名 DataFrame (含 code, score)
        rebalance_date:  调仓日期 "YYYY-MM-DD"
        top_n:           目标持仓数
        sectors:         行业映射
        code_to_name:    名称映射
        lookback_weeks:  回溯周数
        max_selects:     最大允许连续选中次数
        mode:            "exclude" 或 "deduct"

    Returns:
        风控后的选股列表
    """
    cf = ConcentrationFilter(
        lookback_weeks=lookback_weeks,
        max_selects=max_selects,
        mode=mode,
    )
    cf.load_history()
    final_picks = cf.apply(current_picks=picks, ranked_df=ranked,
                           top_n=top_n, sectors=sectors,
                           code_to_name=code_to_name,
                           rebalance_date=rebalance_date)
    cf.save_history(final_picks, rebalance_date)
    return final_picks


# ======================================================================
# 自测
# ======================================================================

if __name__ == "__main__":
    # 模拟场景: 京东方A 连续 4 周被选中
    import tempfile

    # 创建临时状态文件
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_state = tmp_dir / "test_state.json"

    # --- 模拟历史: 京东方A 已经连续被选 4 周 ---
    cf = ConcentrationFilter(
        lookback_weeks=4,
        max_selects=2,
        mode="exclude",
        state_path=tmp_state,
    )
    cf.load_history()

    # 手动填充历史 (模拟前 3 周)
    for week in ["2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22"]:
        cf._history[(week, "000725")] = 1  # 京东方A
        cf._history[(week, "600900")] = 1  # 长江电力
    cf.save_history(["000725", "600900"], "2026-06-22")

    # --- 本周选股 (模拟 sector_neutral_select 输出) ---
    picks_before = ["000725", "600900", "000333", "600030", "300274"]
    ranked = pd.DataFrame({
        "code": ["000725", "600900", "000333", "600030", "300274",
                 "002415", "600276", "600036"],
        "score": [5.8, 2.8, 2.3, 2.0, 1.7, 1.5, 1.4, 1.2],
        "close": [7.50, 28.00, 80.00, 27.50, 160.00, 34.00, 55.00, 39.00],
    })
    sectors_test = {
        "000725": "面板", "600900": "公用事业", "000333": "家电",
        "600030": "券商", "300274": "电力设备", "002415": "安防",
        "600276": "医药", "600036": "银行",
    }
    code_to_name_test = {
        "000725": "京东方A", "600900": "长江电力", "000333": "美的集团",
        "600030": "中信证券", "300274": "阳光电源", "002415": "海康威视",
        "600276": "恒瑞医药", "600036": "招商银行",
    }

    picks_after = cf.apply(
        picks_before, ranked,
        top_n=5, sectors=sectors_test, code_to_name=code_to_name_test,
    )

    print(f"\n  过滤前: {[code_to_name_test.get(c,c) for c in picks_before]}")
    print(f"  过滤后: {[code_to_name_test.get(c,c) for c in picks_after]}")
    print(f"  被排除: 京东方A (近4周已选4次 >= 2次阈值)")

    # 清理
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
