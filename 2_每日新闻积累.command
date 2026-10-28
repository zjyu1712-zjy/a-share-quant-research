#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
"$SCRIPT_DIR/run_mac.sh" news
STATUS=$?
echo
if [[ $STATUS -eq 0 ]]; then
    echo "[完成] 新闻数据库: $SCRIPT_DIR/data/financial_news.db"
else
    echo "[失败] 新闻积累退出码: $STATUS"
fi
if [[ -t 0 ]]; then read -r -p "按 Enter 关闭窗口..."; fi
exit "$STATUS"
