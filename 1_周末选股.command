#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
"$SCRIPT_DIR/run_mac.sh" weekly
STATUS=$?
echo
if [[ $STATUS -eq 0 ]]; then
    echo "[完成] 持仓清单: $SCRIPT_DIR/data/weekly_result.txt"
else
    echo "[失败] 周末选股退出码: $STATUS"
fi
if [[ -t 0 ]]; then read -r -p "按 Enter 关闭窗口..."; fi
exit "$STATUS"
