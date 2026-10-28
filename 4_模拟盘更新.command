#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "============================================================"
echo "  冻结策略前向模拟盘更新"
echo "============================================================"
echo

./run_mac.sh paper update
STATUS=$?

echo
if [ "$STATUS" -eq 0 ]; then
    echo "[完成] 已确认到期模拟成交，并补记可用交易日净值。"
    echo "报告: data/paper_trading/report.html"
else
    echo "[失败] 模拟盘未更新，请查看上方错误信息。"
fi
echo
read -r -p "按回车键关闭窗口..."
exit "$STATUS"
