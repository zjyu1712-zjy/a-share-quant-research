#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

if [[ ! -f "$SCRIPT_DIR/data/portfolio_state.json" ]]; then
    echo "[提示] 尚无持仓状态，请先运行 1_周末选股.command。"
    if [[ -t 0 ]]; then read -r -p "按 Enter 关闭窗口..."; fi
    exit 0
fi

"$SCRIPT_DIR/run_mac.sh" risk
STATUS=$?
echo
if [[ $STATUS -eq 0 ]]; then
    echo "[安全] 当前持仓未触发熔断。"
elif [[ $STATUS -eq 2 ]]; then
    echo "[警报] 已触发熔断，请查看 data/midweek_alert.json。"
else
    echo "[失败] 风险扫描退出码: $STATUS"
fi
if [[ -t 0 ]]; then read -r -p "按 Enter 关闭窗口..."; fi
exit "$STATUS"
