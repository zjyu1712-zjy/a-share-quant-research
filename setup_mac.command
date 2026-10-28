#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
STATUS=0

finish() {
    STATUS=$?
    echo
    if [[ $STATUS -eq 0 ]]; then
        echo "[完成] MacBook 运行环境已经准备好。"
        echo "接下来请按 README.md 运行离线测试和 v2 研究流程。"
        echo "1–4 号快捷入口属于旧版 v1，不是 v2 的实盘选股入口。"
    else
        echo "[失败] 环境安装未完成，退出码: $STATUS"
    fi
    if [[ -t 0 ]]; then
        read -r -p "按 Enter 关闭窗口..."
    fi
    exit "$STATUS"
}
trap finish EXIT

cd "$SCRIPT_DIR" || exit 1

echo "============================================================"
echo "  Quant Weekly System — macOS 环境安装"
echo "============================================================"
echo "项目目录: $SCRIPT_DIR"

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "[错误] 需要 Python 3.10 或更高版本。"
    echo "建议安装 Homebrew Python 3.12：brew install python@3.12"
    exit 1
fi

echo "Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[1/4] 创建独立虚拟环境..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[1/4] 虚拟环境已存在，继续使用。"
fi

echo "[2/4] 更新 pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "[3/4] 安装项目依赖..."
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    echo "[提示] 未发现 .env，已从 .env.example 创建模板。"
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "请打开 .env 填写 DEEPSEEK_API_KEY 后再运行新闻积累。"
fi

chmod 755 \
    "$SCRIPT_DIR/run_mac.sh" \
    "$SCRIPT_DIR/setup_mac.command" \
    "$SCRIPT_DIR/1_周末选股.command" \
    "$SCRIPT_DIR/2_每日新闻积累.command" \
    "$SCRIPT_DIR/3_周中风险扫描.command"

echo "[4/4] 执行环境自检..."
"$SCRIPT_DIR/run_mac.sh" doctor
