#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
TASK="${1:-help}"

if [[ "$TASK" != "help" && "$TASK" != "setup" && ! -x "$VENV_PYTHON" ]]; then
    echo "[错误] 尚未创建 macOS Python 环境。"
    echo "请先双击 setup_mac.command，或运行："
    echo "  \"$SCRIPT_DIR/setup_mac.command\""
    exit 1
fi

cd "$SCRIPT_DIR" || exit 1
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$SCRIPT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

if [[ $# -gt 0 ]]; then
    shift
fi

case "$TASK" in
    setup)
        exec "$SCRIPT_DIR/setup_mac.command" "$@"
        ;;
    doctor)
        echo "项目目录: $SCRIPT_DIR"
        echo "Python:   $("$VENV_PYTHON" -c 'import sys; print(sys.executable)')"
        "$VENV_PYTHON" --version
        "$VENV_PYTHON" -c \
            "import akshare, colorama, feedparser, loguru, matplotlib, numpy, openai, pandas, requests, scipy; print('[OK] Python 依赖导入成功')"
        "$VENV_PYTHON" -c \
            "from pathlib import Path; from src.config import StrategyConfig; p=Path('data/financial_news.db'); c=StrategyConfig.from_env(); print(f'[OK] 配置: {c.timeframe}, Top {c.portfolio.top_n}'); print(f'[OK] 数据库: {p} ({p.stat().st_size:,} bytes)' if p.exists() else '[提示] 新闻数据库尚未创建；纯价格研究与离线测试不需要它。')"
        ;;
    news)
        exec "$VENV_PYTHON" -m src.daily_accumulate --no-pause --log-file "$@"
        ;;
    risk)
        exec "$VENV_PYTHON" -m src.daily_risk_monitor --check "$@"
        ;;
    weekly)
        exec "$VENV_PYTHON" -m src.forward_simulation --single "$@"
        ;;
    main)
        exec "$VENV_PYTHON" -m src.main "$@"
        ;;
    backtest)
        exec "$VENV_PYTHON" -m src.main backtest "$@"
        ;;
    validate)
        exec "$VENV_PYTHON" -m src.main backtest-validate "$@"
        ;;
    walkforward)
        exec "$VENV_PYTHON" -m src.walk_forward_validation "$@"
        ;;
    attribution)
        exec "$VENV_PYTHON" -m src.walk_forward_attribution "$@"
        ;;
    lowturnover)
        exec "$VENV_PYTHON" -m src.low_turnover_validation "$@"
        ;;
    factors)
        exec "$VENV_PYTHON" -m src.factor_screening "$@"
        ;;
    lowvol)
        exec "$VENV_PYTHON" -m src.low_volatility_portfolio_validation "$@"
        ;;
    fundamentals)
        exec "$VENV_PYTHON" -m src.fundamental_factor_screening "$@"
        ;;
    fundportfolio)
        exec "$VENV_PYTHON" -m src.fundamental_portfolio_validation "$@"
        ;;
    statements)
        exec "$VENV_PYTHON" -m src.statement_factor_screening "$@"
        ;;
    qualityvalue)
        exec "$VENV_PYTHON" -m src.quality_value_residual_screening "$@"
        ;;
    balancequality)
        exec "$VENV_PYTHON" -m src.balance_quality_factor_screening "$@"
        ;;
    balanceportfolio)
        exec "$VENV_PYTHON" -m src.balance_quality_portfolio_validation "$@"
        ;;
    balanceic)
        exec "$VENV_PYTHON" -m src.balance_quality_ic_diagnostic "$@"
        ;;
    holdout)
        exec "$VENV_PYTHON" -m src.frozen_holdout_validation "$@"
        ;;
    factorreset)
        exec "$VENV_PYTHON" -m src.factor_reset_screening "$@"
        ;;
    residualidio)
        exec "$VENV_PYTHON" -m src.residual_idio_screening "$@"
        ;;
    paper)
        exec "$VENV_PYTHON" -m src.paper_trading "$@"
        ;;
    v2universe)
        exec "$VENV_PYTHON" -m src.sse_composite_universe "$@"
        ;;
    v2audit)
        exec "$VENV_PYTHON" -m src.v2_monthly_validation "$@"
        ;;
    v2factordata)
        exec "$VENV_PYTHON" -m src.v2_factor_data "$@"
        ;;
    v2factors)
        exec "$VENV_PYTHON" -m src.v2_factor_screening "$@"
        ;;
    v2portfolio)
        exec "$VENV_PYTHON" -m src.v2_factor_portfolio_validation "$@"
        ;;
    v2attribution)
        exec "$VENV_PYTHON" -m src.v2_lowvol_failure_attribution "$@"
        ;;
    grid)
        exec "$VENV_PYTHON" -m src.main grid-search "$@"
        ;;
    help|-h|--help)
        cat <<'EOF'
用法: ./run_mac.sh <任务> [参数]

任务:
  doctor      检查 Python、依赖、配置和数据库
  news        每日新闻积累
  risk        周中风险扫描
  weekly      生成本周持仓信号
  main        运行原始主流程
  backtest    运行回测
  validate    运行策略对比验证
  walkforward 运行走步样本外验证
  attribution 运行走步样本外收益归因
  lowturnover 运行低换手价格策略验证
  factors     运行独立价格因子筛选（不修改正式策略）
  lowvol      运行低波动候选的组合级验证（不修改正式策略）
  fundamentals 筛选估值、质量和盈利改善因子（不修改正式策略）
  fundportfolio 验证1/PB及其与低波动的组合（不修改正式策略）
  statements  筛选真实年报质量与盈利因子（不修改正式策略）
  qualityvalue 筛选质量相对1/PB残差及非金融敏感性（不修改正式策略）
  balancequality 筛选资产负债表与现金流质量因子（不修改正式策略）
  balanceportfolio 验证低应计与营运资本纪律组合（不修改正式策略）
  balanceic   诊断质量因子IC为何没有转化为组合收益（不修改正式策略）
  holdout     运行2025-05-12之后的冻结参数真正未见样本审计
  factorreset 运行策略重建的第二代独立因子准入审计
  residualidio 检验低特质波动剔除旧低波动后的独立增量
  paper       前向模拟盘：init / capture / update / status / dry-run
  v2universe  更新v2历史时点上证综指可交易A股池（默认排除科创板）
  v2audit     审计v2股票池基础设施（不生成买入清单）
  v2factordata 下载并审计v2三个独立因子的全量历史数据
  v2factors   筛选v2低波动、行业中性1/PB和12-1动量因子
  v2portfolio 验证v2已通过因子的三个固定组合（不修改v1模拟盘）
  v2attribution 归因v2低波动因子为何未转化为组合收益
  grid        运行网格搜索

首次使用:
  ./setup_mac.command
EOF
        ;;
    *)
        echo "[错误] 未知任务: $TASK"
        "$0" help
        exit 2
        ;;
esac
