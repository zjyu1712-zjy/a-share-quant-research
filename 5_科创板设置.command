#!/bin/zsh

PROJECT_DIR="${0:A:h}"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR" || exit 1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "没有找到项目运行环境，请先完成项目安装。"
  read "reply?按回车关闭窗口..."
  exit 1
fi

while true; do
  clear
  echo "========================================"
  echo "        科创板入口设置（v2 独立）"
  echo "========================================"
  "$PYTHON_BIN" -m src.star_market status
  echo
  echo "1. 刷新状态"
  echo "2. 确认券商权限后开启入口"
  echo "3. 关闭入口（保留权限确认）"
  echo "4. 撤销权限确认并关闭入口"
  echo "Q. 退出"
  echo
  read "choice?请选择："

  case "$choice" in
    1)
      ;;
    2)
      echo
      echo "程序不会判断您的账户资格，也不能绕过券商适当性要求。"
      echo "仅当券商已经实际开通科创板交易权限时，才能继续。"
      echo "参考条件：申请前20个交易日日均资产不低于50万元、"
      echo "交易经验满24个月并签署风险揭示书；实际规则以届时券商和上交所为准。"
      read "confirmation?如券商已实际开通，请输入“我已确认”："
      if [[ "$confirmation" == "我已确认" ]]; then
        "$PYTHON_BIN" -m src.star_market enable --permission-confirmed
      else
        echo "未完成确认，入口保持关闭。"
      fi
      read "reply?按回车继续..."
      ;;
    3)
      "$PYTHON_BIN" -m src.star_market disable
      read "reply?入口已关闭，按回车继续..."
      ;;
    4)
      read "confirmation?输入“撤销”以清除权限确认："
      if [[ "$confirmation" == "撤销" ]]; then
        "$PYTHON_BIN" -m src.star_market revoke-permission
      else
        echo "未撤销，设置保持不变。"
      fi
      read "reply?按回车继续..."
      ;;
    q|Q)
      break
      ;;
    *)
      echo "无法识别该选项。"
      read "reply?按回车继续..."
      ;;
  esac
done
