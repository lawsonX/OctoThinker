#!/usr/bin/env bash
# 启动 Parquet Viewer. 首次运行自动建 venv 并装依赖.
# 用法: ./run.sh [端口] [数据根目录]
set -e
cd "$(dirname "$0")"
PORT="${1:-8790}"
ROOT="${2:-/hdd2/xiaolirui/octoThinker-Datasets}"

if [ ! -d .venv ]; then
  echo "首次运行: 创建 venv ..."
  python3 -m venv .venv
  ./.venv/bin/pip install -q --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      -r requirements.txt
fi

echo "数据目录: $ROOT"
echo "访问:     http://$(hostname -I | awk '{print $1}'):$PORT  (本机 http://127.0.0.1:$PORT)"
exec ./.venv/bin/python app.py --root "$ROOT" --port "$PORT"
