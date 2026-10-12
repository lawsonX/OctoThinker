#!/usr/bin/env bash
# =============================================================================
# serve.sh — start TensorBoard for the training-monitor dashboards.
#
# Usage:
#   bash monitor/serve.sh [tb_root] [port]
#   bash monitor/serve.sh --root <tb_root> [--port <port>]
#
# tb_root defaults to <repo>/outputs/tb. Each subdirectory of tb_root is one
# training run (created by log2tb.py / run_monitor.sh), so one TensorBoard
# instance shows every monitored run side by side.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY="${MONITOR_PY:-/home/lawson/miniforge3/bin/python3}"

TB_ROOT="$REPO_ROOT/outputs/tb"
PORT=6006

while [ $# -gt 0 ]; do
    case "$1" in
        --root) TB_ROOT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            if [ -z "${1##*[!0-9]*}" ]; then
                TB_ROOT="$1"  # contains a non-digit -> path
            else
                PORT="$1"     # all digits -> port
            fi
            shift ;;
    esac
done

[ -d "$TB_ROOT" ] || { echo "ERROR: tb root not found: $TB_ROOT" >&2; exit 1; }

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-$(hostname)}"

echo "==> tensorboard logdir : $TB_ROOT"
echo "==> runs               : $(find "$TB_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "==> URL (this machine) : http://127.0.0.1:${PORT}"
echo "==> URL (LAN)          : http://${HOST_IP}:${PORT}"
exec "$PY" -m tensorboard.main --logdir "$TB_ROOT" --port "$PORT" --host 0.0.0.0
