#!/usr/bin/env bash
# =============================================================================
# run_monitor.sh — attach a log2tb tailer to a training log (one-shot helper).
#
# Usage:
#   bash monitor/run_monitor.sh <training.log> <run_name> [nanotron|verl|auto]
#
# Starts monitor/log2tb.py in the background (nohup) writing TB events +
# metrics.jsonl to <repo>/outputs/tb/<run_name>. Idempotent: an existing
# tailer for the same run name is killed and restarted. TensorBoard itself is
# NOT started here — open it with `bash monitor/serve.sh`.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY="${MONITOR_PY:-/home/lawson/miniforge3/bin/python3}"

if [ $# -lt 2 ]; then
    echo "usage: bash monitor/run_monitor.sh <training.log> <run_name> [nanotron|verl|auto]" >&2
    exit 1
fi
LOG_FILE="$1"
RUN_NAME="$2"
FMT="${3:-auto}"
SEQ_LEN="${MONITOR_SEQ_LEN:-8192}"

TB_ROOT="$REPO_ROOT/outputs/tb"
TB_DIR="$TB_ROOT/$RUN_NAME"
PID_FILE="$TB_DIR/.log2tb.pid"
MONITOR_OUT="$TB_DIR/monitor.out"
mkdir -p "$TB_DIR"

# idempotent: kill a previous tailer for this run name
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "==> killing previous tailer (pid $OLD_PID) for run '$RUN_NAME'"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
fi

nohup "$PY" "$SCRIPT_DIR/log2tb.py" --log "$LOG_FILE" --tb-dir "$TB_DIR" \
      --format "$FMT" --seq-len "$SEQ_LEN" >> "$MONITOR_OUT" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
disown 2>/dev/null || true

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-$(hostname)}"

echo "==> run name    : $RUN_NAME"
echo "==> log file    : $LOG_FILE"
echo "==> format      : $FMT"
echo "==> tb dir      : $TB_DIR"
echo "==> tailer pid  : $NEW_PID (log: $MONITOR_OUT)"
echo "==> dashboard   : bash $SCRIPT_DIR/serve.sh"
echo "                 http://${HOST_IP}:6006  (run '${RUN_NAME}' appears once steps are logged)"
