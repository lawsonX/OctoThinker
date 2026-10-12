#!/usr/bin/env bash
# =============================================================================
# run_midtraining.sh — launch a nanotron mid-training run on this node.
#
# Prerequisites:
#   * conda env from midtraining/env_setup.sh
#   * base model converted (run/convert_hf_to_nanotron.sh)
#   * data tokenized (run/tokenize_data.sh)
#
# Usage:
#   CONFIG=configs/smoke_3b_tiny.yaml bash run_midtraining.sh
#   CONFIG=configs/ablation_3b_20B_cosine.yaml bash run_midtraining.sh
#   RESUME=1 CONFIG=configs/ablation_3b_20B_cosine.yaml bash run_midtraining.sh
#
# Env vars:
#   CONFIG        path to a nanotron yaml (relative to run/configs/ or absolute)
#   RESUME=1      resume from the latest checkpoint in checkpoints.checkpoints_path
#                 (sets checkpoints.resume_checkpoint_path on a temp copy of the config)
#   MONITOR=1     start monitor/log2tb.py (base python) tailing the log into
#                 outputs/tb/<run>; view with `bash monitor/serve.sh` (:6006)
#   HF_HOME       optional HF cache override (default: leave unset / ~/.cache)
#   MASTER_PORT   torchrun rendezvous port (default 29500)
#   NANOTRON_DIR  override nanotron source dir
#
# The world size (torchrun --nproc_per_node) is derived from the config's
# parallelism product dp x tp x pp, so it always matches the YAML.
# Logs are teed to outputs/midtraining/logs/<run>_<timestamp>.log (nohup-safe).
# =============================================================================
set -euo pipefail

RUN_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(dirname "$(dirname "$RUN_DIR")")"
NANOTRON_DIR="${NANOTRON_DIR:-$REPO_ROOT/midtraining/nanotron}"
ENV_PREFIX="${ENV_PREFIX:-/hdd2/xiaolirui/conda_envs/nanotron}"
MASTER_PORT="${MASTER_PORT:-29500}"

CONFIG="${CONFIG:?set CONFIG to a yaml, e.g. CONFIG=configs/smoke_3b_tiny.yaml}"
case "$CONFIG" in
    /*) ;;
    *)  CONFIG="$RUN_DIR/$CONFIG" ;;
esac
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

if [ -x "$ENV_PREFIX/bin/python" ]; then
    PY="$ENV_PREFIX/bin/python"
    TORCHRUN="$ENV_PREFIX/bin/torchrun"
else
    echo "WARNING: $ENV_PREFIX not found, using python3/torchrun from PATH" >&2
    PY="python3"
    TORCHRUN="torchrun"
fi

# --- derive world size + run name + checkpoints path from the YAML ----------
read -r WORLD_SIZE RUN_NAME CKPT_PATH SEQ_LEN <<<"$("$PY" - "$CONFIG" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
p = cfg.get("parallelism") or {}
world = int(p.get("dp", 1)) * int(p.get("tp", 1)) * int(p.get("pp", 1))
name = (cfg.get("general") or {}).get("run") or "nanotron_run"
ckpt = ((cfg.get("checkpoints") or {}).get("checkpoints_path")) or ""
seq = (cfg.get("tokens") or {}).get("sequence_length") or 8192
print(world, name, ckpt, seq)
EOF
)"

LOG_DIR="$REPO_ROOT/outputs/midtraining/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
CONFIG_TO_RUN="$CONFIG"

if [ "${RESUME:-0}" = "1" ]; then
    [ -n "$CKPT_PATH" ] || { echo "ERROR: RESUME=1 but config has no checkpoints.checkpoints_path" >&2; exit 1; }
    [ -d "$CKPT_PATH" ] || { echo "ERROR: RESUME=1 but checkpoint dir $CKPT_PATH does not exist" >&2; exit 1; }
    CONFIG_TO_RUN="$(mktemp /tmp/nanotron_resume_XXXX.yaml)"
    # nanotron's parse_ckpt_path reads latest.txt inside resume_checkpoint_path
    "$PY" - "$CONFIG" "$CKPT_PATH" "$CONFIG_TO_RUN" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault("checkpoints", {})["resume_checkpoint_path"] = sys.argv[2]
yaml.safe_dump(cfg, open(sys.argv[3], "w"))
EOF
    echo "==> RESUME=1: resume_checkpoint_path=$CKPT_PATH (temp config $CONFIG_TO_RUN)"
fi

# --- environment -------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"        # important for TP/distributed ops (nanotron docs)
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export TOKENIZERS_PARALLELISM=false
if [ -n "${HF_HOME:-}" ]; then
    export HF_HOME
    echo "==> HF_HOME=$HF_HOME"
fi

echo "==> config      : $CONFIG_TO_RUN"
echo "==> run name    : $RUN_NAME"
echo "==> world size  : $WORLD_SIZE (dp*tp*pp from config)"
echo "==> log file    : $LOG_FILE"
echo "==> nanotron    : $NANOTRON_DIR"

# --- launch ------------------------------------------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1

# opt-in training monitor: tail the log into TensorBoard events (base python,
# never touches the training processes). View with `bash monitor/serve.sh`.
if [ "${MONITOR:-0}" = "1" ]; then
    TB_DIR="$REPO_ROOT/outputs/tb/$RUN_NAME"
    mkdir -p "$TB_DIR"
    MONITOR_PY="${MONITOR_PY:-/home/lawson/miniforge3/bin/python3}"
    echo "==> monitor     : MONITOR=1, tailing into $TB_DIR (serve: bash monitor/serve.sh)"
    "$MONITOR_PY" "$REPO_ROOT/monitor/log2tb.py" --log "$LOG_FILE" \
        --tb-dir "$TB_DIR" --format nanotron --seq-len "${SEQ_LEN:-8192}" \
        >> "$TB_DIR/monitor.out" 2>&1 &
    MONITOR_PID=$!
    _kill_monitor() { kill "$MONITOR_PID" 2>/dev/null || true; }
    trap _kill_monitor EXIT
    trap '_kill_monitor; exit 143' TERM
fi

# Background + wait so trapped signals run promptly (a trap cannot fire while
# bash blocks on a foreground child); `set -e` still propagates torchrun's exit.
"$TORCHRUN" --nproc_per_node="$WORLD_SIZE" \
    --nnodes 1 \
    --rdzv_backend c10d \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --max_restarts 0 \
    "$NANOTRON_DIR/run_train.py" --config-file "$CONFIG_TO_RUN" &
TRAIN_PID=$!
wait "$TRAIN_PID"
