#!/usr/bin/env bash
# =============================================================================
# convert_hf_to_nanotron.sh — wrap nanotron's examples/llama/convert_hf_to_nanotron.py
#
# Converts an HF Llama checkpoint (local dir, e.g. models/Llama-3.2-3B-Base)
# into nanotron format so a training config can use
#   model.init_method: {path: <converted dir>}   (ExistingCheckpointInit).
#
# Single-process torchrun is what the example script documents.
#
# Usage:
#   bash convert_hf_to_nanotron.sh <hf_model_dir> <output_dir>
# e.g.
#   bash convert_hf_to_nanotron.sh \
#       /hdd2/xiaolirui/OctoThinker/models/Llama-3.2-3B-Base \
#       /hdd2/xiaolirui/OctoThinker/outputs/midtraining/converted/Llama-3.2-3B-Base-nanotron
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)"
NANOTRON_DIR="$REPO_ROOT/midtraining/nanotron"
LLAMA_EXAMPLES="$NANOTRON_DIR/examples/llama"
ENV_PREFIX="${ENV_PREFIX:-/hdd2/xiaolirui/conda_envs/nanotron}"

HF_PATH="${1:?usage: convert_hf_to_nanotron.sh <hf_model_dir> <output_dir>}"
OUT_DIR="${2:?usage: convert_hf_to_nanotron.sh <hf_model_dir> <output_dir>}"

if [ -x "$ENV_PREFIX/bin/torchrun" ]; then
    PYBIN_DIR="$ENV_PREFIX/bin"
else
    echo "WARNING: $ENV_PREFIX not found, using torchrun from PATH" >&2
    PYBIN_DIR=""
fi

[ -f "$HF_PATH/config.json" ] || { echo "ERROR: $HF_PATH does not look like an HF model dir (no config.json)" >&2; exit 1; }
[ -d "$LLAMA_EXAMPLES" ] || { echo "ERROR: nanotron examples not found at $LLAMA_EXAMPLES" >&2; exit 1; }

mkdir -p "$(dirname "$OUT_DIR")"
# the example imports `convert_weights` from its own directory
export PYTHONPATH="$LLAMA_EXAMPLES:${PYTHONPATH:-}"

echo "==> converting HF checkpoint: $HF_PATH"
echo "==> nanotron output dir   : $OUT_DIR"
# Script args (see examples/llama/convert_hf_to_nanotron.py):
#   --checkpoint_path  dir with HF weights + config.json
#   --save_path        output dir in nanotron format
(cd "$LLAMA_EXAMPLES" && "${PYBIN_DIR:+$PYBIN_DIR/}torchrun" --nproc_per_node=1 \
    --master_addr 127.0.0.1 --master_port "${MASTER_PORT:-29400}" \
    convert_hf_to_nanotron.py \
    --checkpoint_path "$HF_PATH" \
    --save_path "$OUT_DIR")
echo "==> done. Point model.init_method.path at: $OUT_DIR"
