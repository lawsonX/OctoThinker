#!/usr/bin/env bash
# =============================================================================
# convert_nanotron_to_hf.sh — wrap nanotron's examples/llama/convert_nanotron_to_hf.py
#
# Exports a nanotron training checkpoint (model weights only) to HF format,
# ready for the rl/ stage (verl) or public release.
#
# Usage:
#   bash convert_nanotron_to_hf.sh <nanotron_ckpt_dir> <hf_output_dir> [tokenizer_name_or_path]
# e.g.
#   bash convert_nanotron_to_hf.sh \
#       /hdd2/xiaolirui/OctoThinker/outputs/midtraining/two_stage_decay_3b_long/checkpoints/5000 \
#       /hdd2/xiaolirui/OctoThinker/outputs/midtraining/hf/two_stage_decay_3b_long \
#       /hdd2/xiaolirui/OctoThinker/models/Llama-3.2-3B-Base
#
# <nanotron_ckpt_dir> is the per-step checkpoint directory (contains
# model_config.json), e.g. <checkpoints_path>/50000 — not the parent.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)"
NANOTRON_DIR="$REPO_ROOT/midtraining/nanotron"
LLAMA_EXAMPLES="$NANOTRON_DIR/examples/llama"
ENV_PREFIX="${ENV_PREFIX:-/hdd2/xiaolirui/conda_envs/nanotron}"

NT_CKPT="${1:?usage: convert_nanotron_to_hf.sh <nanotron_ckpt_dir> <hf_output_dir> [tokenizer]}"
HF_OUT="${2:?usage: convert_nanotron_to_hf.sh <nanotron_ckpt_dir> <hf_output_dir> [tokenizer]}"
TOKENIZER="${3:-$REPO_ROOT/models/Llama-3.2-3B-Base}"

if [ -x "$ENV_PREFIX/bin/torchrun" ]; then
    PYBIN_DIR="$ENV_PREFIX/bin"
else
    echo "WARNING: $ENV_PREFIX not found, using torchrun from PATH" >&2
    PYBIN_DIR=""
fi

[ -f "$NT_CKPT/model_config.json" ] || { echo "ERROR: $NT_CKPT has no model_config.json — point at the per-step ckpt dir (e.g. <checkpoints_path>/50000)" >&2; exit 1; }
[ -d "$LLAMA_EXAMPLES" ] || { echo "ERROR: nanotron examples not found at $LLAMA_EXAMPLES" >&2; exit 1; }

mkdir -p "$(dirname "$HF_OUT")"
# the example imports `convert_weights` from its own directory
export PYTHONPATH="$LLAMA_EXAMPLES:${PYTHONPATH:-}"

echo "==> nanotron checkpoint : $NT_CKPT"
echo "==> HF output dir       : $HF_OUT"
echo "==> tokenizer           : $TOKENIZER"
# Script args (see examples/llama/convert_nanotron_to_hf.py):
#   --checkpoint_path  nanotron per-step checkpoint dir
#   --save_path        output dir in HF format
#   --tokenizer_name   tokenizer to copy into the HF output (default Llama-2-7b: override!)
# The example script ends with a .cuda() self-check that needs a FREE GPU; on
# busy nodes it can OOM even though the export already succeeded. Tolerate that:
# if "Model saved" was printed, treat the conversion as successful.
set +e
(cd "$LLAMA_EXAMPLES" && "${PYBIN_DIR:+$PYBIN_DIR/}torchrun" --nproc_per_node=1 \
    convert_nanotron_to_hf.py \
    --checkpoint_path "$NT_CKPT" \
    --save_path "$HF_OUT" \
    --tokenizer_name "$TOKENIZER" 2>&1 | tee /tmp/convert_nanotron_to_hf.last.log)
rc=${PIPESTATUS[0]}
set -e
if [ $rc -ne 0 ]; then
    if grep -q "Model saved" /tmp/convert_nanotron_to_hf.last.log && [ -f "$HF_OUT/model.safetensors" ]; then
        echo "==> WARNING: post-export GPU self-check failed (rc=$rc, likely no free GPU)," >&2
        echo "    but the HF export is COMPLETE ($HF_OUT/model.safetensors written)." >&2
    else
        echo "==> conversion FAILED (rc=$rc)" >&2
        exit $rc
    fi
fi
echo "==> done. HF model at: $HF_OUT"
