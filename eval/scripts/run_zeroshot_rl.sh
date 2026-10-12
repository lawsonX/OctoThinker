#!/usr/bin/env bash
# =============================================================================
# Zero-shot CoT evaluation of RL-TUNED models (OctoThinker-*-Zero, *-Zero),
# matching the paper protocol: RL-tuned models are evaluated ZERO-SHOT with
# the same prompt template used in RL training (orz_base "Complex Template",
# reproduced by the custom_prompt field in
# configs/en_math_zero_shot_test_configs.json).
#
# Temperature 0 (greedy decoding), as in the paper's eval protocol.
#
# Usage:
#   conda activate /hdd2/xiaolirui/conda_envs/matheval
#   MODEL_PATH=models/OctoThinker-3B-Long-Zero bash eval/scripts/run_zeroshot_rl.sh
#   MODEL_PATH=models/Llama3.2-3B-Zero CUDA_VISIBLE_DEVICES=3,5 bash eval/scripts/run_zeroshot_rl.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/OctoThinker-3B-Long-Zero}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
MODEL_SIZE=${MODEL_SIZE:-3b}
# Respect a pre-set CUDA_VISIBLE_DEVICES when NGPUS is not given explicitly.
if [ -z "${NGPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NGPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
    else
        NGPUS=6
    fi
fi
TEMPERATURE=${TEMPERATURE:-0}          # paper uses greedy decoding for eval
N_REPEATS=${N_REPEATS:-1}
TEST_CONF=${TEST_CONF:-configs/en_math_zero_shot_test_configs_partial.json}
# Optional override for the zero-shot template; when empty, each benchmark's
# custom_prompt field in the config (the orz_base reproduction) is used.
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}

MODEL_NAME=$(basename "${MODEL_PATH%/}")
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${MODEL_NAME}}
SUMMARIZE_DIR=${SUMMARIZE_DIR:-outputs/summary}

export TOKENIZERS_PARALLELISM=false

echo "MODEL_PATH   = ${MODEL_PATH}"
echo "MODEL_SIZE   = ${MODEL_SIZE} | NGPUS = ${NGPUS} | TEMPERATURE = ${TEMPERATURE}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "TEST_CONF    = ${TEST_CONF}"

cd "${EVAL_DIR}"

EXTRA_ARGS=()
if [ -n "${SYSTEM_PROMPT}" ]; then
    EXTRA_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")
fi

python run_subset_parallel.py \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --ngpus "${NGPUS}" \
    --model-size "${MODEL_SIZE}" \
    --use-vllm \
    --load_in_half \
    --test-conf "${TEST_CONF}" \
    --prompt_format few_shot \
    --no-markup-question \
    --temperature "${TEMPERATURE}" \
    --n-repeats "${N_REPEATS}" \
    "${EXTRA_ARGS[@]}"

# Aggregate. NOTE: summarize_results.py treats each directory under outputs/
# as one model, so zero-shot results for an RL model must live under
# outputs/<model-name> (they do, via OUTPUT_DIR) -- do NOT mix base few-shot
# and RL zero-shot runs for the SAME model name in one outputs/ tree.
python summarize_results.py \
    --dirname outputs \
    --summarize_dir "${SUMMARIZE_DIR}"

echo "Done. Summary: ${EVAL_DIR}/${SUMMARIZE_DIR}/evaluation_results.csv (+ results.png)"
