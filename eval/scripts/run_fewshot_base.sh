#!/usr/bin/env bash
# =============================================================================
# Few-shot CoT evaluation of BASE models (paper protocol for base models:
# GSM8K 8-shot, MATH/MATH500/OlympiadBench/SAT-MATH/MMLU-STEM/OCW 4-shot,
# AMC23 0-shot, MathQA/MAWPS/SVAMP/ASDiv 8-shot -- all wired in
# configs/en_math_cot_few_shot_test_configs.json).
#
# One vLLM instance per GPU, TP=1; benchmarks are sharded across GPUs by
# run_subset_parallel.py.
#
# Usage:
#   conda activate /hdd2/xiaolirui/conda_envs/matheval
#   bash eval/scripts/run_fewshot_base.sh                          # default 3B base
#   MODEL_PATH=models/Llama-3.1-8B-Base MODEL_SIZE=8b bash eval/scripts/run_fewshot_base.sh
#   CUDA_VISIBLE_DEVICES=0,1,2 NGPUS=3 bash eval/scripts/run_fewshot_base.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

# MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/Llama-3.2-3B-Base}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/OctoThinker-3B-Long-Base}
# MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-3B-Base}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
MODEL_SIZE=${MODEL_SIZE:-3b}           # 1b | 3b | 7b | 8b (used for TP sizing; 3b/8b -> TP=1)
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
TEST_CONF=${TEST_CONF:-configs/en_math_cot_few_shot_test_configs_partial.json}
PROMPT_FORMAT=${PROMPT_FORMAT:-few_shot}

MODEL_NAME=$(basename "${MODEL_PATH%/}")
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${MODEL_NAME}}
SUMMARIZE_DIR=${SUMMARIZE_DIR:-outputs/summary}

export TOKENIZERS_PARALLELISM=false

echo "MODEL_PATH   = ${MODEL_PATH}"
echo "MODEL_SIZE   = ${MODEL_SIZE} | NGPUS = ${NGPUS} | TEMPERATURE = ${TEMPERATURE}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "TEST_CONF    = ${TEST_CONF} (prompt_format=${PROMPT_FORMAT})"

cd "${EVAL_DIR}"

python run_subset_parallel.py \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --ngpus "${NGPUS}" \
    --model-size "${MODEL_SIZE}" \
    --use-vllm \
    --load_in_half \
    --test-conf "${TEST_CONF}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --no-markup-question \
    --temperature "${TEMPERATURE}" \
    --n-repeats "${N_REPEATS}"

# Aggregate per-model metrics + predictions, and build a comparison CSV/PNG
# over everything under outputs/ (one row per model directory).
python summarize_results.py \
    --dirname outputs \
    --summarize_dir "${SUMMARIZE_DIR}"

echo "Done. Summary: ${EVAL_DIR}/${SUMMARIZE_DIR}/evaluation_results.csv (+ results.png)"
