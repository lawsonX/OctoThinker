#!/usr/bin/env bash
# =============================================================================
# Single-node GRPO training for OctoThinker (3B) on 6x A100-40GB.
#
# This is a LOCAL, single-node adaptation of the original 8-GPU multi-node
# templates rl/scripts/llama_3b_long.sh and rl/scripts/llama_3b_hybrid.sh
# (those are kept untouched). Run from anywhere; the trainer runs with cwd=rl/
# because verl is vendored (not pip-installed).
#
# Paper (arXiv:2506.20512) Sec 2.1 recipe vs. this script (repo defaults kept):
#   global batch        : paper 128 prompts   | here train_batch_size=256 (repo default)
#   rollouts per prompt : 16                  | here n=16 (same)
#   PPO mini-batch      : paper 64            | here ppo_mini_batch_size=16 (repo default;
#                                               in this verl version that is already
#                                               global_batch*16/... -- paper's "2:1
#                                               sampling:update ratio" maps onto these
#                                               repo knobs; kept as-is intentionally)
#   lr 1e-6, temperature 1.0, top_p 1, KL coef 0 (low_var_kl) -- identical.
#   Paper figures show 400 RL steps; repo templates use total_epochs=100 (~3280
#   steps). This script defaults to ~13 epochs ~= 400-430 steps (8407 prompts /
#   256 per batch = 32 steps/epoch); override TOTAL_STEPS / TOTAL_EPOCHS as needed.
#
# Long branch (Sec 3.4 progressive max-response-length):
#   {0: 2048, 200: 4096, 320: 8192} -- 2048 for steps 0-199, 4096 for 200-319,
#   8192 from step 320. Short/hybrid branches: fixed {0: 4096}.
#
# Usage examples:
#   conda activate /hdd2/xiaolirui/conda_envs/verl
#   bash rl/scripts/run_grpo_3b_local.sh                          # long branch, 6 GPUs
#   BRANCH=short TOTAL_STEPS=400 bash rl/scripts/run_grpo_3b_local.sh
#   CUDA_VISIBLE_DEVICES=3,5 N_GPUS=2 MODEL_PATH=... bash rl/scripts/run_grpo_3b_local.sh
#   WANDB_API_KEY=... bash rl/scripts/run_grpo_3b_local.sh         # enable wandb
#   MONITOR=1 bash rl/scripts/run_grpo_3b_local.sh                 # + TB monitor
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # .../rl  (trainer cwd; verl is vendored here)

# ------------------------------ Configurable env -----------------------------
MODEL_PATH=${MODEL_PATH:-/hdd2/xiaolirui/OctoThinker/models/OctoThinker-3B-Long-Base}
# hydra changes cwd at job start, so every path handed to the trainer must be
# absolute — resolve a user-supplied relative MODEL_PATH against the INVOKER's
# cwd (before this script cd's into rl/).
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" 2>/dev/null && pwd)/$(basename "$MODEL_PATH")"
[ -d "$MODEL_PATH" ] || { echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }
BRANCH=${BRANCH:-long}                 # long | short | hybrid
TEMPLATE=${TEMPLATE:-complex}          # complex (=orz_base, paper default for later exps)
                                       # | simple (=cpt_base "Question:\n...\nAnswer:\n", paper Sec 2.1 base-Llama default)

# GPU selection: pre-set CUDA_VISIBLE_DEVICES wins if N_GPUS is not given
# explicitly (e.g. CUDA_VISIBLE_DEVICES=3,5 -> 2 GPUs). NOTE: this node often
# has GPUs occupied by other services -- always check `nvidia-smi` first.
if [ -z "${N_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        N_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
    else
        N_GPUS=6
    fi
fi

STEPS_PER_EPOCH=32   # floor(8407 train prompts / 256 batch); used for the default below
if [ -z "${TOTAL_EPOCHS:-}" ]; then
    if [ -n "${TOTAL_STEPS:-}" ]; then
        TOTAL_EPOCHS=$(( (TOTAL_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
    else
        TOTAL_EPOCHS=13   # ~416 steps ~= paper's 400 RL steps
    fi
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/hdd2/xiaolirui/OctoThinker/outputs/rl/${BRANCH}}
WANDB_MODE=${WANDB_MODE:-}             # unused placeholder; wandb is keyed off WANDB_API_KEY below

# Length schedule per branch (paper Sec 3.4 for long)
if [ "${BRANCH}" = "long" ]; then
    LENGTH_SCHEDULER_DICT="{0: 2048, 200: 4096, 320: 8192}"
else
    LENGTH_SCHEDULER_DICT="{0: 4096}"
fi

# Template type: 'orz_base' ("Complex Template") or 'cpt_base' ("Simple Template")
if [ "${TEMPLATE}" = "complex" ]; then
    TEMPLATE_TYPE=orz_base
elif [ "${TEMPLATE}" = "simple" ]; then
    TEMPLATE_TYPE=cpt_base
else
    echo "ERROR: TEMPLATE must be 'complex' or 'simple', got '${TEMPLATE}'" >&2; exit 1
fi

# Logger: wandb only when an API key is present, otherwise console-only
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOGGER="['console','wandb']"
else
    LOGGER="['console']"
fi

RUN_NAME=grpo_octothinker_3b_${BRANCH}_math8k_$(TZ='UTC-8' date "+%m%d_%H%M")
CKPT_DIR=${OUTPUT_ROOT}/ckpts/${RUN_NAME}
SAMPLES_DIR=${OUTPUT_ROOT}/samples/${RUN_NAME}
mkdir -p "${CKPT_DIR}" "${SAMPLES_DIR}"

echo "RUN_NAME          = ${RUN_NAME}"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "BRANCH/TEMPLATE   = ${BRANCH} / ${TEMPLATE} (${TEMPLATE_TYPE})"
echo "LENGTH_SCHEDULER  = ${LENGTH_SCHEDULER_DICT}"
echo "N_GPUS (visible)  = ${N_GPUS} (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset, all>}')"
echo "TOTAL_EPOCHS      = ${TOTAL_EPOCHS} (~$(( TOTAL_EPOCHS * STEPS_PER_EPOCH )) steps @ ${STEPS_PER_EPOCH}/epoch)"
echo "OUTPUT_ROOT       = ${OUTPUT_ROOT}"
echo "LOGGER            = ${LOGGER}"

# ------------------------------ Ray cluster ----------------------------------
# Kill any leftover local ray, then start a single-node head. --num-gpus tells
# ray how many GPUs the trainer may allocate; CUDA_VISIBLE_DEVICES (if set)
# restricts which physical GPUs are visible.
cd "${RL_DIR}"   # BEFORE ray start: ray workers inherit this cwd (data paths are relative)
ray stop --force || true
sleep 3
RAY_PORT="${RAY_PORT:-$((26379 + RANDOM % 1000))}"
ray start --head --port="${RAY_PORT}" --num-gpus "${N_GPUS}" --disable-usage-stats || { echo "ERROR: ray start failed" >&2; exit 1; }
sleep 2
ray status || { echo "ERROR: ray cluster not healthy" >&2; exit 1; }
# Make the trainer attach to the just-started cluster instead of spawning one.
export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:${RAY_PORT}}

cd "${RL_DIR}"   # verl is imported from ./verl

# ------------------------------ Opt-in training monitor ----------------------
# MONITOR=1: duplicate trainer stdout to a log file and tail it into
# TensorBoard events via monitor/log2tb.py (base python; runs outside the
# verl env and never touches the training processes). View with
# `bash monitor/serve.sh` (http://<host>:6006). Default: unchanged behavior.
if [ "${MONITOR:-0}" = "1" ]; then
    REPO_ROOT="$(cd "${RL_DIR}/.." && pwd)"
    LOG_DIR="${OUTPUT_ROOT}/logs"
    TB_DIR="${REPO_ROOT}/outputs/tb/${RUN_NAME}"
    mkdir -p "${LOG_DIR}" "${TB_DIR}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    echo "LOG_FILE          = ${LOG_FILE}"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    MONITOR_PY="${MONITOR_PY:-/home/lawson/miniforge3/bin/python3}"
    "${MONITOR_PY}" "${REPO_ROOT}/monitor/log2tb.py" --log "${LOG_FILE}" \
        --tb-dir "${TB_DIR}" --format verl \
        >> "${TB_DIR}/monitor.out" 2>&1 &
    MONITOR_PID=$!
    _kill_monitor() { kill "${MONITOR_PID}" 2>/dev/null || true; }
    trap _kill_monitor EXIT
    trap '_kill_monitor; exit 143' TERM
    echo "MONITOR           = TB dir ${TB_DIR} (serve: bash ${REPO_ROOT}/monitor/serve.sh)"
fi

# ------------------------------ Launch training ------------------------------
# Background + wait so trapped signals run promptly (a trap cannot fire while
# bash blocks on a foreground child); `set -e` still propagates the exit code.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${RL_DIR}/data/math.8k/train.parquet" \
    data.val_files="${RL_DIR}/data/math.8k/test.parquet" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-256} \
    data.val_batch_size=${VAL_BATCH_SIZE:-2048} \
    data.shuffle=True \
    data.max_prompt_length=400 \
    data.template_type=${TEMPLATE_TYPE} \
    data.filter_overlong_prompts=True \
    data.max_response_length=8192 \
    data.dynamic_remove_easy_sample=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.scheduler=stair \
    actor_rollout_ref.rollout.length_scheduler_dict="${LENGTH_SCHEDULER_DICT}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0 \
    trainer.logger=${LOGGER} \
    trainer.project_name='ct_rl' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=4 \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.resume_mode=auto \
    trainer.samples_save_path=${SAMPLES_DIR} \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"

# NOTE on VRAM (A100-PCIE-40GB): gpu_memory_utilization=0.75 leaves ~10GB on the
# vLLM GPUs for the colocated FSDP training pass; the original templates used
# 0.8 on 80GB cards. If vLLM OOMs during rollout, lower it (0.6-0.7); if vLLM
# is throughput-limited, raise max_num_seqs (default 1024; try 2048) or lower
# gpu_memory_utilization to leave more room. With 2 GPUs busy elsewhere, run
# with CUDA_VISIBLE_DEVICES=3,5 N_GPUS=2 (batch sizes above were tuned for the
# full node; scale train_batch_size down, e.g. 64, for smoke tests).
