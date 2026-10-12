# OctoThinker RL — Local Reproduction Notes (single node, 6x A100-40GB)

This directory contains the vendored verl 0.2.0.dev code (with the OctoThinker
custom patches: stair rollout scheduler, `orz_base` template, math-verifier
reward, dynamic easy-sample removal, sample dumping) plus scripts to reproduce
the paper's GRPO stage on one machine.

## 1. Environment

```bash
bash rl/env_setup.sh          # creates /hdd2/xiaolirui/conda_envs/verl (python 3.10)
conda activate /hdd2/xiaolirui/conda_envs/verl
```

Notes:

- verl is **not pip-installed**. Training runs `python3 -m verl.trainer.main_ppo`
  with `cwd=rl/` so the vendored package is picked up from `./verl`.
- torch 2.4.0 (cu121) + prebuilt flash-attn 2.7.0.post2 wheel are installed
  before the pinned `requirements.txt` (the unpinned `flash-attn` line in it is
  filtered out by the setup script).
- Smoke check after setup: `cd rl && python -c "import verl.trainer.main_ppo"`.

## 2. Data

`data/math.8k/{train,test}.parquet` is SimpleRL-Zoo MATH8K (8407 train / 272
val, where the last 18 val rows are `data_source == 'pad'` dummies). To drop
the pads into a cleaned validation file (recommended for val metrics):

```bash
python rl/scripts/clean_val_data.py     # writes data/math.8k/test.clean.parquet (254 rows)
```

Point `data.val_files` at `./data/math.8k/test.clean.parquet` if you want the
cleaned val set (the default script keeps the original to match the repo).

## 3. One-command RL

From a **released OctoThinker base** (e.g. Long branch, ~416 steps ≈ paper's 400):

```bash
MODEL_PATH=/hdd2/xiaolirui/OctoThinker/models/OctoThinker-3B-Long-Base \
bash rl/scripts/run_grpo_3b_local.sh
```

From a **mid-trained HF checkpoint** produced by the nanotron mid-training
stage (see `../REPRODUCTION.md`):

```bash
MODEL_PATH=/hdd2/xiaolirui/OctoThinker/outputs/midtrain/3b_long/checkpoints/hf/final \
BRANCH=long bash rl/scripts/run_grpo_3b_local.sh
```

Important knobs (all env vars):

| var | default | meaning |
|---|---|---|
| `MODEL_PATH` | `models/OctoThinker-3B-Long-Base` | policy init |
| `BRANCH` | `long` | `long` → progressive lengths `{0:2048, 200:4096, 320:8192}`; `short`/`hybrid` → fixed `{0:4096}` |
| `TEMPLATE` | `complex` | `complex` → `orz_base` ("Complex Template"); `simple` → `cpt_base` ("Simple Template", paper Sec 2.1 default for plain base Llama) |
| `N_GPUS` | `6` (or count of pre-set `CUDA_VISIBLE_DEVICES`) | GPUs for Ray |
| `TOTAL_STEPS` / `TOTAL_EPOCHS` | ~`13` epochs ≈ 400 steps | epochs win if both set |
| `OUTPUT_ROOT` | `/hdd2/xiaolirui/OctoThinker/outputs/rl/<branch>` | ckpts + samples root |
| `WANDB_API_KEY` | unset → console-only logging | set to also log to wandb |

Two GPUs on this node may be occupied by an unrelated vLLM server — check
`nvidia-smi` and, if needed, restrict the run, e.g.:

```bash
CUDA_VISIBLE_DEVICES=3,5 bash rl/scripts/run_grpo_3b_local.sh   # auto-detects N_GPUS=2
```

## 4. VRAM behavior on A100-40GB

- `gpu_memory_utilization=0.75` (the original 8x80GB templates used 0.8). Each
  GPU runs a colocated vLLM rollout worker (TP=1) + FSDP shard; ~30GB of model
  + optimizer states (offloaded) + activations leaves ~10GB for vLLM KV cache.
- Signs of vLLM OOM during rollout → lower to 0.6–0.7. Throughput-limited →
  raise `max_num_seqs` (default 1024; try 2048) — passed via
  `actor_rollout_ref.rollout.max_num_seqs=2048` as an extra trailing arg.
- Long-branch runs *start* at 2048 max response length, so peak rollout memory
  only appears after step 320; short/hybrid runs sit at 4096 from step 0.

## 5. Monitoring

- Console logs print `step/... reward/... response_length/...` every step
  (logger=`['console']` when `WANDB_API_KEY` is unset).
- Raw rollout samples are dumped every step to
  `$OUTPUT_ROOT/samples/<run_name>/train/step_*.json` (`trainer.samples_save_path`).
- Checkpoints land in `$OUTPUT_ROOT/ckpts/<run_name>` (`save_freq=10`,
  `resume_mode=auto`); validation runs every `test_freq=4` steps.

## 6. Mapping to the paper (Sec 2.1 / Sec 5)

| paper | this script |
|---|---|
| verl GRPO, MATH8K prompts, lr 1e-6, temp 1.0, KL coef 0 (`low_var_kl`) | identical |
| global batch 128 prompts × 16 rollouts | `train_batch_size=256`, `n=16` (repo default; 256 = 2× paper) |
| PPO mini-batch 64 | `ppo_mini_batch_size=16` (repo default; paper numbers noted in script header) |
| 400 RL steps | default `TOTAL_EPOCHS=13` ≈ 416 steps (32 steps/epoch) |
| long-CoT progressive max-response 2048/4096/8192 | `length_scheduler_dict="{0: 2048, 200: 4096, 320: 8192}"` for `BRANCH=long` |
| "Complex Template" (orz_base) for later experiments | `TEMPLATE=complex` (default); `TEMPLATE=simple` for the plain `Question:/Answer:` template used in the paper's early/base-Llama runs |
| RL-tuned models are evaluated zero-shot with the RL prompt | see `../eval/README_LOCAL.md` and `eval/scripts/run_zeroshot_rl.sh` |
