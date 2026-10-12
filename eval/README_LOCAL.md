# OctoThinker Evaluation — Local Reproduction Notes (single node, 6x A100-40GB)

## 1. Environment

```bash
bash eval/env_setup.sh          # creates /hdd2/xiaolirui/conda_envs/matheval (python 3.10)
conda activate /hdd2/xiaolirui/conda_envs/matheval
```

- torch 2.4.0 (cu121) + pinned `requirements.txt`, with **math-verify pinned to
  0.5.2** (same as the RL reward; the shipped requirements line is unpinned and
  is rewritten by the setup script).
- flash-attn is **not** installed on purpose: inference goes through vLLM,
  which does not need flash-attn on A100.
- Benchmark data already lives in `eval/datasets/` (13 benchmarks incl.
  `math500/test.jsonl`, 500 rows).

## 2. How prompts are assembled (and what changed for zero-shot)

`run_subset_parallel.py` shards benchmarks across GPUs (one vLLM instance per
GPU, TP=1) and invokes `infer/run_cot_eval.py`:

- **Few-shot (base models)**: `--prompt-format few_shot` makes the harness read
  the per-benchmark `few_shot_prompt` class from the config, e.g.
  `MinervaMathPrompt`; `run_cot_eval.py` builds
  `few_shot_examples + "Problem:\n" + question + "\n\nSolution:"` and uses the
  class's stop words (`\nProblem:`). With `--no-markup-question`, no extra
  instruction line is appended to the question.
- **Zero-shot (RL-tuned models) — new**: config entries set
  `"few_shot_prompt": "none"` and a `"custom_prompt"` that reproduces the
  orz_base "Complex Template" from RL training (see
  `rl/verl/utils/dataset/rl_dataset.py`); `[PROMPT]` in it is replaced by the
  question. Minimal code changes (default behavior unchanged when absent):
  - `infer/run_cot_eval.py`: `--system_prompt` arg + `few_shot_prompt="none"`
    zero-shot assembly.
  - `run_subset_parallel.py`: forwards the config's `custom_prompt` (or a
    `--system-prompt` CLI override) via `--system_prompt`; the subprocess
    command is now built as a list so prompts with spaces are not mangled.
- **Answer extraction**: zero-shot reuses the few-shot extractors
  (`extract_math_few_shot_cot_answer` etc.), which are boxed-first already
  (`extract_answer` checks `\boxed{}` before the "final answer is" phrasing), so
  no few-shot-specific phrasing is required. Caveat: olympiad-bench's
  `extract_olympiad_math_answer` looks for a "Final Answer:" marker; RL outputs
  end with `\boxed{...}` which math-verify still parses inside the wrapped
  string, but if olympiad zero-shot scores look depressed, switch that entry's
  `answer_extraction_fn` to `extract_math_few_shot_cot_answer`.

## 3. Running

```bash
# Base models — few-shot, 13 benchmarks (paper protocol):
MODEL_PATH=models/Llama-3.2-3B-Base bash eval/scripts/run_fewshot_base.sh

# RL-tuned models — zero-shot with the RL template, greedy decoding:
MODEL_PATH=models/OctoThinker-3B-Long-Zero bash eval/scripts/run_zeroshot_rl.sh
```

Both wrappers respect a pre-set `CUDA_VISIBLE_DEVICES` (check `nvidia-smi`
first — two GPUs may be busy with an unrelated vLLM server):

```bash
CUDA_VISIBLE_DEVICES=3,5 MODEL_PATH=models/Llama-3.1-8B-Base MODEL_SIZE=8b \
    bash eval/scripts/run_fewshot_base.sh
```

Outputs land in `eval/outputs/<model-name>/`; aggregation runs
`summarize_results.py`, producing `outputs/summary/evaluation_results.csv`
(one row per model dir), per-benchmark `metrics.json`/`predictions.json`, and
`results.png` (headless save; plotting failures only warn, JSON/CSV stay
authoritative).

Few-shot shot counts wired in `configs/en_math_cot_few_shot_test_configs.json`
(match the paper): GSM8K 8-shot, MATH/MATH500/OlympiadBench/SAT-MATH/
MMLU-STEM/OCW 4-shot, AMC23 0-shot, MathQA/MAWPS/SVAMP/ASDiv 8-shot.

## 4. Legacy script

`scripts/en_math_cot_eval_last4dir.sh` still says `n_gpus=8` and calls
`submit_eval_jobs.py` (not present in this repo — it belonged to the original
Slurm launcher). It is left untouched for reference; on this 6-GPU node use the
wrappers above (`NGPUS=6` default) instead of editing it.
