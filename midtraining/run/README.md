# OctoThinker mid-training reproduction (nanotron)

Reproduction assets for the mid-training recipes of the OctoThinker paper
(arXiv:2506.20512, Sections 3, 3.1, 3.3, 3.4 and 4) on a **single node with
6x NVIDIA A100-PCIE-40GB and no Slurm**, using the pristine nanotron 0.4
snapshot in `midtraining/nanotron/` (do not modify it).

Pipeline: **env → jsonl → tokenize → convert base → train → export HF → rl/**.

All scripts are idempotent and cheap to re-run. Nothing here installs packages
or downloads data/models by itself — the heavy steps are explicit commands you
run when ready.

---

## 0. Layout

| path | what |
|---|---|
| `../env_setup.sh` | create the conda env (`/hdd2/xiaolirui/conda_envs/nanotron`, python 3.10) with torch 2.4.0+cu121, flash-attn 2.7.0.post2 prebuilt wheel, nanotron editable (`[nanosets]`), `transformers==4.39.3` |
| `prepare_data.py` | raw uploads → one `{"text": ...}` jsonl per dataset (schema-sniffing, paper Section-3 formatting) |
| `tokenize_data.sh` | jsonl → nanotron `.ds` tokenized datasets (datatrove) |
| `convert_hf_to_nanotron.sh` | HF base weights → nanotron format (`ExistingCheckpointInit`) |
| `convert_nanotron_to_hf.sh` | nanotron checkpoint → HF format (for `rl/` or release) |
| `run_midtraining.sh` | launcher: derives world size from config, sets NCCL env, `tee` logs, `RESUME=1` support |
| `configs/` | verified training YAMLs (see table below) |
| `../../midtraining/data/` | prepared jsonl + tokenized data (created by steps 2–3) |
| `../../outputs/midtraining/` | checkpoints, HF exports, logs (created by training) |

Config files:

| config | purpose |
|---|---|
| `llama-3.2-1b.yaml` / `llama-3.2-3b.yaml` / `llama-3.1-8b.yaml` | full model skeletons (architecture + parallelism + optimizer), Section-4 stable-stage LRs (5e-5 / 2e-5 / 1e-5) |
| `smoke_3b_tiny.yaml` | **20 steps**, pipeline validation — run this first |
| `ablation_3b_20B_cosine.yaml` | Section 3.1 default recipe, 20B tokens, cosine 3e-5→3e-6, web:QA 9:1 |
| `scale_3b_100B_pro_max.yaml` | Section 3.1 scale-up, 100B tokens on MegaMath-Web-Pro-Max, checkpoints ≈20B/70B/100B |
| `two_stage_stable_3b.yaml` | Section 4 stage 1: 200B tokens, constant LR 2e-5, paper Table-3 mixture |
| `two_stage_decay_3b_long.yaml` / `_short.yaml` / `_hybrid.yaml` | Section 4 stage 2: 5000 steps, cosine decay to 10%, the three branch mixtures |

Datasets (8 uploaded; canonical names used throughout):

| canonical name | source dir under `/hdd2/xiaolirui/octoThinker-Datasets` | role |
|---|---|---|
| `FineMath4Plus` | `FineMath-4plus` | web |
| `MegaMathWebPro` | `MegaMath-Web-Pro` | web |
| `MegaMathWebProMax` | `MegaMath-Web-Pro-Max` | web (100B / stable stage) |
| `MegaMathQA` | `MegaMath-QA` | QA |
| `OpenR1Math220K` | `OpenR1-Math-220K` | long-CoT QA |
| `TULU3SFT` | `TULU3-sft` (persona-IF subset) | instruction |
| `UltraChat220K` | `UltraChat-220K` | instruction |
| `WildChat1M` | `WildChat-1M` (GPT-4 only) | instruction |

NOT uploaded (optional placeholders, commented in the stage-1/stage-2
configs — the core Section-3 ablations do **not** need them): DCLM-Baseline,
MegaMath-Code, MegaMath Trans. Code, MegaMath Text Code Block,
OpenMathInstruct2, NuminaMath1.5, AM-DeepSeek-Distilled-40M.

---

## 1. Environment

```bash
bash ../env_setup.sh            # ~10-20 min (mostly flash-attn wheel + datatrove)
conda activate /hdd2/xiaolirui/conda_envs/nanotron
```

Note on the `fast-modeling` extra: `midtraining/nanotron/pyproject.toml` pins
`flash-attn>=2.5.0,<2.7.0` there, which conflicts with the required prebuilt
2.7.0.post2 wheel (the extra's only content). nanotron's llama modeling only
uses stable flash-attn APIs (`flash_attn_varlen_func`, `flash_attn_with_kvcache`,
`bert_padding`, `flash_attn.layers.rotary`), so the wheel supersedes the extra;
`env_setup.sh` installs nanotron `--no-deps` + explicit base deps and documents
this inline.

## 2. Prepare jsonl

```bash
python prepare_data.py                                  # all 8 datasets
python prepare_data.py --datasets MegaMathQA,OpenR1Math220K
python prepare_data.py --max-samples-per-dataset 2000   # smoke subset
```

Output: `../../midtraining/data/jsonl/<Name>/data.jsonl` (UTF-8, one
`{"text": ...}` per line) + `summary.json` with row counts and ≈token counts
(chars/4). The uploader may still be running: unreadable/truncated files are
skipped with warnings; a dataset fails loudly with a schema report only when
nothing usable could be read. Re-run the command when the upload finishes.

Formatting follows paper Section 3: OpenR1 = `problem\n<think>…</think>`;
instruction data = `User:{…}\nAssistant:{…}` per turn; WildChat keeps GPT-4
conversations only; web docs < 50 chars dropped; QA questions < 100 chars
dropped. `prepare_data.py` sniffs columns from the actual files (multiple
candidate names per dataset) rather than assuming one fixed schema.

## 3. Tokenize

```bash
bash tokenize_data.sh                       # all available jsonl dirs, 16 workers
bash tokenize_data.sh --datasets MegaMathQA --n-tasks 32
nohup bash tokenize_data.sh > tokenize.log 2>&1 &   # large datasets
```

Output: `../../midtraining/data/tokenized/<Name>/` with `*.ds`, `*.ds.index`,
`*.ds.metadata`, using tokenizer `models/Llama-3.2-3B-Base` (override with
`TOKENIZER=...`). Any Llama-3-family tokenizer works — the vocab is shared.
`EOS_TOKEN="</s>"` can be passed to append EOS after each document; nanotron's
Nanoset concatenates documents, so the default is none.

## 4. Convert the base model

```bash
bash convert_hf_to_nanotron.sh \
    ../../models/Llama-3.2-3B-Base \
    ../../outputs/midtraining/converted/Llama-3.2-3B-Base-nanotron
```

(1B/8B analogously.) The configs already point `model.init_method.path` at
these locations (`ExistingCheckpointInit` = weights only, no optimizer/LR
state — see `src/nanotron/trainer.py`). Download bases first with
`models/download_models.sh` if not present.

## 5. Smoke test (do this first)

```bash
# tiny tokenized dataset (2k docs is plenty for 20 steps)
python prepare_data.py --datasets MegaMathWebPro --max-samples-per-dataset 2000 \
    --output-root ../../midtraining/data/smoke_jsonl
JSONL_ROOT=../../midtraining/data/smoke_jsonl bash tokenize_data.sh --datasets MegaMathWebPro
mkdir -p ../../midtraining/data/tokenized && \
    mv ../../midtraining/data/tokenized/MegaMathWebPro ../../midtraining/data/tokenized/SmokeTiny

CONFIG=configs/smoke_3b_tiny.yaml bash run_midtraining.sh
```

~20 steps validate data loading, dp=6 launch, checkpointing and give you a
checkpoint to test the export on. Expected wall time: **20–40 min** including
startup/compile.

## 6. Train the 20B ablation (Section 3.1)

```bash
CONFIG=configs/ablation_3b_20B_cosine.yaml bash run_midtraining.sh
# resume after an interruption:
RESUME=1 CONFIG=configs/ablation_3b_20B_cosine.yaml bash run_midtraining.sh
```

Section 3.3 ablations (web:QA 9:1 with different QA sources) and the Section
3.4 89:10:1 mixture are pre-built as commented blocks inside
`configs/ablation_3b_20B_cosine.yaml`.

## 7. Export to HF and hand off to rl/

```bash
bash convert_nanotron_to_hf.sh \
    ../../outputs/midtraining/ablation_3b_20B_cosine/checkpoints/4624 \
    ../../outputs/midtraining/hf/ablation_3b_20B_cosine \
    ../../models/Llama-3.2-3B-Base
```

Then point the RL stage at the HF dir, e.g. adapt `rl/scripts/llama_3b_hybrid.sh`
(`policy_path=...`). The two-stage runs export the same way from
`two_stage_decay_3b_{long,short,hybrid}/checkpoints/5000`.

---

## Wall-time estimates (6x A100-40GB, 3B, seq 8192, bf16, flash-attn)

Assumptions: dp6/tp1/pp1, micro-batch 8, ~8–12k tokens/s/GPU at seq 8192
(aggregate 48–72k tokens/s) → **60–90 s per 528-seq step**
(4,325,376 tokens). Your throughput will print from step 1; rescale linearly.

| run | steps | tokens | est. wall time |
|---|---|---|---|
| smoke | 20 | 0.09B | 20–40 min |
| 20B ablation | 4,624 | 20.0B | **3.5–5 days** |
| 100B scale-up | 23,120 | 100.0B | 2.5–4 weeks |
| stage-1 stable (50k steps) | 50,000 | 216B* | **5–7 weeks** |
| stage-2 decay | 5,000 | 21.6B* | 3.5–5 days |

\* at the 528-seq batch; see deviations below. Recommendation: smoke first,
then the 20B ablation; only commit to stage-1 after the recipe is validated.

---

## Deviations from the paper (explicit)

1. **Global batch 528 vs 512 seqs.** 512 is not divisible by dp=6. Chosen:
   mbs 8 × accum 11 × dp 6 = 528 seqs = 4,325,376 tokens/step (+3.1% over the
   paper's 4M). With 8 GPUs you can get the exact 512 (mbs 8 × accum 8 × dp 8).
   The alternative mbs 8 × accum 8 × dp 6 = 384 seqs (−25%) was rejected.
2. **Step counts are recomputed** from the actual batch: 20B → 4624 steps,
   100B → 23120, exact-200B → 46240. Stage 1 keeps the paper's 50,000 steps
   (Table 3), which is ≈216B tokens at batch 528 — flagged in the config.
   Stage 2 keeps 5000 steps (≈21.6B at 528 vs "≈20B" in the paper).
3. **Configs are reconstructions.** The original `midtraining/nanotron-runner`
   submodule is an empty dangling gitlink (no `.gitmodules`, no content), so
   the YAMLs were rebuilt from the paper + nanotron 0.4 config dataclasses +
   examples. Every key was verified against `src/nanotron/config/*.py`;
   `rope_interleaved: false` is kept for HF-compatible weight conversion.
4. **Mixture dict form.** `docs/nanoset.md` advertises `{path: weight}` under
   `dataset_folder`, but this snapshot's `NanosetDatasetsArgs` parses
   `dataset_folder` (list) + `dataset_weights` (list) and its dacite parsing is
   strict — so all configs use the explicit list form (functionally identical;
   Nanoset normalizes weights either way).
5. **Instruction sub-split.** The paper's 0.10 "Instruction" share is split
   equally over TULU3-sft/WildChat/UltraChat (1/120 each in the 89:10:1
   mixture) — the paper gives no sub-split.
6. **rope_scaling.** nanotron's `llama.py` ignores `rope_scaling` (only
   `rope_theta`/`rope_interleaved` are used), and training is at seq 8192, so
   configs set `max_position_embeddings: 8192`, `rope_scaling: null`. The HF
   exports inherit these values; fine for ≤8192-context evals.
7. **TULU3 persona-IF filter** is heuristic: if the upload carries a
   `source`-like column containing "persona", only those rows are kept.
8. **8B skeleton runs a reduced batch** (264 seqs: mbs 2 × accum 44 × dp 3,
   tp 2, `recompute_layer: true`, `zero_stage: 1` = ZeRO-1) to fit 40GB cards.

## Notes

- Resume: `RESUME=1` copies the config to a temp YAML with
  `checkpoints.resume_checkpoint_path` = the run's `checkpoints_path`
  (nanotron reads `latest.txt` there). Optimizer + LR state are restored.
- Stage-2 starts use `model.init_method.path` (weights only) on purpose — a
  WSD decay stage starts from stable weights with a *fresh* optimizer.
- Data upload status at writing time: UltraChat, WildChat, OpenR1 (partial)
  present; FineMath/MegaMath-web/Pro-Max/MegaMath-QA/TULU3 dirs still empty or
  partial. `prepare_data.py` will say exactly what is missing; just re-run it
  after the upload completes.
