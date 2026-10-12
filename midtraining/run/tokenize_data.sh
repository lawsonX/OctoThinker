#!/usr/bin/env bash
# =============================================================================
# tokenize_data.sh — tokenize the prepared JSONL files with nanotron's
# tools/preprocess_data.py (datatrove DocumentTokenizer).
#
# For each dataset this produces, under midtraining/data/tokenized/<Name>/:
#   *.ds            tokenized documents
#   *.ds.index      document bounds
#   *.ds.metadata   token counts / tokenizer info
#
# Prerequisites:
#   * env from midtraining/env_setup.sh (datatrove installed)
#   * prepare_data.py already run (midtraining/data/jsonl/<Name>/data.jsonl)
#   * a local Llama-3 tokenizer (models/Llama-3.2-3B-Base) — see models/download_models.sh
#
# Usage:
#   bash tokenize_data.sh                         # all 8 datasets, 16 tasks
#   bash tokenize_data.sh --datasets MegaMathQA,OpenR1Math220K
#   bash tokenize_data.sh --n-tasks 32
#   TOKENIZER=/path/to/tokenizer bash tokenize_data.sh
#   EOS_TOKEN="</s>" bash tokenize_data.sh        # append EOS after each doc (default: none)
#
# nohup-friendly:
#   nohup bash tokenize_data.sh > tokenize_all.log 2>&1 &
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname "$0")/../.." >/dev/null 2>&1 && pwd -P)"
NANOTRON_DIR="$REPO_ROOT/midtraining/nanotron"
JSONL_ROOT="${JSONL_ROOT:-$REPO_ROOT/midtraining/data/jsonl}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-$REPO_ROOT/midtraining/data/tokenized}"
TOKENIZER="${TOKENIZER:-$REPO_ROOT/models/Llama-3.2-3B-Base}"
EOS_TOKEN="${EOS_TOKEN:-}"
LOG_DIR="$TOKENIZED_ROOT/logs"
ENV_PREFIX="${ENV_PREFIX:-/hdd2/xiaolirui/conda_envs/nanotron}"
N_TASKS=16
DATASETS=""

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --n-tasks)  N_TASKS="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# canonical dataset names produced by prepare_data.py
ALL_DATASETS="FineMath4Plus MegaMathWebPro MegaMathWebProMax MegaMathQA OpenR1Math220K TULU3SFT UltraChat220K WildChat1M"
[ -n "$DATASETS" ] || DATASETS="$ALL_DATASETS"

if [ -x "$ENV_PREFIX/bin/python" ]; then
    PY="$ENV_PREFIX/bin/python"
else
    echo "WARNING: $ENV_PREFIX not found, falling back to 'python3' on PATH" >&2
    PY="python3"
fi

[ -d "$NANOTRON_DIR/tools" ] || { echo "ERROR: nanotron not found at $NANOTRON_DIR" >&2; exit 1; }
if [ ! -d "$TOKENIZER" ]; then
    echo "WARNING: tokenizer dir $TOKENIZER does not exist yet." >&2
    echo "         Run models/download_models.sh first ( Llama-3.2-3B-Base )." >&2
    echo "         Continuing anyway — preprocess_data.py will fail until it exists." >&2
fi
# datatrove's load_tokenizer uses tokenizers.Tokenizer.from_file(), which needs
# the tokenizer FILE, not a HF model dir — resolve to tokenizer.json when needed.
if [ -d "$TOKENIZER" ] && [ -f "$TOKENIZER/tokenizer.json" ]; then
    TOKENIZER_FILE="$TOKENIZER/tokenizer.json"
else
    TOKENIZER_FILE="$TOKENIZER"
fi

mkdir -p "$LOG_DIR"
# make `import datatrove` (and nanotron, if needed) resolvable for the tool
export PYTHONPATH="$NANOTRON_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "==> python           : $PY"
echo "==> tokenizer        : $TOKENIZER"
echo "==> jsonl root       : $JSONL_ROOT"
echo "==> tokenized root   : $TOKENIZED_ROOT"
echo "==> n-tasks          : $N_TASKS"
echo "==> datasets         : $DATASETS"
[ -n "$EOS_TOKEN" ] && echo "==> eos token        : $EOS_TOKEN"

FAILURES=()
for NAME in $(echo "$DATASETS" | tr ',' ' '); do
    IN_DIR="$JSONL_ROOT/$NAME"
    OUT_DIR="$TOKENIZED_ROOT/$NAME"
    LOG_FILE="$LOG_DIR/$NAME.log"
    if [ ! -d "$IN_DIR" ] || ! ls "$IN_DIR"/*.jsonl >/dev/null 2>&1; then
        echo "==> [skip] $NAME: no jsonl under $IN_DIR (run prepare_data.py for it first)" >&2
        FAILURES+=("$NAME")
        continue
    fi
    echo "==> [tokenize] $NAME -> $OUT_DIR (log: $LOG_FILE)"
    mkdir -p "$OUT_DIR"
    EOS_ARGS=()
    [ -n "$EOS_TOKEN" ] && EOS_ARGS=(--eos-token "$EOS_TOKEN")
    (cd "$NANOTRON_DIR" && "$PY" -u tools/preprocess_data.py \
        --tokenizer-name-or-path "$TOKENIZER_FILE" \
        --output-folder "$OUT_DIR" \
        --logging-dir "$LOG_DIR/datatrove_$NAME" \
        --n-tasks "$N_TASKS" \
        "${EOS_ARGS[@]}" \
        jsonl --dataset "$IN_DIR" --column text --glob-pattern "*.jsonl") >"$LOG_FILE" 2>&1
    RC=$?
    if [ $RC -ne 0 ] || ! ls "$OUT_DIR"/*.ds >/dev/null 2>&1; then
        echo "    FAILED (rc=$RC) — see $LOG_FILE" >&2
        FAILURES+=("$NAME")
    else
        N_FILES=$(ls "$OUT_DIR"/*.ds 2>/dev/null | wc -l)
        echo "    OK ($N_FILES .ds file(s))"
    fi
done

echo ""
if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "datasets not tokenized: ${FAILURES[*]}" >&2
    exit 1
fi
echo "all datasets tokenized."
