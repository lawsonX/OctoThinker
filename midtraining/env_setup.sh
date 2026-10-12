#!/usr/bin/env bash
# =============================================================================
# env_setup.sh — create the conda env for OctoThinker mid-training (nanotron)
#
# Idempotent: every step is safe to re-run; conda create is skipped when the
# prefix already exists and pip steps are naturally idempotent.
#
# Target: single node, 6x NVIDIA A100-PCIE-40GB, no Slurm.
#
# Usage:
#   bash midtraining/env_setup.sh
#
# Environment overrides:
#   SKIP_CONDA_CREATE=1   only run the pip steps inside the existing prefix
# =============================================================================
set -euo pipefail

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/hdd2/xiaolirui/conda_envs/nanotron}"
REPO_ROOT="$(cd -- "$(dirname "$0")/.." >/dev/null 2>&1 && pwd -P)"
NANOTRON_DIR="$REPO_ROOT/midtraining/nanotron"

# This machine sits behind a SANGFOR TLS-intercepting proxy: pypi.nvidia.com and
# download.pytorch.org return EMPTY/CORRUPT payloads (hash mismatches), and the
# internal artifactory mirror only caches RECENT package versions. The Tsinghua
# tuna PyPI mirror is reachable, fast, and mirrors ALL historical versions
# (incl. pre-releases like hydra-core 1.4.0.dev1) — use it for every pip step.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_VERSION="2.4.0"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
TRANSFORMERS_PIN="4.39.3"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10

echo "==> conda env prefix : $CONDA_ENV_PREFIX"
echo "==> nanotron source  : $NANOTRON_DIR"
echo "==> python           : 3.10 (per $NANOTRON_DIR/pyproject.toml: requires-python ~=3.10)"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found on PATH." >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. conda env (python 3.10)
# ---------------------------------------------------------------------------
if [ "${SKIP_CONDA_CREATE:-0}" != "1" ]; then
    if [ -x "$CONDA_ENV_PREFIX/bin/python" ]; then
        echo "==> [skip] conda env already exists at $CONDA_ENV_PREFIX"
    else
        echo "==> [conda create] $CONDA_ENV_PREFIX"
        conda create --prefix "$CONDA_ENV_PREFIX" python=3.10 -y
    fi
else
    echo "==> [skip] SKIP_CONDA_CREATE=1"
fi

PY="$CONDA_ENV_PREFIX/bin/python"
PIP="$CONDA_ENV_PREFIX/bin/python -m pip"
[ -x "$PY" ] || { echo "ERROR: no python at $PY" >&2; exit 1; }
"$PY" --version

# pip retry helper: the shared egress here is slow and mid-stream read errors
# are common; pip's cache makes retries cheap. Cache is ON (--no-cache-dir
# removed) so interrupted 800MB wheels resume from cache.
pip_retry() {
    local attempt
    for attempt in 1 2 3 4 5 6; do
        if $PIP install "$@"; then
            return 0
        fi
        echo "==> [pip_retry] attempt $attempt failed; retrying in 15s" >&2
        sleep 15
    done
    echo "==> [pip_retry] ALL ATTEMPTS FAILED: $*" >&2
    return 1
}

# ---------------------------------------------------------------------------
# 2. PyTorch 2.4.0 (cu121 wheels) — install FIRST so flash-attn and the
#    nanotron editable build compile/link against the right torch.
#    Prefer the shared wheelhouse (pre-downloaded with resumable curl); all of
#    torch 2.4.0's pinned nvidia-*-cu12 wheels live there, so this step is
#    fully offline when the wheelhouse is complete.
# ---------------------------------------------------------------------------
WHEELHOUSE="${WHEELHOUSE:-/hdd2/xiaolirui/wheelhouse}"
echo "==> [pip] torch==$TORCH_VERSION (wheelhouse first: $WHEELHOUSE)"
if [ -d "$WHEELHOUSE" ] && [ -n "$(ls "$WHEELHOUSE"/nvidia-*-py3-none.whl 2>/dev/null)" ]; then
    pip_retry --no-index --find-links "$WHEELHOUSE" "torch==$TORCH_VERSION"
else
    pip_retry "torch==$TORCH_VERSION"
fi

# ---------------------------------------------------------------------------
# 3. flash-attn 2.7.0.post2 prebuilt wheel (cp310, cu12, torch2.4).
#    NOTE on the `fast-modeling` extra: midtraining/nanotron/pyproject.toml
#    defines extras `nanosets` and `fast-modeling`, where fast-modeling pins
#    "flash-attn>=2.5.0,<2.7.0". That pin CONFLICTS with the required prebuilt
#    2.7.0.post2 wheel, and flash-attn is the extra's ONLY dependency. nanotron
#    only imports stable flash-attn APIs (flash_attn_varlen_func,
#    flash_attn_with_kvcache, flash_attn.bert_padding,
#    flash_attn.layers.rotary.RotaryEmbedding — see src/nanotron/models/llama.py)
#    which are unchanged in 2.7.x, so the wheel supersedes the extra. We
#    therefore install nanotron below with --no-deps and pull the base deps
#    explicitly, instead of resolving `.[fast-modeling]`.
# ---------------------------------------------------------------------------
echo "==> [pip] flash-attn prebuilt wheel (2.7.0.post2+cu12torch2.4)"
FA_WHEEL="$WHEELHOUSE/flash_attn-2.7.0.post2-cp310.whl"
if [ -f "$FA_WHEEL" ]; then
    pip_retry "$FA_WHEEL"
else
    pip_retry --no-deps "$FLASH_ATTN_WHEEL"
fi

# ---------------------------------------------------------------------------
# 4. nanotron (editable) + its base runtime deps from pyproject.toml
#    (torch already present; pyyaml/numpy<2/packaging/safetensors/dacite/tqdm/
#    datasets installed explicitly because of the --no-deps above).
# ---------------------------------------------------------------------------
echo "==> [pip] nanotron editable (no-deps) + base deps"
pip_retry --no-deps -e "$NANOTRON_DIR"
pip_retry "pyyaml" "numpy<2" "packaging" "safetensors" "dacite" "tqdm" "datasets"

# ---------------------------------------------------------------------------
# 5. `nanosets` extra: datatrove (git, needed by tools/preprocess_data.py),
#    numba, transformers. Installed as a second editable pass so pip resolves
#    ONLY the extra deps (nanotron itself is already installed).
# ---------------------------------------------------------------------------
echo "==> [pip] nanotron[nanosets] extra (datatrove@git, numba, transformers)"
pip_retry -e "$NANOTRON_DIR[nanosets]"

# ---------------------------------------------------------------------------
# 6. transformers pin per midtraining/nanotron/examples/llama/requirements.txt
# ---------------------------------------------------------------------------
echo "==> [pip] transformers==$TRANSFORMERS_PIN"
pip_retry "transformers==$TRANSFORMERS_PIN"

# ---------------------------------------------------------------------------
# 7. version report
# ---------------------------------------------------------------------------
echo ""
echo "================ installed versions ================"
"$PY" - <<'EOF'
import importlib
for mod in ["torch", "flash_attn", "transformers", "datatrove", "nanotron", "numpy", "datasets"]:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"{mod:15s} {ver}")
    except Exception as e:  # noqa: BLE001
        print(f"{mod:15s} IMPORT FAILED: {e}")
import torch
print(f"{'cuda avail':15s} {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")
EOF
echo "===================================================="
echo "Done. Activate with:  conda activate $CONDA_ENV_PREFIX"
