#!/usr/bin/env bash
# Fast-path nanotron env creation for THIS congested network: clone the
# finished `matheval` env (already has torch 2.4.0 + nvidia cu12 stack), then
# install the nanotron-specific delta (editable nanotron, datatrove,
# transformers 4.39.3 downgrade, flash-attn wheel).
#
# Usage:  bash midtraining/env_setup_from_clone.sh
set -euo pipefail
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# The SANGFOR MITM cert for tuna expired (2026-10); pip cannot verify it.
# Traffic is intercepted either way on this network, so skip verification for
# the two pypi mirrors we use.
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com af.hikvision.com.cn"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10

MATHEVAL=/hdd2/xiaolirui/conda_envs/matheval
ENV_PREFIX=/hdd2/xiaolirui/conda_envs/nanotron
NANOTRON_DIR="/hdd2/xiaolirui/OctoThinker/midtraining/nanotron"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

pip_retry() {
    local attempt
    for attempt in 1 2 3 4 5 6 7 8; do
        if "$ENV_PREFIX/bin/python" -m pip install "$@"; then return 0; fi
        echo "==> [pip_retry] attempt $attempt failed; retrying in 15s" >&2; sleep 15
    done
    return 1
}

if [ -x "$MATHEVAL/bin/python" ] && "$MATHEVAL/bin/python" -c "import torch" 2>/dev/null; then
    echo "=== matheval env has torch -> cloning as base for nanotron ==="
    if [ ! -x "$ENV_PREFIX/bin/python" ]; then
        conda create --prefix "$ENV_PREFIX" --clone "$MATHEVAL" -y
    else
        echo "nanotron prefix exists, skipping clone"
    fi
else
    echo "=== matheval env not ready -> fresh python 3.10 + torch from tuna ==="
    if [ ! -x "$ENV_PREFIX/bin/python" ]; then
        conda create --prefix "$ENV_PREFIX" python=3.10 -y
    fi
    pip_retry "torch==2.4.0"
fi

echo "=== [delta 1/5] flash-attn prebuilt wheel ==="
if "$ENV_PREFIX/bin/python" -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed -> skipping"
else
    FA_LOCAL=/tmp/eval_wheels/flash_attn-2.7.0.post2-cp310-cp310-linux_x86_64.whl
    if [ -f "$FA_LOCAL" ]; then
        pip_retry --no-deps "$FA_LOCAL"
    else
        pip_retry --no-deps "$FA_URL"
    fi
fi

echo "=== [delta 2/5] nanotron editable (no-deps) ==="
pip_retry --no-deps -e "$NANOTRON_DIR"

echo "=== [delta 3/5] nanotron base deps + era-appropriate datatrove (numpy<2 compatible) ==="
pip_retry "pyyaml" "numpy<2" "packaging" "safetensors" "dacite" "tqdm" "datasets"
# nanotron[nanosets] pins datatrove@git (main) which now requires numpy>=2 and
# conflicts with nanotron's numpy<2. datatrove 0.3.0 is the era-appropriate
# release (nanotron 0.4, 2025-03) and its jsonl readers live in the core package.
pip_retry "datatrove==0.3.0"  # provides JsonlReader used by tools/preprocess_data.py
pip_retry "numba"
# install the nanosets extra WITHOUT dep resolution (deps handled above)
pip_retry --no-deps -e "$NANOTRON_DIR[nanosets]"

echo "=== [delta 4/5] transformers pin (llama conversion scripts) ==="
# unsloth Llama-3.2 tokenizer.json requires tokenizers>=0.19; transformers 4.39.3
# hard-requires tokenizers<0.19, so we pin 4.46.3 (matches the eval env; the llama
# HF<->nanotron conversion scripts are compatible).
pip_retry "transformers==4.46.3"
pip_retry "orjson"

# Safety: any dep-resolving step above (nanotron[nanosets]) may have upgraded
# torch — restore 2.4 from the matheval clone source if so (same py3.10).
CUR_TORCH="$("$ENV_PREFIX/bin/python" -c "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null || echo none)"
case "$CUR_TORCH" in
    2.4.0|2.4.0+cu121) ;;
    *)
        echo "==> [restore] torch is '$CUR_TORCH' -> restoring torch 2.4.0+cu121 from matheval env"
        for d in torch torchgen nvidia "torch-2.4.0+cu121.dist-info" "flash_attn" "flash_attn-2.7.0.post2.dist-info"; do
            rm -rf "$ENV_PREFIX/lib/python3.10/site-packages/$d" 2>/dev/null || true
            [ -e "$MATHEVAL/lib/python3.10/site-packages/$d" ] && cp -a "$MATHEVAL/lib/python3.10/site-packages/$d" "$ENV_PREFIX/lib/python3.10/site-packages/"
        done
        "$ENV_PREFIX/bin/python" -c "import torch; print('torch restored:', torch.__version__)"
        ;;
esac

echo "=== [delta 5/5] verifying ==="
"$ENV_PREFIX/bin/python" - <<'EOF'
import importlib, torch
for mod in ["torch", "flash_attn", "transformers", "datatrove", "nanotron", "datasets"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{mod:14s} IMPORT FAILED: {e}")
print(f"{'cuda':14s} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
EOF

echo "=== DONE. Activate with: conda activate $ENV_PREFIX ==="
