#!/usr/bin/env bash
# Fast-path RL env creation for THIS congested network: clone the finished
# `matheval` env (already has torch 2.4.0 + nvidia cu12 stack + pip metadata),
# then install the RL-specific delta. Falls back gracefully if matheval is
# incomplete (torch missing -> falls back to a fresh env + tuna).
#
# Usage:  bash rl/env_setup_from_clone.sh
set -euo pipefail
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# The SANGFOR MITM cert for tuna expired (2026-10); pip cannot verify it.
# Traffic is intercepted either way on this network, so skip verification for
# the two pypi mirrors we use.
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com af.hikvision.com.cn"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10

MATHEVAL=/hdd2/xiaolirui/conda_envs/matheval
ENV_PREFIX=/hdd2/xiaolirui/conda_envs/verl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

pip_retry() {
    local attempt
    # reuse the locally-fetched wheelhouse when present (saves re-downloading
    # big wheels like vllm/ray through the shaped link)
    local extra=()
    if [ -d /tmp/eval_wheels ]; then extra=(--find-links /tmp/eval_wheels); fi
    for attempt in 1 2 3 4 5 6 7 8; do
        if "$ENV_PREFIX/bin/python" -m pip install "${extra[@]}" "$@"; then return 0; fi
        echo "==> [pip_retry] attempt $attempt failed; retrying in 15s" >&2; sleep 15
    done
    return 1
}

if [ -x "$MATHEVAL/bin/python" ] && "$MATHEVAL/bin/python" -c "import torch" 2>/dev/null; then
    echo "=== matheval env has torch -> cloning as base for verl ==="
    if [ ! -x "$ENV_PREFIX/bin/python" ]; then
        conda create --prefix "$ENV_PREFIX" --clone "$MATHEVAL" -y
    else
        echo "verl prefix exists, skipping clone"
    fi
else
    echo "=== matheval env not ready -> falling back to fresh python 3.10 ==="
    if [ ! -x "$ENV_PREFIX/bin/python" ]; then
        conda create --prefix "$ENV_PREFIX" python=3.10 -y
    fi
    pip_retry "torch==2.4.0" "torchvision==0.19.0"
fi

echo "=== [delta 1/4] flash-attn prebuilt wheel ==="
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

# Safety: flash-attn's wheel metadata declares a torch dep — if an earlier
# attempt already let pip upgrade torch (e.g. to 2.14.x), restore 2.4 by
# copying from the matheval clone source (same python 3.10 + nvidia stack).
CUR_TORCH="$("$ENV_PREFIX/bin/python" -c "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null || echo none)"
case "$CUR_TORCH" in
    2.4.0|2.4.0+cu121) ;;
    *)
        echo "==> [restore] torch is '$CUR_TORCH' -> restoring torch 2.4.0+cu121 from matheval env"
        for d in torch torchgen nvidia triton triton-3.0.0.dist-info "torch-2.4.0+cu121.dist-info" "flash_attn" "flash_attn-2.7.0.post2.dist-info" "nvidia_"*triton*; do
            rm -rf "$ENV_PREFIX/lib/python3.10/site-packages/$d" 2>/dev/null || true
            [ -e "$MATHEVAL/lib/python3.10/site-packages/$d" ] && cp -a "$MATHEVAL/lib/python3.10/site-packages/$d" "$ENV_PREFIX/lib/python3.10/site-packages/"
        done
        "$ENV_PREFIX/bin/python" -c "import torch; print('torch restored:', torch.__version__)"
        ;;
esac

echo "=== [delta 2/4] RL pinned requirements (drop bare flash-attn line) ==="
# drop the bare flash-attn line (prebuilt wheel already installed) and
# pyairports==2.1.1 (unresolvable on every mirror — only 0.0.1 exists; verl
# does not import it; it is a stray transitive pin in the freeze)
grep -v -e '^flash-attn\s*$' -e '^pyairports' "${SCRIPT_DIR}/requirements.txt" > /tmp/rl_requirements_no_flashattn.txt
# the cloned base may have torch with a +cu121 local tag; plain == pins won't
# match it -> rewire to the installed version to avoid a 800MB re-download
INSTALLED_TORCH="$("$ENV_PREFIX/bin/python" -c "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null || true)"
if [[ "$INSTALLED_TORCH" == *+* ]]; then
    echo "env has torch $INSTALLED_TORCH (local tag) -> rewiring == pin"
    sed -i -E "s/^torch==[^[:space:]]+/torch==${INSTALLED_TORCH}/" /tmp/rl_requirements_no_flashattn.txt
fi
pip_retry -r /tmp/rl_requirements_no_flashattn.txt

echo "=== [delta 2.5/4] torchdata 0.8.0 (verl's StatefulDataLoader; --no-deps: its torch==2.4.0 pin would clobber our +cu121 build) ==="
pip_retry --no-deps "torchdata==0.8.0"

echo "=== [delta 3/4] verifying ==="
"$ENV_PREFIX/bin/python" -c "import torch, transformers, ray, vllm, hydra, omegaconf, math_verify, liger_kernel; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('transformers', transformers.__version__); \
print('ray', ray.__version__); \
print('vllm', vllm.__version__); \
print('hydra', hydra.__version__, '| omegaconf', omegaconf.__version__); \
print('math_verify OK | liger_kernel OK')"
cd "${REPO_ROOT}/rl"
PYTHONPATH="${REPO_ROOT}/rl:${PYTHONPATH:-}" "$ENV_PREFIX/bin/python" -c "import verl.trainer.main_ppo; print('verl.trainer.main_ppo imports OK (vendored, PYTHONPATH=rl)')"

echo "=== DONE. Activate with: conda activate $ENV_PREFIX ==="
