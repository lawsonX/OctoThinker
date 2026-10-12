#!/usr/bin/env bash
# Environment setup for OctoThinker evaluation (vLLM few-shot / zero-shot math eval).
#
# Creates a conda env at /hdd2/xiaolirui/conda_envs/matheval (python 3.10) and installs
# torch 2.4.0 (cu121) plus the pinned eval/requirements.txt with math-verify pinned
# to ==0.5.2 (same version as the RL reward; the requirements file leaves it unpinned).
#
# flash-attn is intentionally NOT installed: the eval path loads models through
# vLLM, which does not need flash-attn for inference on A100.
#
# Run:  bash eval/env_setup.sh
set -euo pipefail

# SANGFOR proxy breaks pypi.nvidia.com / download.pytorch.org and the internal
# artifactory only caches recent versions; tuna mirrors all of PyPI and is fast.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10

ENV_PREFIX=/hdd2/xiaolirui/conda_envs/matheval
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Creating conda env at ${ENV_PREFIX} (python 3.10) ==="
if [ -d "${ENV_PREFIX}" ]; then
    echo "Env already exists, skipping conda create (delete it first to recreate)."
else
    conda create --prefix "${ENV_PREFIX}" python=3.10 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip

echo "=== [2/4] Installing torch 2.4.0 (PyPI wheel = cu121) ==="
pip install torch==2.4.0

echo "=== [3/4] Installing pinned requirements (math-verify -> ==0.5.2) ==="
# Pin math-verify to the same version used by the RL reward (rl/requirements.txt).
# The shipped line is `math-verify[antlr4_11_0]` (unpinned); rewrite it in a temp copy.
sed 's/^math-verify\[antlr4_11_0\].*/math-verify[antlr4_11_0]==0.5.2/' "${SCRIPT_DIR}/requirements.txt" > /tmp/eval_requirements_pinned.txt

# torch installed from download.pytorch.org carries a +cu121 LOCAL version tag;
# a plain ==2.4.0 pin does not match it (PEP 440) and pip would re-download
# 800MB every attempt. Rewire exact pins to the installed local version.
for P in torch torchvision; do
    INSTALLED="$(python -c "import importlib.metadata as m; print(m.version('$P'))" 2>/dev/null || true)"
    if [ -n "$INSTALLED" ] && [[ "$INSTALLED" == *+* ]]; then
        echo "env has $P $INSTALLED (local tag) -> rewiring == pin"
        sed -i -E "s/^${P}==[^[:space:]]+/${P}==${INSTALLED}/" /tmp/eval_requirements_pinned.txt
    fi
done

# The internal artifactory mirror lacks a few niche pins from this big freeze
# (e.g. pyairports==2.1.1). Retry-loop: whenever pip reports an unresolvable
# distribution, drop that package's version pin (keep the package) and retry.
FILTERED=/tmp/eval_requirements_filtered.txt
cp /tmp/eval_requirements_pinned.txt "$FILTERED"
for attempt in $(seq 1 30); do
    if pip install -r "$FILTERED" > /tmp/eval_pip_attempt.log 2>&1; then
        echo "requirements installed after $attempt attempt(s)"
        break
    fi
    # Retry-worthy failure shapes on this link:
    #   1) mirror lacks a pinned version      -> drop the pin, then retry
    #   2) mid-stream download stall          -> just retry (pip resumes)
    if grep -q "No matching distribution found\|Could not find a version that satisfies" /tmp/eval_pip_attempt.log; then
        BAD=$(grep -oP '(?:No matching distribution found for|satisfies the requirement) \K[A-Za-z0-9_.\-]+(?===|\s|\()' /tmp/eval_pip_attempt.log | head -1)
        if [ -n "${BAD}" ]; then
            echo "mirror lacks pin for '${BAD}' -> dropping its version pin (attempt $attempt)"
            sed -i -E "s/^(${BAD//./\\.})(\[.*?\])?==[^[:space:]]+/\1\2/" "$FILTERED"
        else
            echo "unresolvable requirement, retrying as-is (attempt $attempt)"
        fi
    else
        echo "pip attempt $attempt failed (download stall?); retrying" >&2
    fi
    sleep 10
done
if ! pip install -r "$FILTERED" --dry-run > /dev/null 2>&1; then
    echo "requirements still not satisfiable after all retries; last log:" >&2
    tail -20 /tmp/eval_pip_attempt.log >&2
    exit 1
fi

echo "=== [4/4] Verifying ==="
python -c "import torch, vllm, transformers, datasets, sympy, math_verify; \
from importlib.metadata import version; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('vllm', vllm.__version__); \
print('transformers', transformers.__version__); \
print('datasets', datasets.__version__); \
print('sympy', sympy.__version__); \
print('math_verify', version('math-verify'))"

echo "=== DONE. Activate with: conda activate ${ENV_PREFIX} ==="
