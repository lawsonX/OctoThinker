#!/usr/bin/env bash
# Environment setup for OctoThinker RL (vendored verl 0.2.0.dev + GRPO training).
#
# Creates a conda env at /hdd2/xiaolirui/conda_envs/verl (python 3.10) and installs:
#   1. torch 2.4.0 + torchvision 0.19.0 (cu121 wheels)
#   2. prebuilt flash-attn 2.7.0.post2 wheel (avoids a long source build;
#      must be installed BEFORE the requirements file, which contains an
#      unpinned `flash-attn` line that we filter out)
#   3. everything else from rl/requirements.txt
#
# Run:  bash rl/env_setup.sh
set -euo pipefail

# SANGFOR proxy breaks pypi.nvidia.com / download.pytorch.org and the internal
# artifactory only caches recent versions; tuna mirrors all of PyPI but the
# shared link stalls on multi-hundred-MB streams. Strategy: torch/torchvision/
# nvidia wheels are pre-downloaded ONCE into a shared wheelhouse with resumable
# curl (models/fetch_wheels.sh + models/fetch_nvidia_wheels.sh) and installed
# offline; everything else installs from tuna.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=10
WHEELHOUSE="${WHEELHOUSE:-/hdd2/xiaolirui/wheelhouse}"

ENV_PREFIX=/hdd2/xiaolirui/conda_envs/verl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== [1/5] Creating conda env at ${ENV_PREFIX} (python 3.10) ==="
if [ -d "${ENV_PREFIX}" ]; then
    echo "Env already exists, skipping conda create (delete it first to recreate)."
else
    conda create --prefix "${ENV_PREFIX}" python=3.10 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip

echo "=== [2/5] Installing torch 2.4.0 / torchvision 0.19.0 (offline from wheelhouse when available) ==="
if [ -d "$WHEELHOUSE" ] && [ -n "$(ls "$WHEELHOUSE"/torch-2.4.0-*.whl "$WHEELHOUSE"/nvidia-*-py3-none.whl 2>/dev/null)" ]; then
    pip install --no-index --find-links "$WHEELHOUSE" torch==2.4.0 torchvision==0.19.0
else
    pip install torch==2.4.0 torchvision==0.19.0
fi

echo "=== [3/5] Installing prebuilt flash-attn wheel ==="
if [ -f "$WHEELHOUSE/flash_attn-2.7.0.post2-cp310.whl" ]; then
    pip install --no-deps "$WHEELHOUSE/flash_attn-2.7.0.post2-cp310.whl"
else
    pip install --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

echo "=== [4/5] Installing pinned requirements (dropping the unpinned flash-attn line) ==="
# The prebuilt flash-attn wheel is already installed; filter the bare
# `flash-attn` line out of requirements.txt so pip does not try to rebuild it.
grep -v -e '^flash-attn\s*$' "${SCRIPT_DIR}/requirements.txt" > /tmp/rl_requirements_no_flashattn.txt
pip install -r /tmp/rl_requirements_no_flashattn.txt

echo "=== [5/5] Verifying ==="
python -c "import torch, transformers, ray, vllm, hydra, omegaconf, math_verify, liger_kernel; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('transformers', transformers.__version__); \
print('ray', ray.__version__); \
print('vllm', vllm.__version__); \
print('hydra-core', hydra.__version__, '| omegaconf', omegaconf.__version__); \
print('math_verify OK | liger_kernel OK')"
# verl is NOT pip-installed; it is vendored under rl/verl and imported with cwd=rl/.
cd "${REPO_ROOT}/rl"
PYTHONPATH="${REPO_ROOT}/rl:${PYTHONPATH:-}" python -c "import verl.trainer.main_ppo; print('verl.trainer.main_ppo imports OK (vendored, PYTHONPATH=rl)')"

echo "=== DONE. Activate with: conda activate ${ENV_PREFIX} ==="
