#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${_SCRIPT_DIR}/env.sh"

_CONDA_SH="${ENV_ROOT}/miniforge3/etc/profile.d/conda.sh"
if [[ ! -f "${_CONDA_SH}" ]]; then
  echo "Missing Conda initialization script: ${_CONDA_SH}" >&2
  return 1 2>/dev/null || exit 1
fi

source "${_CONDA_SH}"
conda activate recsys26

echo "[OK] Activated conda env: $(python -V)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"

unset _SCRIPT_DIR _CONDA_SH
