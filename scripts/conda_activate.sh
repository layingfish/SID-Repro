#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: source conda_activate.sh <env_name>" >&2
  return 2 2>/dev/null || exit 2
fi

env_name="$1"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${_SCRIPT_DIR}/env.sh"

_CONDA_SH="${ENV_ROOT}/miniforge3/etc/profile.d/conda.sh"
if [[ ! -f "${_CONDA_SH}" ]]; then
  echo "Missing Conda initialization script: ${_CONDA_SH}" >&2
  return 1 2>/dev/null || exit 1
fi

source "${_CONDA_SH}"
conda activate "${env_name}"

python - <<'PY'
import os
import sys

import torch
import transformers

print("[OK]", "python", sys.version.split()[0], "env", repr(os.environ.get("CONDA_DEFAULT_ENV", "")))
print("[OK]", "torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
print("[OK]", "transformers", transformers.__version__)
PY

unset _SCRIPT_DIR _CONDA_SH
