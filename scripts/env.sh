#!/usr/bin/env bash

_RECSYS26_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export RECSYS26_ROOT="${RECSYS26_ROOT:-$(cd "${_RECSYS26_SCRIPT_DIR}/.." && pwd -P)}"

# External resources. Each path may be overridden before sourcing this file.
export DATA_ROOT="${DATA_ROOT:-${RECSYS26_ROOT}/datasets}"
export MODEL_ROOT="${MODEL_ROOT:-${RECSYS26_ROOT}/models}"
export ENV_ROOT="${ENV_ROOT:-${RECSYS26_ROOT}/envs}"
export RUN_ROOT="${RUN_ROOT:-${RECSYS26_ROOT}/logs}"
export TOKENIZER_ROOT="${TOKENIZER_ROOT:-${RUN_ROOT}/tokenizers}"

# HuggingFace / Transformers caches.
export HF_HOME="${HF_HOME:-${MODEL_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATA_ROOT}/hf}"

# PyTorch and experiment logging.
export TORCH_HOME="${TORCH_HOME:-${MODEL_ROOT}/torch}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset _RECSYS26_SCRIPT_DIR
