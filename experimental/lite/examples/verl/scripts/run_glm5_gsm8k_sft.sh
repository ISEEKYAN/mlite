#!/usr/bin/env bash
# GLM-5.1 (glm_moe_dsa / fused-DSA) gsm8k SFT launcher for the Megatron Lite verl example.
#
# GLM-5.1 is a fused-DSA model: it must run under the DSA verl env
# (pytorch_26.04-py3.sqsh + mlite-2604-verl-dsa-sm90-overlay; see
# wiki/dead_ends/mlite-env-setup.md). The reference full config is
# 256 GPU pp8 ep8 cp8 etp1 with full recompute (完全重算).
#
# This wrapper only sets GLM-5.1 defaults and delegates to the shared
# run_qwen3moe_sft.sh driver (engine.model_name=auto resolves glm_moe_dsa -> glm5).
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"

DATASET_DIR="${DATASET_DIR:-${HOME}/data/gsm8k_sft}"
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/users/bayan/code/models/GLM-5.1}"
export TRAIN_FILES="${TRAIN_FILES:-${DATASET_DIR}/train.parquet}"
export VAL_FILES="${VAL_FILES:-${DATASET_DIR}/test.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/glm5_gsm8k_sft}"
export PROJECT_NAME="${PROJECT_NAME:-verl-mlite-glm5-gsm8k-sft}"
export RUN_NAME="${RUN_NAME:-glm5_gsm8k_sft_mlite}"

# Reference full mesh: 256 GPU = pp8 * cp8 * dp4 (tp1), experts over ep8 (etp1).
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-8}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

export TOTAL_STEPS="${TOTAL_STEPS:-30}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"

# fused-DSA model: full recompute (完全重算) is the reference config; param/opt/grad offload
# for memory headroom. Caller can override.
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
export GRAD_OFFLOAD="${GRAD_OFFLOAD:-True}"
export MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-dist_opt}"

# Default to full recompute unless caller already passed an explicit recompute spec.
have_recompute=0
for a in "$@"; do
  case "$a" in
    *engine.impl_cfg.recompute=*) have_recompute=1 ;;
  esac
done
RECOMPUTE_ARGS=()
if [[ "${have_recompute}" == "0" ]]; then
  RECOMPUTE_ARGS+=("+engine.impl_cfg.recompute=full")
fi

exec bash "${SCRIPT_DIR}/run_qwen3moe_sft.sh" "${RECOMPUTE_ARGS[@]}" "$@"
