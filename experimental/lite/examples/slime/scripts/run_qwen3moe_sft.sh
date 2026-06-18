#!/usr/bin/env bash
# Supervised fine-tuning of qwen3_moe (Qwen3-30B-A3B) with the Megatron Lite
# backend inside the slime framework. Mirrors slime's SFT examples (e.g.
# examples/retool/retool_qwen3_4b_sft.sh): SGLang rollout is bypassed via
# --debug-train-only and SFT data comes from slime.rollout.sft_rollout.
#
# Run this inside an allocation that already has slime + Megatron Lite + their
# dependencies importable (see README.md). GPU runs go through Slurm.
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then set -x; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"          # .../examples/slime
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"        # .../experimental/lite
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"           # Megatron-LM repo root

add_pythonpath() { [[ -n "${1:-}" ]] && export PYTHONPATH="${1}:${PYTHONPATH:-}"; }
add_pythonpath "${EXAMPLE_ROOT}"   # so `import slime_mlite` resolves
add_pythonpath "${LITE_ROOT}"      # so `import megatron.lite` resolves
add_pythonpath "${REPO_ROOT}"

# ---- 1. user-adjustable -----------------------------------------------------
: "${SLIME_ROOT:?set SLIME_ROOT to the slime checkout (with the train-backend seam)}"
: "${MODEL_PATH:?set MODEL_PATH to a Qwen3-30B-A3B HF checkpoint directory}"
: "${TRAIN_DATA:?set TRAIN_DATA to an SFT messages parquet path}"

MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh}"
NUM_GPUS="${NUM_GPUS:-8}"
SAVE_DIR="${SAVE_DIR:-${EXAMPLE_ROOT}/outputs/qwen3moe_sft}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"
MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-qwen3_moe}"
MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-dist_opt}"
# Offload optimizer state to CPU; needed to fit Qwen3-30B-A3B on 8x80GB.
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-1}"

LR="${LR:-1e-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
NUM_EPOCH="${NUM_EPOCH:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
DRY_RUN="${DRY_RUN:-0}"

# ---- 2. derived -------------------------------------------------------------
mkdir -p "${SAVE_DIR}"
# shellcheck disable=SC1090
source "${MODEL_SCRIPT}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1

# ---- 3. arg groups ----------------------------------------------------------
BACKEND_ARGS=(
   --train-backend mlite
   --train-backend-module slime_mlite
   --mlite-model-name "${MLITE_MODEL_NAME}"
   --mlite-impl lite
   --mlite-optimizer-backend "${MLITE_OPTIMIZER_BACKEND}"
)
if [[ "${OPTIMIZER_OFFLOAD}" == "1" || "${OPTIMIZER_OFFLOAD}" == "True" || "${OPTIMIZER_OFFLOAD}" == "true" ]]; then
   BACKEND_ARGS+=(--mlite-optimizer-offload)
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${TRAIN_DATA}"
   --input-key messages
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --clip-grad 1.0
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

# ---- 4. launch --------------------------------------------------------------
COMMAND=(
   python3 "${SLIME_ROOT}/train_async.py"
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${BACKEND_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${SFT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
   printf '%q ' "${COMMAND[@]}"; printf '\n'
   exit 0
fi

"${COMMAND[@]}"
