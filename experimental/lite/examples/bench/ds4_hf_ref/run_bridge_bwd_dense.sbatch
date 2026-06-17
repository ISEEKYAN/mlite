#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH -J ds4-bridge-bwd
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/bridge_bwd_%j.log
set -euxo pipefail

# DENSE ds4 bridge (megatron.bridge) forward, official-config: num_hash_layers=0 +
# dense_topall + keep-experts 8 + attention_backend=auto (official recipe sets None,
# our bench default forces 'flash'). rope-fusion off (non-fused reference path).
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
BASE_IMAGE=$ROOT/env/pytorch_26.04-py3.sqsh
REPO_ROOT=$ROOT/megatron_lite/mlite-ds4-hf
HF_PATH=$ROOT/models/DeepSeek-V4-Flash
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-64}"
EP="${EP:-1}"
KEEP="${KEEP:-8}"
ATTN="${ATTN:-auto}"
ROPE="${ROPE:-off}"
OUT_LOGITS="${OUT_LOGITS:-$ROOT/ds4_vs_hf/bridge_logits_dense.pt}"

srun --ntasks-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT/ds4_vs_hf" \
  bash -c '
    set -eux
    export PATH="$(printf %s "$PATH" | tr ":" "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES || true
    hash -r || true
    ROOT='"$ROOT"'; REPO_ROOT='"$REPO_ROOT"'; HF_PATH='"$HF_PATH"'
    LAYERS='"$LAYERS"'; SEQ_LEN='"$SEQ_LEN"'; EP='"$EP"'; KEEP='"$KEEP"'; ATTN='"$ATTN"'; ROPE='"$ROPE"'
    TS512=$ROOT/ts512-site; FHT=$ROOT/fht-cu13-clean
    BRIDGE=$ROOT/verl_optimize/megatron_bridge_latest/src
    MCORE=$ROOT/megatron_lite/mcore-dev-latest
    DEV=$ROOT/mlite-2604-dsa-glm5-overlay-develop/lib/python3.12/site-packages
    export PYTHONPATH="$TS512:$FHT:$REPO_ROOT/experimental/lite:$BRIDGE:$MCORE:$DEV"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    torchrun --nproc_per_node=$EP --master_port=29537 \
      '"$ROOT"'/ds4_vs_hf/bridge_forward.py \
      --layers $LAYERS --seq-len $SEQ_LEN --seed 42 --ep $EP --hf-path "$HF_PATH" \
      --keep-experts $KEEP --num-hash-layers 0 --dense-topall \
      --attention-backend $ATTN --rope-fusion $ROPE --backward \
      --out-logits '"$OUT_LOGITS"' \
      --check-input '"$ROOT"'/ds4_vs_hf/input_ids_dense.pt
  '
