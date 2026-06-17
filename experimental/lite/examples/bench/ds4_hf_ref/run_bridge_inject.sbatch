#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH -J ds4-bridge-inj
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/bridge_inj_%j.log
set -euxo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
BASE_IMAGE=$ROOT/env/pytorch_26.04-py3.sqsh
REPO_ROOT=$ROOT/megatron_lite/mlite-ds4-hf
HF_PATH=$ROOT/models/DeepSeek-V4-Flash
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-64}"
EP="${EP:-4}"

srun --ntasks-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT/experimental/lite" \
  bash -c '
    set -eux
    export PATH="$(printf %s "$PATH" | tr ":" "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES || true
    hash -r || true
    ROOT='"$ROOT"'; REPO_ROOT='"$REPO_ROOT"'; HF_PATH='"$HF_PATH"'
    LAYERS='"$LAYERS"'; SEQ_LEN='"$SEQ_LEN"'; EP='"$EP"'
    TS512=$ROOT/ts512-site; FHT=$ROOT/fht-cu13-clean
    BRIDGE=$ROOT/verl_optimize/megatron_bridge_latest/src
    MCORE=$ROOT/megatron_lite/mcore-dev-latest
    DEV=$ROOT/mlite-2604-dsa-glm5-overlay-develop/lib/python3.12/site-packages
    export PYTHONPATH="$TS512:$FHT:$REPO_ROOT/experimental/lite:$BRIDGE:$MCORE:$DEV"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    torchrun --nproc_per_node=$EP --master_port=29533 -m examples.bench.ds4_hf_ref.bridge_inject_tid2eid run \
      --backend bridge --hf-path "$HF_PATH" --model-name deepseek_v4 \
      --truncate-layers $LAYERS --disable-mtp --no-optimizer --ep $EP --etp 1 \
      --same-data-across-dp --seq-len $SEQ_LEN --steps 1 --seed 42 --skip-weight-hash \
      --override-transformer-json '"'"'{"apply_rope_fusion": false}'"'"' \
      --output-json '"$ROOT"'/ds4_vs_hf/bridge_ds4_norope.json
  '
