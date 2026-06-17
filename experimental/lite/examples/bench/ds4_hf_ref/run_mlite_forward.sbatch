#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -J ds4-mlite-fwd
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/mlite_fwd_%j.log
set -euxo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
BASE_IMAGE=$ROOT/env/pytorch_26.04-py3.sqsh
REPO_ROOT=$ROOT/megatron_lite/mlite-ds4-hf
EP="${EP:-1}"
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-64}"
RUN_TAG="${RUN_TAG:-run1}"
OUT_LOGITS="${OUT_LOGITS:-$ROOT/ds4_vs_hf/mlite_logits.pt}"
OUT_INPUT="${OUT_INPUT:-$ROOT/ds4_vs_hf/input_ids.pt}"

srun --ntasks-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT/ds4_vs_hf" \
  bash -c '
    set -eux
    export PATH="$(printf %s "$PATH" | tr ":" "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES || true
    hash -r || true
    ROOT='"$ROOT"'; REPO_ROOT='"$REPO_ROOT"'
    SM90=$ROOT/mlite-2604-verl-dsa-sm90-overlay/lib/python3.12/site-packages
    MCORE_DEV=$ROOT/verl_optimize/mcore_dev
    export PYTHONPATH="$SM90/nvidia_cutlass_dsl/python_packages:$SM90:$MCORE_DEV:$REPO_ROOT/experimental/lite:${PYTHONPATH:-}"
    torchrun --nproc_per_node='"$EP"' --master_port=29512 \
      '"$ROOT"'/ds4_vs_hf/mlite_forward.py \
      --layers '"$LAYERS"' --seq-len '"$SEQ_LEN"' --ep '"$EP"' \
      --out-logits '"$OUT_LOGITS"' --out-input '"$OUT_INPUT"' --run-tag '"$RUN_TAG"'
  '
