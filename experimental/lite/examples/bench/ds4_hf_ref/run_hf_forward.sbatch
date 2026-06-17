#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -J ds4-hf-fwd
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/hf_fwd_%j.log
set -euxo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
BASE_IMAGE=$ROOT/env/pytorch_26.04-py3.sqsh
LAYERS="${LAYERS:-2}"
DTYPE="${DTYPE:-bf16}"
OUT="${OUT:-$ROOT/ds4_vs_hf/hf_logits.pt}"

srun --ntasks-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT/ds4_vs_hf" \
  bash -c '
    set -eux
    export PATH="$(printf %s "$PATH" | tr ":" "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES || true
    hash -r || true
    ROOT='"$ROOT"'
    export PYTHONPATH="$ROOT/ts512-site:$ROOT/ds4_vs_hf:${PYTHONPATH:-}"
    python '"$ROOT"'/ds4_vs_hf/hf_forward.py --layers '"$LAYERS"' --device cpu --dtype '"$DTYPE"' \
      --input '"$ROOT"'/ds4_vs_hf/input_ids.pt \
      --out '"$OUT"'
  '
