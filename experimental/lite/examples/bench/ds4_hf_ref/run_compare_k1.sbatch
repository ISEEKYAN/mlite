#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:08:00
#SBATCH -J ds4-cmpk1
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/compare_k1_%j.log
set -euxo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
srun --container-image="$ROOT/env/pytorch_26.04-py3.sqsh" --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT/ds4_vs_hf" bash -c '
    export PATH="$(printf %s "$PATH" | tr ":" "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    unset PYTHONHOME CONDA_PREFIX || true; hash -r || true
    echo "########## K=1: mlite(bf16) vs HF(bf16) ##########"
    python compare.py --mlite mlite_logits_k1.pt --hf hf_logits_k1.pt
  '
