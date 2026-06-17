#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH -J ds4-cmp2
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/compare2_%j.log
set -euxo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
srun --container-image="$ROOT/env/pytorch_26.04-py3.sqsh" --container-mounts=/lustre:/lustre \
  --container-workdir="$ROOT/ds4_vs_hf" bash -c '
    export PATH="$(printf %s "$PATH" | tr ":" "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    unset PYTHONHOME CONDA_PREFIX || true; hash -r || true
    echo "########## mlite(bf16) vs HF(bf16) + mlite run-to-run ##########"
    python compare.py --mlite mlite_logits.pt --hf hf_logits.pt --mlite2 mlite_logits_run2.pt
    echo "########## mlite(bf16) vs HF(fp32) ##########"
    python compare.py --mlite mlite_logits.pt --hf hf_logits_fp32.pt
    echo "########## HF(bf16) vs HF(fp32) = HF bf16 rounding floor ##########"
    python compare.py --mlite hf_logits.pt --hf hf_logits_fp32.pt
  '
