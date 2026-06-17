#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH -J ds4-ce
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/ce_%j.log
set -eux
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
srun --container-image=$ROOT/env/pytorch_26.04-py3.sqsh --container-mounts=/lustre:/lustre \
  --container-workdir=$ROOT/ds4_vs_hf bash -c 'python3 recompute_ce_dense.py'
