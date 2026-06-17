#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -J ds4-mlite-bench
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/mlite_bench_%j.log
set -euxo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
BASE_IMAGE=$ROOT/env/pytorch_26.04-py3.sqsh
REPO_ROOT=$ROOT/megatron_lite/mlite-ds4-hf
HF_PATH=$ROOT/models/DeepSeek-V4-Flash
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-64}"

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
    LAYERS='"$LAYERS"'; SEQ_LEN='"$SEQ_LEN"'
    SM90=$ROOT/mlite-2604-verl-dsa-sm90-overlay/lib/python3.12/site-packages
    MCORE_DEV=$ROOT/verl_optimize/mcore_dev
    export PYTHONPATH="$SM90/nvidia_cutlass_dsl/python_packages:$SM90:$MCORE_DEV:$REPO_ROOT/experimental/lite"
    torchrun --nproc_per_node=1 --master_port=29541 -m examples.bench.correctness run \
      --backend mlite --hf-path "$HF_PATH" --model-name deepseek_v4 \
      --truncate-layers $LAYERS --disable-mtp --no-optimizer --ep 1 --etp 1 \
      --impl-cfg-json "{\"optimizer\": null, \"mtp_enable\": false, \"mtp_enable_train\": false}" \
      --same-data-across-dp --seq-len $SEQ_LEN --steps 1 --seed 42 --skip-weight-hash \
      --output-json '"$ROOT"'/ds4_vs_hf/mlite_ds4.json
  '
