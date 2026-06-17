#!/usr/bin/env bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -J ds4-mlite-dense
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/ds4_vs_hf/mlite_dense_%j.log
set -euxo pipefail

# DENSE ds4 forward: num_hash_layers=0 (no tid2eid hash routing, learned router) +
# dense_topall (num_experts_per_tok == kept routed-expert count = all experts active),
# keep-experts 8 so the proxy fits a single card. Mirrors run_bridge_forward_dense.sbatch.
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
BASE_IMAGE=$ROOT/env/pytorch_26.04-py3.sqsh
REPO_ROOT=$ROOT/megatron_lite/mlite-ds4-hf
EP="${EP:-1}"
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-64}"
KEEP="${KEEP:-8}"
RUN_TAG="${RUN_TAG:-run1}"
OUT_LOGITS="${OUT_LOGITS:-$ROOT/ds4_vs_hf/mlite_logits_dense.pt}"
OUT_INPUT="${OUT_INPUT:-$ROOT/ds4_vs_hf/input_ids_dense.pt}"

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
    torchrun --nproc_per_node='"$EP"' --master_port=29516 \
      '"$ROOT"'/ds4_vs_hf/mlite_forward.py \
      --layers '"$LAYERS"' --seq-len '"$SEQ_LEN"' --ep '"$EP"' \
      --keep-experts '"$KEEP"' --num-hash-layers 0 --dense-topall \
      --out-logits '"$OUT_LOGITS"' --out-input '"$OUT_INPUT"' --run-tag '"$RUN_TAG"'
  '
