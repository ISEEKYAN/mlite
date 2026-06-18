# slime Megatron Lite Example

This directory contains the Megatron Lite training backend for the
[slime](https://github.com/THUDM/slime) RL framework, plus a Qwen3 MoE
(Qwen3-30B-A3B) SFT launch script.

The Python package is `slime_mlite`. Importing it registers slime's training
backend as `mlite` (via slime's train-backend registry), while Megatron Lite
model implementations still use `impl=lite`.

Unlike the VERL integration, slime has no built-in backend registry, so this
example pairs with a small **seam** in slime itself (the train-backend registry
in `slime/ray/train_backend.py` plus registry-based dispatch in
`slime/ray/actor_group.py` and `slime/utils/arguments.py`). All Megatron Lite
specific code lives here; slime only learns how to load and dispatch a backend
by name.

## Layout

- `slime_mlite/actor.py`: `MLiteTrainRayActor(TrainRayActor)` backed by
  `megatron.lite.runtime` (build model from HF, SFT training step, checkpoint
  save).
- `slime_mlite/arguments.py`: `mlite_parse_args` (thin wrapper over Megatron's
  parser) and `add_mlite_arguments` (the `--mlite-*` flags).
- `slime_mlite/data.py`: packs slime rollout data into Megatron Lite THD
  micro-batches using `megatron.lite.primitive.pack_nested_thd`.
- `slime_mlite/loss.py`: supervised fine-tuning loss over masked tokens.
- `scripts/run_qwen3moe_sft.sh`: Qwen3 MoE SFT launcher (`--debug-train-only`,
  no SGLang).

## Prerequisites

Expose these before running (GPU runs go through Slurm):

- slime with the train-backend seam. See
  [`REQUIRED_SLIME.txt`](REQUIRED_SLIME.txt) for the reference commit.
- Megatron-LM from this repository (add `experimental/lite` to `PYTHONPATH`).
- A Qwen3-30B-A3B HF checkpoint.

## Usage

```bash
export SLIME_ROOT=/path/to/slime
export MODEL_PATH=/path/to/Qwen3-30B-A3B
export TRAIN_DATA=/path/to/sft_messages.parquet
bash scripts/run_qwen3moe_sft.sh
```

The launcher selects this backend with `--train-backend mlite
--train-backend-module slime_mlite`. SFT data flows through slime's
`slime.rollout.sft_rollout.generate_rollout`, and the loss is
`--loss-type sft_loss`.

## Scope

This is the S1 slice: the slime seam, the `MLiteTrainRayActor` skeleton, and the
SFT training step. Weight resync to the rollout engine (colocate/disaggregate)
and the RL training step are follow-ups; `update_weights` is a no-op on the
SFT-only `--debug-train-only` path and raises otherwise.
