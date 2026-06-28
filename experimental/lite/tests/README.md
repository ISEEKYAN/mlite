# Megatron Lite Validation

`experimental/lite/tests` separates MLite validation into two layers:

- Unit tests: CPU/single-process contract tests for primitive, model, runtime, checkpoint sentinels, config, and helper behavior. Pure helper tests stub optional imports when no Transformer Engine runtime path is exercised; tests that need the real package explicitly skip when unavailable.
- Smoke tests: real `torch.distributed` tests for TP/EP/PP/CP/FSDP2/offload/checkpoint/distopt and tiny Qwen lite forward/backward behavior. Smoke runs are capped at one node and at most 8 GPUs.

Run unit coverage:

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" pytest experimental/lite/tests/unit
```

Pure config and weight-mapping modules can be imported without `safetensors`.
Tests or runtime paths that read/write HF checkpoint files require the optional
`safetensors` package and fail with an explicit dependency error when it is absent.
Training DCP loads require the completion manifest and model schema emitted by
the current writer. Pre-manifest checkpoints are rejected by default; a
one-time migration tool may opt in with `allow_legacy_checkpoint=True`, then
must re-save in the current format before normal training resumes.

Run smoke coverage on one node:

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" MLITE_RUN_SMOKE=1 MLITE_SMOKE_NPROC=8 \
  experimental/lite/tests/run_primitive_validation.sh
```

The smoke suite is skipped by default in regular `pytest` runs. Enable it with `--mlite-smoke` or `MLITE_RUN_SMOKE=1`.
Release-acceptance commands should also pass `--mlite-fail-on-skip`; this turns
any selected skip or xfail (including a missing CUDA/kernel dependency) into a
test failure instead of silently accepting an unexecuted gate.

The pinned GLM-5.2-FP8 real-weight projection gate downloads only the three
required byte ranges (12,590,080 bytes total), verifies their release hashes,
and then runs on one GPU:

```bash
python experimental/lite/tests/fetch_glm52_fp8_projection_authority.py /tmp/glm52-fp8
GLM52_FP8_PROJECTION_AUTHORITY_DIR=/tmp/glm52-fp8 \
  pytest --mlite-smoke --mlite-fail-on-skip -q -s \
  experimental/lite/tests/smoke/model/test_glm52_fp8_real_weight_projection_smoke.py
```

This is pinned real-checkpoint, dequantized-BF16 q_a projection-level evidence
through the production `torch.nn.Linear` + Transformer Engine RMSNorm path. It
is not HF quantized-runtime, full-model, or long-context parity.

The GLM-5.2 indexer-RoPE authority is independent of vanilla Transformers
5.12, whose indexer ignores the released `indexer_rope_interleave=true` field.
Fetch the pinned release config and vLLM v0.23.0 sources, verify their exact
revisions and digests, then run the score/top-k oracle:

```bash
python experimental/lite/tests/fetch_glm52_rope_layout_authority.py /tmp/glm52-rope
pytest --mlite-fail-on-skip -q -s \
  experimental/lite/tests/unit/model/test_glm52_hf_attention_parity.py
```

The oracle binds MLite's real `DSAIndexer` to the released adjacent-pair
layout at score/top-k level. The full-model comparison uses an explicitly
adapted, instance-local HF indexer subclass; it is supporting numerical
evidence, not a claim of unmodified-Transformers or Megatron-Bridge parity.

Current matrix:

| Surface | Unit | Smoke |
| --- | --- | --- |
| TP/EP/PP/CP/SP topology | `unit/primitive/test_parallel_unit.py`, `unit/primitive/test_parallel_dimensions_independent_unit.py` | `smoke/primitive/test_parallel_topologies_smoke.py` |
| TP linear/vocab primitives | `unit/primitive/test_parallel_dimensions_independent_unit.py` | Qwen model smoke exercises TP linear surfaces |
| EP token dispatch | `unit/primitive/test_parallel_dimensions_independent_unit.py` | Qwen model smoke exercises router, dispatcher, and experts |
| THD packing helpers | `unit/primitive/test_parallel_unit.py` | CP topology smoke exercises distributed CP groups |
| GQA/attention split contract | `unit/primitive/test_attention_moe_unit.py` | Qwen model smoke exercises attention forward/backward |
| MoE router/aux-loss contract | `unit/primitive/test_attention_moe_unit.py` | Qwen model smoke exercises router, dispatcher, and experts |
| LoRA adapter primitives | `unit/primitive/test_module_primitives_independent_unit.py` | Qwen model smoke can enable adapters in follow-up coverage |
| MTP/MRoPE/Gated Delta helper contracts | `unit/primitive/test_module_primitives_independent_unit.py`, `unit/primitive/test_ops_data_trainstep_unit.py` | Qwen3.5 MoE model smoke exercises MRoPE/Gated DeltaNet paths |
| Loss/logprob/math ops | `unit/primitive/test_ops_data_trainstep_unit.py` | Qwen model smoke exercises loss plumbing |
| Data/recompute/train-step primitives | `unit/primitive/test_ops_data_trainstep_unit.py` | model/runtime smoke exercises training loop integration |
| DDP + distributed optimizer | `unit/primitive/test_checkpoint_unit.py`, `unit/primitive/test_checkpoint_runtime.py` | `smoke/primitive/test_distopt_checkpoint_smoke.py` |
| FSDP2 config/wrap/offload | `unit/primitive/test_fsdp2_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| FSDP2 save/load resume | `unit/primitive/test_checkpoint_unit.py`, `unit/primitive/test_checkpoint_runtime.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Checkpoint restore vs direct training | `unit/primitive/test_checkpoint_unit.py`, `unit/primitive/test_checkpoint_runtime.py` | FSDP2 and distopt checkpoint smokes cover distributed restore paths |
| Runtime backend registry/config | `unit/primitive/test_runtime_config_unit.py`, `unit/runtime/test_runtime_backend_unit.py` | covered through checkpoint/model handles |
| Runtime env/offload controls | `unit/runtime/test_runtime_backend_unit.py` | `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Optimizer update-state offload fraction | `unit/primitive/test_runtime_config_unit.py` and single-process CUDA coverage in `unit/primitive/test_fsdp2_offload_gpu.py` | multi-rank offloaded grad clipping is checked against the non-offloaded baseline in `smoke/primitive/test_fsdp2_offload_checkpoint_smoke.py` |
| Qwen3 MoE lite config/build/forward | `unit/model/test_qwen_config_unit.py` | `smoke/model/test_qwen_lite_forward_smoke.py` |
| Qwen3.5 MoE lite config/build/forward | `unit/model/test_qwen_config_unit.py` | `smoke/model/test_qwen_lite_forward_smoke.py` |
| GLM-5.2 released RoPE/index-share layout | `unit/model/test_glm52_hf_attention_parity.py`, `unit/model/test_glm5_lite_static.py` | `smoke/model/test_glm52_hf_production_layout_full_model_smoke.py`, `smoke/model/test_glm52_fp8_real_weight_projection_smoke.py` |
| VERL external-engine registry and THD/CP runtime | `unit/verl/test_mlite_engine_config.py` | `unit/verl/test_mlite_engine_cp_smoke.py` (distributed smoke markers; not Ray worker/trainer E2E) |

Classic FSDP is not a separate MLite primitive in the current source tree; MLite's native sharded optimizer coverage is FSDP2 plus Megatron DDP/distopt.
