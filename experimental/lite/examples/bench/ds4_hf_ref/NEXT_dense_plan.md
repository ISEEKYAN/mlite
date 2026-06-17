# ds4 dense-plan + official-bridge-config — handoff recipe (fresh doer)

Goal (bayan/coord 2026-06-17): force ds4 **dense** (remove hash-MoE/tid2eid routing) and
re-run the **3-way** (mlite / megatron.bridge / HF) × **(fwd + bwd)** using the **official**
Megatron-Bridge ds4 config, to see whether the bridge ds4 divergence lives in the MoE-routing
path. If dense+official-config makes mlite-vs-bridge tight → ds4 gets a Megatron primary ref.

## What's already established (do NOT re-litigate)
- **ds4 is M1-closed on HF** (coord-confirmed): mlite-vs-HF logits cosine **0.998071**,
  argmax 92.2%, self-CE 0.7%, **run-to-run max_abs=0 bitwise**, K=1≈K=2 (per-layer benign bf16).
  HF is the trustworthy independent ref (two independent impls — mlite fused-native + HF eager —
  agree; they share bit-identical weights via mlite's release→HF converter).
- **bridge is the outlier**: 3-way (full logits, same tokens) mlite-vs-bridge **0.903**,
  HF-vs-bridge **0.908**, mlite-vs-HF **0.998**. tid2eid is NOT the cause (manual inject =
  no-op, changed_entries=0; bridge loads it bitwise-correct). The "mean 19.3 all-positive / nan"
  was a misread per-token-loss tensor, not logits. rope-fusion ON was worse (0.79). sliding_window
  128 > seq 64 rules out sliding. These were all with a **likely-wrong bench bridge config** (below).

## The likely bench-bridge MISCONFIG (root-cause candidates — TEST THESE FIRST)
Compare against the OFFICIAL recipe `verl_optimize/megatron_bridge_latest/src/megatron/bridge/
recipes/deepseek/deepseek_v4.py` (`deepseek_v4_flash_pretrain_config`):
1. **attention_backend**: official sets `cfg.model.attention_backend = None`. Our bench FORCES
   `attention_backend="flash"` — hardcoded default in `_lower_transformer_overrides`
   (`megatron/lite/runtime/backends/bridge/runtime.py:80`). flash likely mishandles dsv4_hybrid
   (head_dim 512 / partial-RoPE 64 / sinks). **Override it: `--override-transformer-json
   '{"attention_backend": null}'`** (cfg.override_transformer_config wins over the default).
2. **_DENSE_FORWARD_MODELS**: the bench forces ds4 through `_packed_batch_to_dense_inputs`
   (BSHD, `(3,1,T)` MRoPE-style position_ids — written for qwen3.5 GDN). ds4 is yarn-rope, not
   mrope. Check whether ds4 should leave `_DENSE_FORWARD_MODELS` and use the THD/native path.
3. **mcore commit**: official README says ds4 recipes were tested with **Megatron-LM dev
   `35f36c7c9dba` + PR #4839** ("the copy in the NeMo container is NOT expected to work"). Our
   `megatron_lite/mcore-dev-latest` is `9af7c7937` — DIFFERENT. Check out the pinned commit.
4. moe_token_dispatcher_type="alltoall"; apply_rope_fusion = use_fused_kernels; recompute
   moe_act+mhc; EP8/PP4/TP1/ETP1; mixed_precision bf16_mixed (or bf16_with_*fp8 for quant).

## ⚠️ Reality check from the official README (sets the ceiling)
`examples/models/deepseek_v4/README.md`: DeepSeek-V4-Flash bridge verification =
**"last-token logit cosine 0.96-0.99 (short ~0.98, long >1024 tok ~0.96-0.99) vs official
inference."** So even CORRECTLY configured, bridge ds4 is a **~0.98-cosine** reference, NOT
bitwise. **mlite-vs-HF (0.998) already beats the official bridge bar.** => Do not expect a
glm5/kimi-style 2-3e-4/5e-6 "tight primary ref" from bridge ds4 on the FULL (hash-MoE) model.
The dense experiment is what might change this — test it, but calibrate expectations.

## The dense plan (bayan)
Force dense so routing-selection divergence is removed, identical config on all 3 backends:
- **num_hash_layers = 0** (no tid2eid path) — NOT currently a bench flag. Add it: mlite via
  `--impl-cfg-json` (deepseek_v4 config field `num_hash_layers`); bridge via
  `--override-transformer-json`; HF via the converter's config. Confirm all 3 honor it.
- **top-all = dense**: `num_experts_per_tok == n_routed_experts`. Easiest: `--keep-experts N`
  (shrinks n_routed to N) AND set `num_experts_per_tok = N` so top-N of N = all experts active.
  Note the bench keep-experts hook currently sets `num_experts_per_tok = min(old_topk, keep)`
  (= 6, not N) — needs a tweak/flag to force top-all. (Or bayan's alt: shared-expert-only.)
- truncate-layers 1-2, single-card proxy (keep-experts avoids the full-256 OOM).

## 3-way × (fwd+bwd) — the deliverable
Reuse existing harness (this dir): `bridge_forward.py` (dumps full logits via labels=None),
`compare3.py` (3-way cosine/mean_abs/CE), `mlite_forward.py`, `hf_forward.py`, the converter
in `hf_forward.py`/`ds4_dequant.py`. ADD backward: run a train step (labels set), capture
grad fingerprint (the bench `correctness.py run` already does grad_fingerprint + grad_norm;
use it for fwd+bwd loss/grad — I only did forward logits). Report per pair (fwd: logits
cosine/mean_abs/CE; bwd: grad max_abs/rel, loss): mlite-vs-bridge (primary, key),
mlite-vs-HF (0.998 baseline), HF-vs-bridge, all + run-to-run.

## Env (combined overlay for bridge; SM90 overlay for mlite — triton atomicrmw incompat)
See `run_bridge_forward.sbatch` / `run_mlite_forward.sbatch` headers and
[[mlite-bench-precision-reference-availability]] for the exact PYTHONPATH stack.
HF path: `code/models/DeepSeek-V4-Flash`. Artifacts/scratch: `code/ds4_vs_hf/`.

## Operator (parallel) is checking
NVIDIA-NeMo/Megatron-Bridge `examples/models/deepseek_v4/{README,conversion.sh,inference.sh}`
+ `tests/.../test_deepseek_v4_conversion.py` (the official verification flow = the correct
reference for how to drive bridge ds4). FP8: `cfg.mixed_precision="bf16_with_fp8_current_scaling_mixed"`.
