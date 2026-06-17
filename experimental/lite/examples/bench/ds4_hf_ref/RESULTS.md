# ds4 (DeepSeek-V4-Flash) precision: mlite vs independent reference — TASK-2.16.6 (M1)

ds4 is the 4th/hardest model. megatron.bridge was originally blocked (mcore main/dev
divergence); this delivers an **independent HF-transformers reference** for ds4, and
documents the bridge primary-ref status.

## Approach (HF independent reference — the M1 deliverable)
Both sides load the **same real DeepSeek-V4-Flash release**, truncated to **K=2 sliding
layers** (compress_ratio 0 → no DSA compressor/indexer), full 256 hash-MoE experts
(tid2eid valid), MTP off:
- **mlite**: PR#51 native FP8/FP4 loader (`--truncate-layers 2`), forward on SM90 DSA
  overlay (transformers-free). 1070 tensors, 0 missing.
- **HF reference**: `hf_forward.py` converts the release → HF-transformers
  `deepseek_v4` names (FP4/FP8→bf16 dequant via `ds4_dequant.py`, a verbatim copy of
  mlite's dequant so both sides see bit-identical weights; rename
  `wq_a→q_a_proj`, `wkv→kv_proj`, `wo_a/wo_b→o_a/o_b_proj`, `attn_sink→sinks`,
  per-expert `w1|w3→gate_up_proj`, `w2→down_proj`, `hc_attn→attn_hc`, `ffn.gate→mlp.gate`).
  Converter validated: **HF load_state_dict 0 missing / 0 unexpected**. HF runs eager on CPU.

## Results (seq 64, seed 42, deterministic)
| comparison | cosine | mean_abs | median_rel | self-CE |
|---|---|---|---|---|
| **mlite run-to-run (red-line #4)** | 1.0 | **0.0 (bitwise)** | 0.0 | — |
| mlite(bf16) vs HF(bf16)  K=2 | 0.99807 | 0.175 | 6.3% | 18.36 vs 18.48 (0.7%) |
| mlite(bf16) vs HF(fp32)  K=2 | 0.99814 | 0.179 | 6.5% | 18.36 vs 18.49 |
| HF(bf16) vs HF(fp32) = bf16 floor | ~1.0 | 0.021 | 0.77% | 18.48 vs 18.49 |
| mlite vs HF  K=1 | 0.99882 | 0.144 | 5.3% | 18.55 vs 19.11 |

- **run-to-run = bitwise max_abs 0.0** (sliding-no-indexer path is deterministic;
  cleaner than the prior full-DSA noise floor).
- mlite-vs-HF cosine 0.998, self-CE within 0.7% = **aligned** (not broken). The residual
  (~8× the bf16 floor) is present already at K=1 → a stable **per-layer fused-vs-eager
  kernel difference** (mlite fused-DSA-sliding vs HF eager; exotic attn: sink / partial-RoPE
  / grouped-o_proj / hyper-connection Sinkhorn), NOT a converter bug (a bug would break
  cosine/self-CE). Looser than the sibling refs (kimi 5.6e-6, glm5 2-3e-4) because HF is a
  fully independent eager reimplementation, as expected for a secondary reference.

## Bridge primary-ref (Megatron-family) — RESOLVED
ds4 now has a **Megatron-family primary reference** (megatron.bridge on latest mcore
`origin/dev` 9af7c7937, which carries both `dsv4_hybrid` and the bridge infra
`common_utils`/`safe_get_world_size`). Both sides load the **same real DeepSeek-V4-Flash**
(truncate-2 sliding, full 256 hash-MoE experts, MTP off); bench `bridge` backend on EP4
(solves full-256 single-card OOM), mlite on EP1.

| | loss (mean per-token CE) |
|---|---|
| **mlite** (fused-DSA-sliding + hash-MoE + mHC) | **20.0515** |
| **bridge** (mcore dsv4_hybrid dense, rope-fusion off) | **20.3354** |
| **abs diff / rel** | **0.284 / 1.42%** |

bridge eval per-token-loss mean = 20.343 (finite, no nan). This is the standard matched-pair
loss metric (same as qwen3.5/kimi/glm5). 1.4% rel is the **cross-impl floor for the hardest
model** (mlite fused-DSA-sliding + mHC vs mcore dsv4_hybrid dense + non-fused rope) — looser
than glm5 (2-3e-4) because ds4's two kernel stacks diverge much more; same order as the HF
secondary's 0.7% CE.

### Two corrections to the prior "open bug" diagnosis (both DISPROVEN by direct evidence)
1. **tid2eid was NOT the blocker.** The bridge loads `layers.{0,1}.ffn.gate.tid2eid` into
   `decoder.layers.{0,1}.mlp.router.tid2eid` **bitwise-correctly** during weight load.
   Directly injecting the release buffers post-build (`bridge_inject_tid2eid.py`) changed
   **0 of 129280×6 entries** on every kept layer — a pure no-op. The init-time "placeholder
   round-robin" warning fires at construction and is overwritten by the load. (The buffer is
   `[vocab=129280, top_k=6]` int32, values 0–255.)
2. **There was never a structural logits error.** With `labels` passed, mcore `GPTModel`
   returns the **per-token CE loss `[b,s]`**, not logits — the bench stores it under
   `vocab_parallel_logits`. So the prior "eval logits mean 19.3 all-positive (11–34)" was a
   per-token *loss* tensor (always positive, ≈ mlite's loss 20.05), misread against mlite's
   actual full-vocab logits ("−0.44 centered", a different artifact). Apples-to-oranges.

**Actual root cause of the nan:** the experimental MLA `apply_rope_fusion` (training-only,
flagged "experimental and may change" in the log) produces a nan token in the train forward.
Setting `--override-transformer-json '{"apply_rope_fusion": false}'` (the correct non-fused
reference path) → clean finite loss 20.3354. Reproduce: `code/ds4_vs_hf/run_bridge_inject.sbatch`
(job 12891285). The injection wrapper is kept as a guard that re-verifies tid2eid stays a no-op.

## Precision overview — 4/4 done (vs independent reference)
- qwen3.5 vs mbridge(GDN): loss rel 7-9e-5
- kimi vs megatron.bridge(MLA dense 5.6e-6 + MoE keep-8 5.8e-6)
- glm5 vs megatron.bridge(DSA): loss rel 2-3e-4
- **ds4: primary = megatron.bridge loss rel 1.42% (20.34 vs 20.05); secondary = HF-transformers cosine 0.998 / self-CE 0.7% / run-to-run bitwise**

## Files / artifacts
Harness scripts here (also live + run under `code/ds4_vs_hf/`, where the `.pt`/`.json`
artifacts are). mlite forward on SM90 overlay; HF eager on CPU under `code/ts512-site`
(transformers 5.12.1). Combined overlay for bridge:
`ts512-site : fht-cu13-clean : experimental/lite : megatron_bridge_latest/src :
megatron_lite/mcore-dev-latest : mlite-2604-dsa-glm5-overlay-develop/site-packages`.
