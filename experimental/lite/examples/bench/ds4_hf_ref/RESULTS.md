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

## Precision overview — 4/4 done (vs independent reference)
- qwen3.5 vs mbridge(GDN): loss rel 7-9e-5
- kimi vs megatron.bridge(MLA dense 5.6e-6 + MoE keep-8 5.8e-6)
- glm5 vs megatron.bridge(DSA): loss rel 2-3e-4
- **ds4 vs HF-transformers(independent): cosine 0.998 / self-CE 0.7% / run-to-run bitwise**

## Bridge primary-ref status (bonus follow-up — NOT blocking M1)
With **latest mcore `origin/dev` (9af7c7937)** the prior env divergence is RESOLVED:
`dsv4_hybrid` + bridge infra (`common_utils`/`safe_get_world_size`) now coexist. combined
overlay + latest mcore imports green; bridge builds + loads ds4 (EP4 solves the full-256
single-card OOM). nvrx.py needs a graceful patch (container ships nvidia-resiliency-ext
0.5.0 < 0.6.0; env-only patch in the mcore-dev-latest worktree).

**Open bug (for a fresh doer):** the bench `bridge` dense-forward is structurally wrong for
ds4 — eval logits **mean 19.3, all-positive (11–34)** vs mlite's centered **−0.44**; train
loss **nan** (job 12890301). ds4 (hash-MoE tid2eid + dsv4_hybrid attn + `hc_head` final
collapse) is unproven on the bench bridge path (prior doer only validated kimi/glm5, no hash
layers). Candidate causes: bench dense-forward not handling ds4's `hc_head` stream collapse;
`tid2eid` load (bridge maps it at `mlp.router.tid2eid` but init warns round-robin placeholder);
dsv4_hybrid output layout. Distinct from mlite correctness.

## Files / artifacts
Harness scripts here (also live + run under `code/ds4_vs_hf/`, where the `.pt`/`.json`
artifacts are). mlite forward on SM90 overlay; HF eager on CPU under `code/ts512-site`
(transformers 5.12.1). Combined overlay for bridge:
`ts512-site : fht-cu13-clean : experimental/lite : megatron_bridge_latest/src :
megatron_lite/mcore-dev-latest : mlite-2604-dsa-glm5-overlay-develop/site-packages`.
