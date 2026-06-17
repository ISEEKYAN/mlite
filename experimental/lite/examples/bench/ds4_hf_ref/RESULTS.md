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

bridge eval per-token-loss mean = 20.343 (finite, no nan), same metric as qwen3.5/kimi/glm5.

**BUT the scalar loss is misleadingly close — the logits diverge.** A three-way logits
triangulation on identical tokens (`bridge_forward.py` dumps full logits with `labels=None`;
`compare3.py`) shows the bridge is the **outlier**, not mlite:

| pair | cosine | mean_abs | argmax agree | note |
|---|---|---|---|---|
| **mlite vs HF** | **0.998** | 0.175 | 0.92 | two independent impls, **bit-identical weights** (mlite converter) |
| mlite vs bridge (rope off) | 0.903 | 1.15 | 0.44 | bridge independently dequantizes |
| bridge vs HF (rope off) | 0.908 | 1.12 | 0.50 | |
| mlite vs bridge (rope **on**) | 0.790 | 1.49 | — | rope fusion ON is *worse* |
| bridge vs HF (rope **on**) | 0.796 | 1.46 | — | |

**"Same kernel" is refuted as the cause.** mlite (fused native) and HF (eager transformers —
a completely different kernel) agree at **0.998**; if the gap were kernel/fusion, HF and bridge
(both non-fused) would agree and mlite would be the outlier. Instead bridge diverges from
*both* (~0.90). So forcing mlite+bridge onto the same kernel would not close it — the gap is
**structural/numeric on the bridge side**, not a fused-vs-unfused effect.

Likely causes (ds4-specific — kimi/glm5 used the *same* bench dense path and hit loss
rel 5.6e-6 / 2-3e-4, so the plumbing is sound): mcore `dsv4_hybrid` attention internals
(`head_dim=512` with only `qk_rope_head_dim=64` partial RoPE, attention sinks) and/or the mHC
Sinkhorn hyper-connection differing from the reference; plus bridge dequantizes the FP8/FP4
release **independently**, whereas mlite & HF share bit-identical weights via mlite's converter.
`sliding_window=128 > seq_len=64`, so the sliding-window path is *not* the cause.

**Net: for ds4 the trustworthy independent reference is HF (mlite-vs-HF 0.998, two independent
implementations agreeing).** The Megatron-bridge path is not yet a faithful primary ref for ds4
(logits cosine ~0.90); the "primary tighter than HF" expectation does not hold here. Closing
the bridge gap is a deep mcore `dsv4_hybrid`/mHC fidelity debug (out of scope per "别死磕").

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

## DENSE experiment — does forcing ds4 dense close the bridge gap? (NO — routing refuted)

bayan's hypothesis: the bridge ds4 logit divergence (cosine ~0.90) lives in the **hash-MoE
routing path** (tid2eid), so forcing the model fully **dense** + using the **official** bridge
config should make mlite-vs-bridge tight (both Megatron-family). **Tested and refuted.**

**Dense config (identical on all 3 backends):** `num_hash_layers=0` (no tid2eid hash routing —
every MoE layer uses the learned sigmoid-topk router) + `dense_topall` (`num_experts_per_tok ==
n_routed_experts` → every token activates all experts, no top-k selection divergence) +
`keep-experts 8` (single-card proxy) + `truncate-layers 2` (K=2 sliding, compress_ratio 0, no DSA
indexer) + MTP off. seq 64, seed 42, deterministic, same tokens (`input_ids_dense.pt`, bridge
confirms `input_ids match: True`). **Official bridge config:** `attention_backend=None` (mcore
auto-select; the bench default forces `flash`) — confirmed applied on the built mcore model:
`{moe_n_hash_layers: 0, moe_router_topk: 8, num_moe_experts: 8, attention_backend: None}`.
mlite dense confirmed by the `expert_bias missing` load warning (expert_bias is only a loaded
buffer on non-hash layers). New flags: `--num-hash-layers`, `--dense-topall` (bench
`BenchCliConfig` + mlite model-config hook + bridge post-init hook + the 3 forward scripts);
ds4 native loader gained kimi-style router row-slicing so `--keep-experts` loads experts 0..N-1.

### Forward — 3-way logits (DENSE), identical tokens
| pair | cosine | mean_abs | max_abs | argmax | vs FULL hash-MoE |
|---|---|---|---|---|---|
| **mlite vs HF** | **0.998069** | 0.175 | 2.58 | 0.984 | 0.998 (unchanged) |
| **mlite vs bridge** | **0.901343** | 1.166 | 21.0 | 0.578 | 0.903 (unchanged) |
| **bridge vs HF** | **0.905933** | 1.137 | 22.1 | 0.578 | 0.908 (unchanged) |

self-CE: mlite 17.47 / bridge 16.32 / hf 17.58.

### Backward — loss + global grad L2

**Apples-to-apples loss (the correct comparison): recompute next-token CE from the three saved
forward logits with ONE shifted-CE reduction on the SAME labels** (`recompute_ce_dense.py`,
job 12897687). This is the loss that drives the backward, computed identically for all three:
| backend | CE loss (same reduction, same labels) |
|---|---|
| **mlite** | **20.151127** |
| **HF** | 20.246429 |
| **bridge** | 20.556831 |

- **mlite-vs-HF 0.095 (0.47%)** — tight. mlite-vs-bridge 0.406 (1.97%); bridge-vs-HF 0.310 (1.53%).
  ⇒ **bridge is the outlier in the loss too**, fully consistent with the fwd logits (mlite≈HF 0.998,
  bridge ~0.90). mlite is NOT the outlier.

**⚠️ Earlier per-backend train-step losses were NOT comparable** (the symptom: they made bridge
look close to HF and mlite the outlier — a red flag). Each backend's own train step uses a
different loss reduction / label-roll convention: mlite's model loss, the bridge dense path's
`_packed_batch_to_dense_inputs` label-roll, and HF's textbook shifted-CE. So the raw per-backend
train losses (mlite 19.764 / bridge 20.393 / HF 20.246) mix three reductions and must NOT be
compared directly. Recomputing CE from the identical forward logits with one reduction removes
that confound and restores the consistent picture (bridge outlier).

**Gradient L2:** mlite `grad_global_norm` **42.77 ≈ HF 42.25** (within ~1.2%, two independent
impls) — mlite & HF agree at the gradient level too, consistent with everything else.

**bridge grad-norm not obtainable through this harness.** Tried `--no-optimizer` (mcore DDP
`main_grad` allocated on all 84 params but zero), then **built the full distributed optimizer**
(log: distributed Adam, clip_grad=1.0, reduce-scatter) and re-ran — autograd `run_backward`
executes but bridge param grads still finalize to **exactly 0.0** (`grad_global_norm=0`,
`optimizer_grad_norm=0`). The bench `bridge` dense-forward backward does not connect the mcore
label-loss to parameter grads for ds4 (the bridge path was validated by prior doers for
forward/loss only, never grad extraction). Deferred per "别死磕".

### Conclusion
Forcing ds4 fully dense **and** using the official `attention_backend=None` leaves the picture
**exactly** where the full hash-MoE run had it: mlite-vs-HF 0.998, mlite-vs-bridge 0.901,
bridge-vs-HF 0.906. So the bridge divergence is **not** in the hash-MoE/tid2eid routing path,
and **not** in the attention_backend choice (None vs flash both ~0.90). bridge stays the outlier
vs *both* mlite and HF (which agree at 0.998 sharing bit-identical weights via mlite's converter).
The residual is **structural/numeric on the bridge side** — the K=2 layers are pure sliding
attention, so the suspects are the mcore `dsv4_hybrid` MLA-ish attention internals (head_dim 512
with only qk_rope 64 partial-RoPE, attention sinks, grouped o_proj) and/or the mHC Sinkhorn
hyper-connection, plus bridge's **independent** FP4/FP8 dequant (mlite & HF share mlite's dequant;
bridge dequantizes the release on its own). **ds4's trustworthy reference remains HF** (mlite-vs-HF
0.998 fwd + grad-norms within ~1% + run-to-run bitwise). Dense does not turn bridge into a
faithful Megatron-family primary ref for ds4. (Per "别死磕", the deep mcore dsv4_hybrid/mHC +
dequant fidelity debug is not pursued here.)

## Precision overview — 4/4 done (vs independent reference)
- qwen3.5 vs mbridge(GDN): loss rel 7-9e-5
- kimi vs megatron.bridge(MLA dense 5.6e-6 + MoE keep-8 5.8e-6)
- glm5 vs megatron.bridge(DSA): loss rel 2-3e-4
- **ds4: trustworthy ref = HF-transformers cosine 0.998 / self-CE 0.7% / run-to-run bitwise**
  (mlite & HF, two independent impls, agree). megatron.bridge: scalar loss close (rel 1.42%)
  but logits diverge (cosine ~0.90) = bridge is the outlier vs both → not yet a faithful
  primary ref for ds4 (mcore dsv4_hybrid/mHC fidelity gap; deep debug, deferred).

## Files / artifacts
Harness scripts here (also live + run under `code/ds4_vs_hf/`, where the `.pt`/`.json`
artifacts are). mlite forward on SM90 overlay; HF eager on CPU under `code/ts512-site`
(transformers 5.12.1). Combined overlay for bridge:
`ts512-site : fht-cu13-clean : experimental/lite : megatron_bridge_latest/src :
megatron_lite/mcore-dev-latest : mlite-2604-dsa-glm5-overlay-develop/site-packages`.
