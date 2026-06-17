"""Bridge (megatron.bridge) ds4 reference forward — full logits dump.

Mirror of mlite_forward.py but on the Megatron-family primary reference
(megatron.bridge on latest mcore origin/dev). Builds the truncated real
DeepSeek-V4-Flash through the bench `bridge` runtime (dense BSHD, attention
backend flash, apply_rope_fusion OFF = the correct non-fused reference path),
forwards the SAME seeded tokens with labels=None so mcore GPTModel returns full
logits [1,S,V], and saves bridge_logits.pt for a logits-level mlite/bridge/HF
triangulation (compare.py).

Run under the combined overlay (ts512 + FHT-cu13 + megatron_bridge_latest +
mcore-dev-latest + develop SM90/flash_mla). EP4 to fit the full 256 experts.
"""
import argparse
import dataclasses
import os
import sys

import torch

_LITE = "/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/megatron_lite/mlite-ds4-hf/experimental/lite"
sys.path.insert(0, _LITE)

from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime import create_runtime

from examples.bench.bench import BenchCliConfig, build_runtime_config, build_session_config
from examples.bench.session import _infinite_packed_batches, _resolve_vocab_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--hf-path", default="/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/models/DeepSeek-V4-Flash")
    ap.add_argument("--out-logits", default="bridge_logits.pt")
    ap.add_argument("--check-input", default="input_ids.pt", help="assert tokens match this saved input")
    ap.add_argument("--rope-fusion", default="off", choices=["off", "on"],
                    help="apply_rope_fusion: 'off' (non-fused MLA rope) or 'on' (fused, train-only experimental)")
    ap.add_argument("--keep-experts", type=int, default=None)
    ap.add_argument("--num-hash-layers", type=int, default=None,
                    help="override hash-routed layer count (0 = dense / learned router)")
    ap.add_argument("--dense-topall", action="store_true",
                    help="num_experts_per_tok = routed-expert-count (all experts active)")
    ap.add_argument("--attention-backend", default="flash",
                    help="'flash' (bench default) or 'auto' (official ds4 recipe = None)")
    args = ap.parse_args()

    os.environ["MEGATRON_LITE_DETERMINISTIC"] = "1"
    set_deterministic(args.seed)

    import json
    # attention_backend: 'flash' = bench default; 'auto'/'none' -> null = the official
    # deepseek_v4 recipe (cfg.model.attention_backend = None, mcore auto-selects).
    if args.attention_backend in ("auto", "none", "null"):
        attn_backend = None
    else:
        attn_backend = args.attention_backend
    override = {
        "apply_rope_fusion": args.rope_fusion == "on",
        "attention_backend": attn_backend,
    }

    cfg = BenchCliConfig(
        backend="bridge",
        hf_path=args.hf_path,
        model_name="deepseek_v4",
        ep=args.ep,
        tp=1,
        etp=1,
        pp=1,
        cp=1,
        steps=1,
        num_microbatches=1,
        seq_len=args.seq_len,
        seed=args.seed,
        device="cuda",
        no_optimizer=True,
        skip_optimizer_build=True,
        truncate_layers=args.layers,
        keep_experts=args.keep_experts,
        disable_mtp=True,
        same_data_across_dp=True,
        num_hash_layers=args.num_hash_layers,
        dense_topall=args.dense_topall,
        override_transformer_json=json.dumps(override),
    )

    rt_cfg = build_runtime_config(cfg)
    rt = create_runtime(rt_cfg)
    handle = rt.build_model()
    session_cfg = build_session_config(cfg)

    vocab = _resolve_vocab_size(handle)
    batch = next(_infinite_packed_batches(vocab, session_cfg.seq_len, device="cuda", seed=session_cfg.seed))
    input_ids = batch.input_ids.detach().clone()
    # labels=None -> mcore GPTModel returns logits (not per-token loss).
    batch = dataclasses.replace(batch, labels=None)

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    with rt.eval_mode(handle):
        result = rt.forward_backward(handle, iter([batch]), loss_fn=None, num_microbatches=1, forward_only=True)
    logits = result.model_output.vocab_parallel_logits

    if rank == 0:
        # Sanity: identical tokens to the mlite/HF runs.
        if args.check_input and os.path.exists(args.check_input):
            saved = torch.load(args.check_input).reshape(-1).cpu()
            cur = input_ids.reshape(-1).cpu()[: saved.numel()]
            match = bool(torch.equal(saved.long(), cur.long()))
            print(f"[bridge] input_ids match {args.check_input}: {match}", flush=True)
        logits = logits.detach().float().cpu()
        torch.save(logits, args.out_logits)
        s = logits.reshape(-1)
        print(f"[bridge] logits shape={tuple(logits.shape)} "
              f"min={s.min():.5f} max={s.max():.5f} mean={s.mean():.5f} "
              f"first8={[round(float(x),4) for x in s[:8]]}", flush=True)
    print("DONE rank", rank, flush=True)


if __name__ == "__main__":
    main()
