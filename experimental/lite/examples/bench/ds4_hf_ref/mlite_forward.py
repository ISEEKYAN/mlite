"""mlite ds4 reference forward (ds4-vs-HF precision, TASK-2.16.6).

Builds a truncated (K-layer) real DeepSeek-V4-Flash via the mlite runtime, loads
the real release weights (FP8/FP4 dequant via the PR#51 native loader, full experts
so hash routing stays valid), forwards a fixed seeded input, saves logits + the
input_ids (1-D) so the HF reference forwards the identical tokens.

Run under the SM90 DSA overlay (fused ds4 forward), transformers-free.
"""
import argparse
import os
import sys
from pathlib import Path

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
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--hf-path", default="/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/models/DeepSeek-V4-Flash")
    ap.add_argument("--out-logits", default="mlite_logits.pt")
    ap.add_argument("--out-input", default="input_ids.pt")
    ap.add_argument("--run-tag", default="run1")
    ap.add_argument("--keep-experts", type=int, default=None)
    ap.add_argument("--num-hash-layers", type=int, default=None,
                    help="override hash-routed layer count (0 = dense / learned router)")
    ap.add_argument("--dense-topall", action="store_true",
                    help="num_experts_per_tok = routed-expert-count (all experts active)")
    ap.add_argument("--backward", action="store_true",
                    help="run a train step (loss + global grad L2) instead of the logits dump")
    args = ap.parse_args()

    os.environ["MEGATRON_LITE_DETERMINISTIC"] = "1"
    set_deterministic(args.seed)

    cfg = BenchCliConfig(
        backend="mlite",
        hf_path=args.hf_path,
        model_name="deepseek_v4",
        impl="lite",
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
        num_hash_layers=args.num_hash_layers,
        dense_topall=args.dense_topall,
        impl_cfg_json='{"optimizer": null, "mtp_enable": false, "mtp_enable_train": false}',
    )

    rt_cfg = build_runtime_config(cfg)
    rt = create_runtime(rt_cfg)
    handle = rt.build_model()
    session_cfg = build_session_config(cfg)

    vocab = _resolve_vocab_size(handle)
    data_seed = session_cfg.seed
    import dataclasses
    batch = next(_infinite_packed_batches(vocab, session_cfg.seq_len, device="cuda", seed=data_seed))
    input_ids = batch.input_ids.detach().clone()
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    if args.backward:
        # bwd leg: train step on the SAME tokens (labels kept = next-token, set by the
        # packed-batch generator); report loss + global grad L2 (backend-comparable).
        from examples.bench.correctness import _grad_global_norm
        rt.zero_grad(handle)
        with rt.train_mode(handle):
            result = rt.forward_backward(handle, iter([batch]), loss_fn=None, num_microbatches=1)
        if rank == 0:
            loss = float(result.metrics.get("loss", 0.0))
            print(f"[{args.run_tag}] mlite BWD loss={loss:.6f} grad_global_norm={_grad_global_norm(handle):.6f}",
                  flush=True)
        print("DONE rank", rank, flush=True)
        return

    # ds4 model returns logits only when labels is None (else it returns the loss).
    batch = dataclasses.replace(batch, labels=None)
    with rt.eval_mode(handle):
        result = rt.forward_backward(handle, iter([batch]), loss_fn=None, num_microbatches=1, forward_only=True)
    logits = result.model_output.vocab_parallel_logits

    if rank == 0:
        logits = logits.detach().float().cpu()
        torch.save(input_ids.cpu(), args.out_input)
        torch.save(logits, args.out_logits)
        s = logits.reshape(-1)
        print(f"[{args.run_tag}] mlite logits shape={tuple(logits.shape)} "
              f"min={s.min():.5f} max={s.max():.5f} mean={s.mean():.5f} "
              f"first8={[round(float(x),4) for x in s[:8]]}", flush=True)
        print("input_ids first16:", input_ids[:16].tolist(), flush=True)
    print("DONE rank", rank, flush=True)


if __name__ == "__main__":
    main()
