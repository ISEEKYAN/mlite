"""Run the bench correctness `run` for the bridge backend, injecting the real
DeepSeek-V4 hash-routing `tid2eid` buffers from the release safetensors.

Background (bayan's directive): the bench bridge load path does NOT fire the
bridge's `maybe_modify_loaded_hf_weight`/`_Tid2EidMapping`, so the mcore router
keeps a round-robin placeholder `tid2eid` -> hash routing is wrong -> bridge
forward is structurally broken (eval_logits mean ~19.3 all-positive, train
loss=nan, job 12890088/12890301). Rather than fix the generic loader we sync
the 3 saved buffers directly: read `layers.{0,1,2}.ffn.gate.tid2eid` from the
release safetensors and `copy_` into the matching mcore buffers
`decoder.layers.{N}.mlp.router.tid2eid`. Both sides then route on the SAME
table -> faithful routing comparison.

Usage: same argv as `examples.bench.correctness run --backend bridge ...`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

from examples.bench import correctness
from megatron.lite.runtime.backends.bridge import runtime as bridge_runtime

_TID2EID_RE = re.compile(r"layers\.(\d+)\.mlp\.router\.tid2eid$")


def _load_release_tid2eid(hf_path: str, layer_indices: set[int]) -> dict[int, torch.Tensor]:
    """Return {layer_index: tid2eid tensor} read from the release safetensors."""
    from safetensors import safe_open

    index_path = os.path.join(hf_path, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    out: dict[int, torch.Tensor] = {}
    for layer in sorted(layer_indices):
        key = f"layers.{layer}.ffn.gate.tid2eid"
        shard = weight_map.get(key)
        if shard is None:
            continue
        with safe_open(os.path.join(hf_path, shard), framework="pt") as sf:
            out[layer] = sf.get_tensor(key)
    return out


def _inject(model: torch.nn.Module, hf_path: str) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # Discover the tid2eid buffers actually present in this (possibly truncated) model.
    buffers = {}
    for name, buf in model.named_buffers():
        m = _TID2EID_RE.search(name)
        if m:
            buffers.setdefault(int(m.group(1)), []).append((name, buf))
    if not buffers:
        if rank == 0:
            print("[inject-tid2eid] WARNING: no *.mlp.router.tid2eid buffers found", flush=True)
        return

    release = _load_release_tid2eid(hf_path, set(buffers))
    for layer, entries in sorted(buffers.items()):
        src = release.get(layer)
        for name, buf in entries:
            if src is None:
                if rank == 0:
                    print(f"[inject-tid2eid] WARNING: no release tid2eid for layer {layer}", flush=True)
                continue
            if tuple(src.shape) != tuple(buf.shape):
                raise RuntimeError(
                    f"tid2eid shape mismatch for {name}: buffer {tuple(buf.shape)} vs release {tuple(src.shape)}"
                )
            before = buf.detach().clone()
            buf.copy_(src.to(device=buf.device, dtype=buf.dtype))
            if rank == 0:
                changed = int((before != buf).sum().item())
                print(
                    f"[inject-tid2eid] layer {layer} {name}: shape={tuple(buf.shape)} "
                    f"dtype={buf.dtype} changed_entries={changed} "
                    f"row0_before={before[0].tolist()} row0_after={buf[0].tolist()}",
                    flush=True,
                )


def main() -> None:
    # Parse only the bits we need; correctness.main() re-parses the full argv.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--hf-path")
    known, _ = ap.parse_known_args()
    hf_path = known.hf_path

    orig_build = bridge_runtime.BridgeRuntime.build_model

    def patched_build(self, *args, **kwargs):
        handle = orig_build(self, *args, **kwargs)
        path = hf_path or getattr(self, "_hf_path", None)
        model = handle._extras["model_list"][0]
        _inject(model, path)
        return handle

    bridge_runtime.BridgeRuntime.build_model = patched_build
    try:
        correctness.main()
    finally:
        bridge_runtime.BridgeRuntime.build_model = orig_build


if __name__ == "__main__":
    main()
