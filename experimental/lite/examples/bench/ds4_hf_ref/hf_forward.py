"""HF transformers deepseek_v4 reference forward (ds4-vs-HF precision, TASK-2.16.6).

Reads the real DeepSeek-V4-Flash release, converts release->HF-transformers names
(dequant FP4/FP8 -> fp32 via mlite's exact helpers), loads into a truncated HF
DeepseekV4ForCausalLM, forwards the SAME input_ids mlite used, saves logits.

HF deepseek_v4 is eager-only -> run on CPU (no GPU/kernel deps; fully deterministic).
"""
import argparse
import json
import sys

import torch
from safetensors import safe_open

from ds4_dequant import dequantize_scaled_tensor

HF_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/models/DeepSeek-V4-Flash"


def build_reader(hf_dir):
    idx = json.load(open(f"{hf_dir}/model.safetensors.index.json"))["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(f"{hf_dir}/{f}", framework="pt")
        return handles[f].get_tensor(name)

    def has(name):
        return name in idx

    return get, has


def read_dequant(get, has, name, target_shape):
    """Read release weight `name` (+ optional .scale) -> fp32 tensor of target_shape."""
    w = get(name)
    base = name[:-7] if name.endswith(".weight") else name
    scale_name = f"{base}.scale"
    if has(scale_name):
        return dequantize_scaled_tensor(w, get(scale_name), torch.Size(target_shape))
    return w.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--input", default="input_ids.pt")
    ap.add_argument("--out", default="hf_logits.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    PARAM_DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    hf_cfg_dict = json.load(open(f"{HF_DIR}/config.json"))
    K = args.layers
    cfg = DeepseekV4Config(**{k: v for k, v in hf_cfg_dict.items()
                              if k not in ("architectures", "quantization_config", "torch_dtype",
                                           "transformers_version", "_name_or_path")})
    cfg.num_hidden_layers = K
    cfg.num_nextn_predict_layers = 0
    cfg.compress_ratios = list(hf_cfg_dict["compress_ratios"][:K])
    cfg.layer_types = None
    cfg.mlp_layer_types = None
    cfg.__post_init__(compress_ratios=cfg.compress_ratios,
                      num_hash_layers=hf_cfg_dict.get("num_hash_layers", 3))
    print("layer_types:", cfg.layer_types, "mlp_layer_types:", cfg.mlp_layer_types, flush=True)
    assert all(t == "sliding_attention" for t in cfg.layer_types), \
        "this converter slice only covers sliding layers (K<=2); add CSA/HCA for K>2"

    # Build on meta to read exact target shapes/keys without allocating.
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(cfg)
    sd_meta = model.state_dict()

    def want_dtype(key):
        # All-bf16 to mirror mlite's bf16 forward (the hc / RMSNorm forwards upcast to
        # fp32 internally via .float(), so bf16-stored norm/hc weights still compute in
        # fp32) and to avoid fp32-norm-output -> bf16-linear dtype mismatches.
        if key.endswith(".tid2eid"):
            return torch.int64
        return PARAM_DTYPE

    get, has = build_reader(HF_DIR)
    n_experts = cfg.n_routed_experts
    inter = cfg.moe_intermediate_size
    new_sd = {}

    # --- top level ---
    new_sd["model.embed_tokens.weight"] = read_dequant(get, has, "embed.weight", sd_meta["model.embed_tokens.weight"].shape)
    new_sd["model.norm.weight"] = read_dequant(get, has, "norm.weight", sd_meta["model.norm.weight"].shape)
    new_sd["lm_head.weight"] = read_dequant(get, has, "head.weight", sd_meta["lm_head.weight"].shape)
    for a, b in (("hc_base", "hc_head_base"), ("hc_fn", "hc_head_fn"), ("hc_scale", "hc_head_scale")):
        new_sd[f"model.hc_head.{a}"] = read_dequant(get, has, b, sd_meta[f"model.hc_head.{a}"].shape)

    ATTN = {"q_a_proj.weight": "wq_a.weight", "q_a_norm.weight": "q_norm.weight",
            "q_b_proj.weight": "wq_b.weight", "kv_proj.weight": "wkv.weight",
            "kv_norm.weight": "kv_norm.weight", "o_a_proj.weight": "wo_a.weight",
            "o_b_proj.weight": "wo_b.weight", "sinks": "attn_sink"}
    HC = {"attn_hc.base": "hc_attn_base", "attn_hc.fn": "hc_attn_fn", "attn_hc.scale": "hc_attn_scale",
          "ffn_hc.base": "hc_ffn_base", "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.scale": "hc_ffn_scale"}

    for L in range(K):
        hp = f"model.layers.{L}"
        rp = f"layers.{L}"
        new_sd[f"{hp}.input_layernorm.weight"] = read_dequant(get, has, f"{rp}.attn_norm.weight", sd_meta[f"{hp}.input_layernorm.weight"].shape)
        new_sd[f"{hp}.post_attention_layernorm.weight"] = read_dequant(get, has, f"{rp}.ffn_norm.weight", sd_meta[f"{hp}.post_attention_layernorm.weight"].shape)
        for hk, rk in ATTN.items():
            new_sd[f"{hp}.self_attn.{hk}"] = read_dequant(get, has, f"{rp}.attn.{rk}", sd_meta[f"{hp}.self_attn.{hk}"].shape)
        for hk, rk in HC.items():
            new_sd[f"{hp}.{hk}"] = read_dequant(get, has, f"{rp}.{rk}", sd_meta[f"{hp}.{hk}"].shape)
        # router
        new_sd[f"{hp}.mlp.gate.weight"] = read_dequant(get, has, f"{rp}.ffn.gate.weight", sd_meta[f"{hp}.mlp.gate.weight"].shape)
        if has(f"{rp}.ffn.gate.tid2eid"):
            new_sd[f"{hp}.mlp.gate.tid2eid"] = get(f"{rp}.ffn.gate.tid2eid").long()
        if has(f"{rp}.ffn.gate.bias") and f"{hp}.mlp.gate.e_score_correction_bias" in sd_meta:
            new_sd[f"{hp}.mlp.gate.e_score_correction_bias"] = read_dequant(get, has, f"{rp}.ffn.gate.bias", sd_meta[f"{hp}.mlp.gate.e_score_correction_bias"].shape)
        # shared experts
        for hk, rk in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
            new_sd[f"{hp}.mlp.shared_experts.{hk}.weight"] = read_dequant(get, has, f"{rp}.ffn.shared_experts.{rk}.weight", sd_meta[f"{hp}.mlp.shared_experts.{hk}.weight"].shape)
        # routed experts: stack per-expert (w1|w3)->gate_up_proj, w2->down_proj
        gate_up = torch.empty(sd_meta[f"{hp}.mlp.experts.gate_up_proj"].shape, dtype=torch.float32)
        down = torch.empty(sd_meta[f"{hp}.mlp.experts.down_proj"].shape, dtype=torch.float32)
        for e in range(n_experts):
            w1 = read_dequant(get, has, f"{rp}.ffn.experts.{e}.w1.weight", (inter, cfg.hidden_size))
            w3 = read_dequant(get, has, f"{rp}.ffn.experts.{e}.w3.weight", (inter, cfg.hidden_size))
            w2 = read_dequant(get, has, f"{rp}.ffn.experts.{e}.w2.weight", (cfg.hidden_size, inter))
            gate_up[e] = torch.cat([w1, w3], dim=0)
            down[e] = w2
        new_sd[f"{hp}.mlp.experts.gate_up_proj"] = gate_up
        new_sd[f"{hp}.mlp.experts.down_proj"] = down
        print(f"layer {L} converted", flush=True)

    # cast + key audit
    missing = sorted(set(sd_meta) - set(new_sd))
    unexpected = sorted(set(new_sd) - set(sd_meta))
    print("MISSING in converted (HF expects, not provided):", missing, flush=True)
    print("UNEXPECTED in converted (provided, HF doesn't want):", unexpected, flush=True)
    for k in list(new_sd):
        if k in sd_meta:
            new_sd[k] = new_sd[k].to(want_dtype(k))

    # materialize model on real device + load
    model = DeepseekV4ForCausalLM(cfg)
    res = model.load_state_dict(new_sd, strict=False, assign=True)
    print("load_state_dict missing:", res.missing_keys, flush=True)
    print("load_state_dict unexpected:", res.unexpected_keys, flush=True)
    model = model.to(args.device).eval()

    input_ids = torch.load(args.input).to(args.device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [S] -> [1, S]
    print("input_ids", tuple(input_ids.shape), input_ids.dtype, flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits.float().cpu()
    torch.save(logits, args.out)
    s = logits.reshape(-1)
    print(f"HF logits shape={tuple(logits.shape)} min={s.min():.5f} max={s.max():.5f} "
          f"mean={s.mean():.5f} first8={[round(float(x),4) for x in s[:8]]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
