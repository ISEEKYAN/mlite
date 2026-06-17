"""Dump HF transformers deepseek_v4 state_dict keys+shapes (truncated config) and
the real DeepSeek-V4-Flash release tensor shapes for layers 0-1, to design the
release->HF-transformers converter (ds4-vs-HF precision, TASK-2.16.6)."""
import json
import sys

import torch
from safetensors import safe_open

HF_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/models/DeepSeek-V4-Flash"
K = 2          # truncate to 2 sliding layers
N_EXPERTS = 8  # small expert count just to see packing structure

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

hf_cfg_dict = json.load(open(f"{HF_DIR}/config.json"))
# keep only fields the dataclass accepts is handled by transformers; pass through.
cfg = DeepseekV4Config(**{k: v for k, v in hf_cfg_dict.items()
                          if k not in ("architectures", "quantization_config", "torch_dtype",
                                       "transformers_version", "_name_or_path")})
cfg.num_hidden_layers = K
cfg.n_routed_experts = N_EXPERTS
cfg.num_experts_per_tok = min(cfg.num_experts_per_tok, N_EXPERTS)
cfg.num_nextn_predict_layers = 0
# truncate per-layer schedules
cfg.compress_ratios = list(hf_cfg_dict["compress_ratios"][:K])
cfg.layer_types = None
cfg.mlp_layer_types = None
# re-run post_init to rederive layer_types/mlp_layer_types from truncated compress_ratios
cfg.__post_init__(compress_ratios=cfg.compress_ratios, num_hash_layers=hf_cfg_dict.get("num_hash_layers", 3))

print("=== derived schedule ===")
print("layer_types:", cfg.layer_types)
print("mlp_layer_types:", cfg.mlp_layer_types)
print("qk_rope_head_dim:", getattr(cfg, "qk_rope_head_dim", None), "head_dim:", cfg.head_dim)

try:
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(cfg)
    src = "meta"
except Exception as e:  # noqa: BLE001
    print("meta build failed, retry tiny CPU:", repr(e))
    cfg.hidden_size = 256
    cfg.moe_intermediate_size = 128
    cfg.vocab_size = 1000
    cfg.q_lora_rank = 128
    cfg.o_lora_rank = 128
    cfg.head_dim = 128
    model = DeepseekV4ForCausalLM(cfg)
    src = "cpu-tiny"

print(f"\n=== HF state_dict keys+shapes (src={src}) ===")
sd = model.state_dict()
for k in sorted(sd):
    print(f"  {k}  {tuple(sd[k].shape)}  {sd[k].dtype}")

print("\n=== HF persistent buffers (subset of state_dict) vs non-persistent ===")
persistent = set(sd.keys())
for name, buf in model.named_buffers():
    tag = "persistent" if name in persistent else "NON-persistent(not in sd)"
    print(f"  {name}  {tuple(buf.shape)}  {tag}")

print("\n=== release layer-0 tensor shapes (real dims) ===")
idx = json.load(open(f"{HF_DIR}/model.safetensors.index.json"))["weight_map"]
files = {}
for k, f in idx.items():
    files.setdefault(f, []).append(k)
want = [k for k in idx if k.startswith("layers.0.") and ".experts." not in k] + \
       ["embed.weight", "norm.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale",
        "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w2.weight", "layers.0.ffn.experts.0.w3.weight",
        "layers.0.ffn.experts.0.w1.scale", "layers.0.ffn.experts.0.w2.scale", "layers.0.ffn.experts.0.w3.scale"]
shapes = {}
for k in want:
    f = idx.get(k)
    if f is None:
        shapes[k] = "MISSING-in-index"
        continue
    with safe_open(f"{HF_DIR}/{f}", framework="pt") as fh:
        t = fh.get_slice(k)
        shapes[k] = (tuple(t.get_shape()), t.get_dtype())
for k in sorted(shapes):
    print(f"  {k}  {shapes[k]}")
print("\nDONE")
