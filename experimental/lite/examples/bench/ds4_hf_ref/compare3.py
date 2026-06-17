"""Three-way ds4 logits triangulation on IDENTICAL tokens.

mlite (fused native) and HF (eager transformers) share bit-identical weights via
mlite's release->HF converter, so their agreement isolates the *forward kernels*.
bridge (megatron.bridge mcore dsv4_hybrid) dequantizes the release independently.
Run on login node (CPU): python3 compare3.py
"""
import torch
import torch.nn.functional as F

V = 129280


def stats(name, a, c):
    a, c = a.reshape(-1).float(), c.reshape(-1).float()
    n = min(a.numel(), c.numel())
    a, c = a[:n], c[:n]
    d = (a - c).abs()
    cos = F.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)).item()
    print(f"{name:34s} cosine={cos:.6f}  mean_abs={d.mean():.4f}  max_abs={d.max():.4f}")


def selfce(x):
    lg = x.reshape(-1, V)
    return F.cross_entropy(lg[:-1], lg[1:].argmax(-1)).item()


def argmax_agree(a, c):
    return (a.reshape(-1, V).argmax(-1) == c.reshape(-1, V).argmax(-1)).float().mean().item()


m = torch.load("mlite_logits.pt").float()
h = torch.load("hf_logits.pt").float()
b = torch.load("bridge_logits.pt").float()           # rope fusion OFF (best)
print("--- logits (same tokens) ---")
stats("mlite vs HF      (share weights)", m, h)
stats("mlite vs bridge", m, b)
stats("bridge vs HF", b, h)
print(f"self-CE:  mlite={selfce(m):.4f}  bridge={selfce(b):.4f}  hf={selfce(h):.4f}")
print(f"argmax agree: mlite-hf={argmax_agree(m,h):.3f}  mlite-bridge={argmax_agree(m,b):.3f}  bridge-hf={argmax_agree(b,h):.3f}")
