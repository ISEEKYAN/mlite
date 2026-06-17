"""Compare mlite vs HF ds4 logits (+ optional run-to-run). TASK-2.16.6."""
import argparse
import torch
import torch.nn.functional as F


def stats(name, a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    abs_diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    rel = (abs_diff / denom)
    print(f"=== {name} ===")
    print(f"  shapes equal numel: {a.numel()}")
    print(f"  max_abs={abs_diff.max():.6e}  mean_abs={abs_diff.mean():.6e}")
    print(f"  max_rel={rel.max():.6e}  median_rel={rel.median():.6e}")
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"  cosine={cos:.8f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlite", default="mlite_logits.pt")
    ap.add_argument("--hf", default="hf_logits.pt")
    ap.add_argument("--mlite2", default=None, help="second mlite run for run-to-run")
    args = ap.parse_args()

    m = torch.load(args.mlite).float()
    h = torch.load(args.hf).float()
    print("mlite logits shape:", tuple(m.shape), "hf logits shape:", tuple(h.shape))
    # align: hf is [1,S,V], mlite may be [S,V] or [1,S,V]
    stats("mlite-vs-HF logits", m, h)

    # cross-entropy "loss" proxy using next-token shift on each, then compare the loss scalars
    def loss_of(logits):
        lg = logits.reshape(-1, logits.shape[-1])
        # predict token i+1 from position i (teacher-forced on argmax of the other? )
        # use self next-token as label to get a scalar that depends on the full dist
        labels = lg[1:].argmax(-1)
        return F.cross_entropy(lg[:-1], labels).item()

    print(f"mlite self-CE={loss_of(m):.6f}  hf self-CE={loss_of(h):.6f}")

    if args.mlite2:
        m2 = torch.load(args.mlite2).float()
        stats("mlite run-to-run (run1 vs run2)", m, m2)


if __name__ == "__main__":
    main()
