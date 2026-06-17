import torch, torch.nn.functional as F
V=129280
ids=torch.load('input_ids_dense.pt').reshape(-1).long()
def ce(path):
    lg=torch.load(path).float().reshape(-1,V); n=min(lg.shape[0], ids.numel()); lg=lg[:n]
    return F.cross_entropy(lg[:-1], ids[1:n]).item()
m=ce('mlite_logits_dense.pt'); h=ce('hf_logits_dense.pt'); b=ce('bridge_logits_dense.pt')
print(f"mlite CE={m:.6f}  HF CE={h:.6f}  bridge CE={b:.6f}", flush=True)
print(f"mlite-vs-HF {abs(m-h):.6f} ({abs(m-h)/h*100:.2f}%)", flush=True)
print(f"mlite-vs-bridge {abs(m-b):.6f} ({abs(m-b)/b*100:.2f}%)", flush=True)
print(f"bridge-vs-HF {abs(b-h):.6f} ({abs(b-h)/h*100:.2f}%)", flush=True)
