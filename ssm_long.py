#!/usr/bin/env python3
"""Does the SSM-only specialist eventually grok 32-pair MQAR on its own (20k steps)?
If it stays low, the shared model's ssm-mode success is genuine positive transfer."""
import random, torch, torch.nn.functional as F
from mechanism_dropout import Model, make_batch, evaluate

torch.manual_seed(0); random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d, heads, L, batch = 128, 4, 2, 64
p = dict(n_pairs=32, n_queries=16, n_keys=64, n_vals=64)
V, maxT = 128, 96
m = Model(V, d, heads, L, maxT).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
print("ssm-only long\n step    ssm", flush=True)
for step in range(20000):
    seq, mask, _ = make_batch(B=batch, device=dev, **p)
    logits = m(seq[:, :-1], ["ssm"] * L)
    tgt, lm = seq[:, 1:], mask[:, 1:]; sel = lm.reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, V)[sel], tgt.reshape(-1)[sel])
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 2000 == 0:
        print(f"{step+1:6d}  {evaluate(m, ['ssm']*L, batch, p, dev, 10):.3f}", flush=True)
