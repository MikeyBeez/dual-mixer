#!/usr/bin/env python3
"""
Discriminator: did attention 'scaffold' the SSM path, or did SSM-mode training do real work?

Train with ATTENTION ONLY (no ssm-mode gradients), long, and evaluate in BOTH modes every 1k
steps.
  - If @ssm rises to ~1.0 on its own -> attention-trained weights are directly readable by the
    SSM mixer; attention did the work, SSM-mode training was redundant (scaffold/transfer).
  - If @ssm stays low while @attn hits 1.0 -> the shared model's SSM success REQUIRED the
    SSM-mode training steps. Those steps co-adapted / regularized the shared weights. (Mikey's
    reading.)
"""
import random, torch, torch.nn.functional as F
from mechanism_dropout import Model, make_batch, evaluate

torch.manual_seed(0); random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d, heads, L, batch = 128, 4, 2, 64
p = dict(n_pairs=32, n_queries=16, n_keys=64, n_vals=64)
V, maxT = 128, 96

m = Model(V, d, heads, L, maxT).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
print("ATTENTION-ONLY training, evaluated in both modes")
print(" step   @attn   @ssm", flush=True)
for step in range(15000):
    seq, mask, _ = make_batch(B=batch, device=dev, **p)
    logits = m(seq[:, :-1], ["attn"] * L)            # ONLY attention-mode gradients
    tgt, lm = seq[:, 1:], mask[:, 1:]; sel = lm.reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, V)[sel], tgt.reshape(-1)[sel])
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 1000 == 0:
        a = evaluate(m, ["attn"] * L, batch, p, dev, 10)
        s = evaluate(m, ["ssm"] * L, batch, p, dev, 10)
        print(f"{step+1:6d}  {a:.3f}  {s:.3f}", flush=True)
