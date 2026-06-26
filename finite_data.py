#!/usr/bin/env python3
"""
Does mechanism-dropout act as a regularizer that improves the STRONG model?

With infinite fresh data, attention already hits 1.000 (no headroom). Here we use a FIXED
small training set + a held-out test set, so attention can overfit and there is room for a
regularizer to help generalization. Compare attn-only vs shared@attn on TEST accuracy.
(Small data + weight decay is the classic grokking / overfitting regime.)
"""
import random, torch, torch.nn.functional as F
from mechanism_dropout import Model, make_batch

torch.manual_seed(0); random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d, heads, L, batch = 128, 4, 2, 64
p = dict(n_pairs=32, n_queries=16, n_keys=64, n_vals=64)
V, maxT = 128, 96
TRAIN_N, TEST_N = 256, 512

tr_seq, tr_mask, _ = make_batch(B=TRAIN_N, device=dev, **p)   # fixed training pool
te_seq, te_mask, _ = make_batch(B=TEST_N, device=dev, **p)    # held-out test pool


def minibatch():
    i = torch.randint(0, TRAIN_N, (batch,), device=dev)
    return tr_seq[i], tr_mask[i]


@torch.no_grad()
def acc(model, modes, seq, mask):
    pred = model(seq[:, :-1], modes).argmax(-1)
    tgt, lm = seq[:, 1:], mask[:, 1:]
    return (pred[lm] == tgt[lm]).float().mean().item()


def run(label, mode_fn, eval_modes, steps=20000):
    torch.manual_seed(0)
    m = Model(V, d, heads, L, maxT).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    print(f"\n== {label} ==   step  train   test", flush=True)
    for step in range(steps):
        seq, mask = minibatch()
        logits = m(seq[:, :-1], mode_fn())
        tgt, lm = seq[:, 1:], mask[:, 1:]; sel = lm.reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, V)[sel], tgt.reshape(-1)[sel])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 2000 == 0:
            tr = acc(m, eval_modes, tr_seq, tr_mask)
            te = acc(m, eval_modes, te_seq, te_mask)
            print(f"{step+1:6d}   {tr:.3f}  {te:.3f}", flush=True)
    return acc(m, eval_modes, te_seq, te_mask)


a = run("attn-only", lambda: ["attn"] * L, ["attn"] * L)
s = run("shared (mechanism-dropout)", lambda: [random.choice(["attn", "ssm"])] * L, ["attn"] * L)
print(f"\nFINAL held-out TEST @attn:  attn-only={a:.3f}  shared={s:.3f}  delta={s-a:+.3f}",
      flush=True)
