#!/usr/bin/env python3
"""
Convergence check: did shared@attn actually stop learning, or did I just stop early?

Trains the shared (per-forward) model for a long budget, logging attention-mode and
SSM-mode accuracy every eval_every steps — plus a short attn-only reference so we can see
how fast a real specialist converges for comparison.

If shared@attn stays pinned near the SSM level across the whole run, it has plateaued at
failure. If it climbs, the earlier conclusion was premature.
"""
import argparse, random
import torch, torch.nn.functional as F
from mechanism_dropout import Model, make_batch, evaluate


def curve_train(label, mode_fn, steps, eval_every, model, p, V, L, batch, device):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"\n== {label} ==", flush=True)
    print(" step   attn    ssm", flush=True)
    for step in range(steps):
        seq, mask, _ = make_batch(B=batch, device=device, **p)
        logits = model(seq[:, :-1], mode_fn())
        tgt, lm = seq[:, 1:], mask[:, 1:]
        sel = lm.reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, V)[sel], tgt.reshape(-1)[sel])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % eval_every == 0:
            a = evaluate(model, ["attn"] * L, batch, p, device, 10)
            s = evaluate(model, ["ssm"] * L, batch, p, device, 10)
            print(f"{step+1:6d}  {a:.3f}  {s:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--n-queries", type=int, default=16)
    ap.add_argument("--n-keys", type=int, default=64)
    ap.add_argument("--n-vals", type=int, default=64)
    a = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = dict(n_pairs=a.n_pairs, n_queries=a.n_queries, n_keys=a.n_keys, n_vals=a.n_vals)
    V = a.n_keys + a.n_vals
    maxT = 2 * a.n_pairs + 2 * a.n_queries
    L = a.layers
    print(f"device={device} steps={a.steps} layers={L} d={a.d}", flush=True)

    # reference: how fast does a real attention specialist converge?
    m_ref = Model(V, a.d, a.heads, L, maxT).to(device)
    curve_train("attn-only reference", lambda: ["attn"] * L, 3000, 500,
                m_ref, p, V, L, a.batch, device)

    # the real question: shared per-forward, trained LONG
    m_sh = Model(V, a.d, a.heads, L, maxT).to(device)
    curve_train("shared per-forward (LONG)",
                lambda: [random.choice(["attn", "ssm"])] * L,
                a.steps, a.eval_every, m_sh, p, V, L, a.batch, device)


if __name__ == "__main__":
    main()
