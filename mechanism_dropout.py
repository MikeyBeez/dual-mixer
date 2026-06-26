#!/usr/bin/env python3
"""
Mechanism-dropout PoC — can ONE set of weights run as either softmax attention OR a
state-space-style linear mixer?

A tiny causal transformer where every layer SHARES its Q/K/V/output/MLP weights but can
mix tokens two ways:
  attn : exact softmax attention   (good at exact recall)
  ssm  : causal linear attention   (cheap, fixed-size state, weak at recall)

The ONLY difference between the two mechanisms is how the shared Q,K,V are combined.

We train three ways on MQAR (multi-query associative recall — the task that separates them):
  attn-only, ssm-only, and SHARED (mechanism-dropout: each layer's mixer is chosen at
  random every step, so the weights must work under either). Then we evaluate the shared
  model forced fully into each mode.

Falsifiable question: does shared@attn match the attn-only specialist on recall?
If yes, one set of weights genuinely serves both mechanisms.
"""
import argparse, json, random, time
import torch, torch.nn as nn, torch.nn.functional as F


# --------------------------------------------------------------------------- data
def make_batch(B, n_pairs, n_queries, n_keys, n_vals, device):
    """MQAR: k0 v0 k1 v1 ... | q0 a0 q1 a1 ...  (loss only on the a_i answer tokens)."""
    V = n_keys + n_vals
    seqs, masks = [], []
    for _ in range(B):
        keys = torch.randperm(n_keys)[:n_pairs]
        vals = torch.randint(0, n_vals, (n_pairs,)) + n_keys
        ctx = torch.stack([keys, vals], 1).flatten()                 # k v k v ...
        qi = torch.randint(0, n_pairs, (n_queries,))
        qry = torch.stack([keys[qi], vals[qi]], 1).flatten()         # q a q a ...
        seq = torch.cat([ctx, qry])
        m = torch.zeros(len(seq), dtype=torch.bool)
        m[torch.arange(len(ctx) + 1, len(seq), 2)] = True            # answer positions
        seqs.append(seq); masks.append(m)
    return torch.stack(seqs).to(device), torch.stack(masks).to(device), V


# --------------------------------------------------------------------------- model
class Layer(nn.Module):
    def __init__(self, d, heads, collar=0):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.collar = collar
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        if collar:  # thin per-mechanism low-rank adapter on the shared Q/K/V (LoRA-style)
            self.adA = nn.ModuleDict({m: nn.Linear(3 * d, collar, bias=False) for m in ("attn", "ssm")})
            self.adB = nn.ModuleDict({m: nn.Linear(collar, 3 * d, bias=False) for m in ("attn", "ssm")})
            for m in ("attn", "ssm"):
                nn.init.zeros_(self.adB[m].weight)  # start as identity (collar off)

    def forward(self, x, mode):
        B, T, D = x.shape
        qkv = self.qkv(self.n1(x))
        if self.collar:
            qkv = qkv + self.adB[mode](self.adA[mode](qkv))          # mechanism-specific collar
        qkv = qkv.view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                              # [B,h,T,dh]
        if mode == "attn":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # ssm == causal linear attention (fixed-size running state)
            qf, kf = F.elu(q) + 1, F.elu(k) + 1
            kv = torch.einsum("bhtd,bhte->bhtde", kf, v).cumsum(2)    # running state
            z = kf.cumsum(2)
            num = torch.einsum("bhtd,bhtde->bhte", qf, kv)
            den = torch.einsum("bhtd,bhtd->bht", qf, z).clamp_min(1e-6).unsqueeze(-1)
            y = num / den
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.mlp(self.n2(x))
        return x


class Model(nn.Module):
    def __init__(self, V, d, heads, L, maxT, collar=0):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(maxT, d)
        self.layers = nn.ModuleList([Layer(d, heads, collar) for _ in range(L)])
        self.nf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx, modes):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for layer, m in zip(self.layers, modes):
            x = layer(x, m)
        return self.head(self.nf(x))


# --------------------------------------------------------------------------- train / eval
def train(model, mode_fn, steps, B, p, device, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        seq, mask, V = make_batch(B=B, device=device, **p)
        logits = model(seq[:, :-1], mode_fn())
        tgt, lm = seq[:, 1:], mask[:, 1:]
        sel = lm.reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, V)[sel], tgt.reshape(-1)[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, modes, B, p, device, batches=20):
    cor = tot = 0
    for _ in range(batches):
        seq, mask, V = make_batch(B=B, device=device, **p)
        pred = model(seq[:, :-1], modes).argmax(-1)
        tgt, lm = seq[:, 1:], mask[:, 1:]
        cor += (pred[lm] == tgt[lm]).sum().item(); tot += int(lm.sum())
    return cor / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--n-queries", type=int, default=16)
    ap.add_argument("--n-keys", type=int, default=64)
    ap.add_argument("--n-vals", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared-mode", default="perlayer", choices=["perlayer", "perforward"],
                    help="perlayer: each layer's mixer random; perforward: whole net one mixer/step")
    ap.add_argument("--collar", type=int, default=0,
                    help="rank of per-mechanism low-rank Q/K/V adapter (0 = fully shared, no collar)")
    ap.add_argument("--out", default="mechanism_dropout_results.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = dict(n_pairs=args.n_pairs, n_queries=args.n_queries,
             n_keys=args.n_keys, n_vals=args.n_vals)
    V = args.n_keys + args.n_vals
    maxT = 2 * args.n_pairs + 2 * args.n_queries
    L = args.layers
    print(f"device={device} vocab={V} seqlen={maxT} layers={L} d={args.d} "
          f"pairs={args.n_pairs} queries={args.n_queries}")

    def fresh():
        return Model(V, args.d, args.heads, L, maxT, args.collar).to(device)

    results = {}
    t0 = time.time()

    # 1) attention-only specialist
    m_attn = fresh()
    train(m_attn, lambda: ["attn"] * L, args.steps, args.batch, p, device)
    results["attn_only"] = evaluate(m_attn, ["attn"] * L, args.batch, p, device)

    # 2) ssm-only (linear-attention) specialist
    m_ssm = fresh()
    train(m_ssm, lambda: ["ssm"] * L, args.steps, args.batch, p, device)
    results["ssm_only"] = evaluate(m_ssm, ["ssm"] * L, args.batch, p, device)

    # 3) SHARED weights, mechanism-dropout
    if args.shared_mode == "perforward":
        shared_fn = lambda: [random.choice(["attn", "ssm"])] * L     # whole net, one mixer/step
    else:
        shared_fn = lambda: [random.choice(["attn", "ssm"]) for _ in range(L)]
    m_shared = fresh()
    train(m_shared, shared_fn, args.steps, args.batch, p, device)
    results["shared_mode"] = args.shared_mode
    results["shared_at_attn"] = evaluate(m_shared, ["attn"] * L, args.batch, p, device)
    results["shared_at_ssm"] = evaluate(m_shared, ["ssm"] * L, args.batch, p, device)

    results["params_per_model"] = sum(x.numel() for x in m_attn.parameters())
    results["seconds"] = round(time.time() - t0, 1)

    print("\n=== MQAR accuracy (answer tokens) ===")
    print(f"  attn-only specialist : {results['attn_only']:.3f}")
    print(f"  ssm-only  specialist : {results['ssm_only']:.3f}")
    print(f"  shared @ attn mode   : {results['shared_at_attn']:.3f}")
    print(f"  shared @ ssm  mode   : {results['shared_at_ssm']:.3f}")
    gap = results["attn_only"] - results["shared_at_attn"]
    print(f"\n  shared@attn vs attn-only gap: {gap:+.3f}  "
          f"({'shared weights serve attention' if gap < 0.05 else 'shared underperforms'})")
    print(f"  ({results['params_per_model']/1e3:.0f}K params/model, {results['seconds']}s total)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
