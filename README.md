# Dual-Mixer: one set of weights, two token-mixers

Can a single set of transformer weights run as either softmax attention OR a cheap
state-space (linear-attention) mixer? Train with "mechanism-dropout" — randomly pick the
mixer each step — and find out.

Headline: yes, and the two mechanisms **co-adapt**. Neither one's training alone yields a
dual-capable model; only the joint training does. Full story (grokking, the SSM transfer,
and the scaffolding-vs-co-adaptation discriminator) in RESULTS.md.

## Run (needs torch + CUDA)
- `python mechanism_dropout.py` — 3-config comparison (attn-only / ssm-only / shared)
- `python convergence.py --steps 20000` — the learning curve that revealed grokking
- `python cross_eval.py` — the discriminator: train attention alone, read it out in both modes
- `python ssm_long.py` — SSM-only never groks 32-pair recall on its own
