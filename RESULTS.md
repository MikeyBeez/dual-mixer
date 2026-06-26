# Dual-Mixer / Mechanism-Dropout — Results (CORRECTED)

**Question:** can ONE set of weights run as either softmax attention or a state-space-style
linear mixer? **Answer: YES — fully-shared weights reach perfect recall in BOTH modes.**

> NOTE: an earlier version of this file concluded "no." That was a premature, under-trained
> result. The model groks late; stopping at 3–4k steps caught it in the flat pre-grok region.
> The correction is below and is the real finding.

## Setup

Tiny causal transformer: 2 layers, d=128, 4 heads (~441K params). Every layer SHARES its
Q/K/V/output/MLP weights; the only difference between mechanisms is how the shared Q,K,V are
combined — `attn` = softmax attention, `ssm` = causal linear attention. Task: MQAR
(multi-query associative recall, 32 key→value pairs). Run on pop (RTX 5070 Ti, torch 2.11+cu130).

## The correction: train to convergence, watch for grokking

The shared (per-forward) model's learning curve, attention-mode and SSM-mode accuracy:

```
 step   attn    ssm
  3000  0.072  0.082
  4000  0.091  0.084
  5000  1.000  0.084     <- attention mode GROKS (sudden)
  7000  1.000  0.152
  8000  1.000  0.925     <- ssm mode starts to grok
 12000  1.000  1.000     <- BOTH modes perfect
 20000  1.000  1.000     (stable plateau)
```

Flat near zero through ~4k steps, then a sharp phase transition — classic delayed
generalization. My first runs (3–4k steps) stopped inside the flat region and I wrongly
called it a failure.

## Corrected findings

- **Shared weights serve both.** Trained to ~12k steps, one fully-shared set of weights does
  1.000 MQAR recall in attention mode AND in SSM mode. (The collar run confirms shared@attn =
  1.000, gap +0.000 vs the attention specialist — and it needs no mechanism-specific params.)
- **Grokking is staggered:** attention mode at ~5k steps, the linear/SSM mode at ~12k.
- **Co-adaptation, not scaffolding (discriminator run).** Two specialists pin it down:
  an ssm-ONLY model never groks 32-pair recall (flat ~0.08 across 20k steps), AND an
  attn-ONLY model trained 15k steps reads out at 1.000 in attention mode but only ~0.04 in
  SSM mode. So neither mode's training alone yields a both-capable model: attention-trained
  weights are NOT SSM-readable, and SSM-mode training alone can't learn the task. Only JOINT
  training (both modes' gradients on the shared weights) gets the SSM mode to 1.000.
- **What this means.** It is NOT "attention learns the circuit and the SSM path reuses it"
  (that's refuted — attn-only weights give 0.04 in SSM mode). The SSM-mode gradients do real
  work co-adapting the shared weights; the joint constraint is what produces a solution both
  readouts can use. Consistent with the two mechanisms regularizing the shared solution rather
  than one scaffolding the other. (Whether it's *symmetric* — SSM-mode training also helping
  attention — is untestable here: attention is already perfect alone, and joint training if
  anything slightly delayed its grok, 5k vs 3k. That needs a task with complementary failure
  modes.)
- **The collar (rank-16 per-mechanism Q/K/V adapter) was not needed** for shared@attn; its
  effect on ssm grok speed was inconclusive at 8k steps.

## Takeaway

"Just train it on both" works for fully-shared weights — it just needs enough steps, and the
generalization arrives suddenly, not gradually. The earlier "gradient conflict kills it"
reading was an artifact of stopping early. Methodology note to self: always run a learning
curve to a plateau before concluding a negative, especially on small algorithmic tasks where
grokking is common.

## Files
- `mechanism_dropout.py` — 3-config comparison (attn-only / ssm-only / shared), with --collar.
- `convergence.py` — long shared run with a logged learning curve (this is what caught it).
- `results_perlayer.json`, `results_perforward.json` — the premature 3–4k-step runs.
- `conv.log` — the learning curve above. `mech_collar.json` — the collar run.
