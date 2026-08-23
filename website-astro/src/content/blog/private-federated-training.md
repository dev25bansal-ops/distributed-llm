---
title: "Training on pooled devices without sharing your data"
description: "A walkthrough of DistLLM's federated fine-tuning: LoRA adapters trained in place, differentially-private gradients over gossip, and RDP accounting that tells you when to stop."
pubDate: 2026-08-24
tags: [engineering, privacy]
draft: true
---

## From pooling inference to pooling learning

[Last time](/blog/pooling-consumer-gpus) we looked at how DistLLM splits a
model across the machines already in your home — a gaming desktop, a laptop,
a mini-PC — and serves inference from the chain. That solves *serving*. But
there is a second problem hiding behind it: those same households are full
of private data nobody wants to upload anywhere. Photos, documents,
messages, code.

The usual answer is "collect it all in the cloud, train there." The
federated answer is the inverse: **the model travels, the data stays
home**. Each device trains a small LoRA adapter on its own data, and only
gradient updates ever cross the network. DistLLM ships this today in
`FederatedFineTuner`, and this post walks through what actually happens
underneath — because "federated" claims are cheap and the details are where
privacy lives or dies.

## A round, end to end

Every federated round runs five steps on each node:

1. **Stale-gradient pruning.** Contributions left over from earlier rounds
   are dropped before averaging. On consumer networks this matters more
   than you'd think: peers vanish mid-round, gossip messages arrive late
   and out of order. Every gradient message carries the round number it was
   produced for, and a node will refuse to average a contribution whose
   round doesn't match the one in progress — checked at prune time, again
   at receipt, and once more defensively inside the averager.
2. **Local training.** `local_steps` of training on the private dataset
   produce LoRA gradients. Nothing about the raw data leaves the device.
3. **Differential privacy.** Before anything is sent, the total L2 norm of
   all gradient tensors is clipped to `dp_max_grad_norm`, then Gaussian
   noise with `sigma = max_grad_norm * noise_multiplier` is added. Clipping
   bounds how much any single training example can move the gradient;
   noise makes membership in the dataset statistically ambiguous.
4. **Gossip exchange.** Gradients go out over the P2P gossip protocol to
   registered peers.
5. **Average and apply.** Local + peer gradients are merged (FedAvg by
   default) and applied to the local adapter.

Step 3 deserves emphasis: DP is applied *before* broadcast, inside each
node. The coordinator never has to be trusted with clean gradients because
it never sees them.

## Watching privacy get spent

The part most federated demos skip: privacy is not free, and pretending it
is free is how projects end up shipping leaks. Each round's noise addition
is one query against the Gaussian mechanism, and queries compose. DistLLM
tracks cumulative spend with an RDP accountant (Rényi Differential Privacy,
Mironov 2017), which gives tighter bounds than naively adding epsilons.

The repo ships a runnable demo — two simulated nodes, zero networking,
everything synthetic:

```bash
python examples/federated_training_demo.py
```

Output (clipping bound 1.0, noise sigma 1.0, delta 1e-5):

```text
round grads recv  dp clips  eps spent
--------------------------------------
    1          2         2     5.3026
    2          2         4     7.8376
    3          2         6     9.8376
    4          2         8    11.7565
```

Read that last column as a fuel gauge. Four rounds cost ~11.8 epsilon under
these settings; if your budget is 8, you stop after round two — or you
raise sigma and accept a slower-learning model. That trade-off curve is the
honest shape of private training, and having it printed per round beats
discovering it in an audit.

## When devices are too different

Plain averaging assumes nodes see roughly the same data distribution.
Consumer reality is the opposite: one laptop holds work documents, another
holds a teenager's photo library. FedProx (Li et al., 2020) addresses this
by adding a proximal term `mu * (w_local - w_global)` to every gradient —
a leash tying each device's model to the shared global weights so no node
drags the consensus toward its own quirks. It's one constructor flag:
`algorithm="fedprox"`, plus a `fedprox_mu` coefficient and periodic
`set_global_model()` calls as rounds complete.

## What this doesn't do (yet)

Credibility requires saying the quiet parts:

- **No secure aggregation yet.** Peers currently see each other's noised
  individual gradients, not just sums. The noise provides the formal
  guarantee; cryptographic sum-hiding is future work.
- **Inference-time DP fails closed.** The DP inference wrapper refuses to
  generate until its logit-level mechanism is wired, rather than emit
  unprotected output while charging a budget. Training-side DP, described
  here, is fully wired.
- **Epsilon is a budgeting tool**, computed under Gaussian-mechanism
  assumptions. It bounds what the noise hides, not every conceivable side
  channel.

The demo runs on CPU in a couple of seconds. If you have torch installed,
that is the whole barrier to seeing federated DP training happen in front
of you: `python examples/federated_training_demo.py`.
