"""Federated LoRA fine-tuning demo — two nodes, zero network.

Two in-process ``FederatedFineTuner`` nodes train on their own private
data and exchange *gradient updates only* through direct function calls
(no sockets, no coordinator).  Differential privacy is applied to every
gradient before it leaves a node, and an RDP accountant reports the
cumulative (epsilon, delta) privacy spend after each round.

What this demonstrates:
- FedAvg gradient exchange over pluggable gossip callbacks
- Per-node DP: total-norm gradient clipping + calibrated Gaussian noise
- Renyi DP accounting of the cumulative privacy budget across rounds
- Round-versioned gradient messages (peers tag contributions with the
  federated round they belong to)

Usage:
    python examples/federated_training_demo.py

Requires: pip install torch (no model download -- everything is synthetic)
"""

from __future__ import annotations

import queue
import threading

import torch

from distllm.core.dp_inference.accounting import RDPAccounting
from distllm.core.federated_finetuner import DPBudgetExhausted, FederatedFineTuner

# ── Privacy / training configuration ────────────────────────────────────────
NUM_ROUNDS = 4
LOCAL_STEPS = 10
LEARNING_RATE = 0.3
# Budget must cover the whole run: one sigma=1.0 Gaussian query costs
# ~5.3 epsilon at delta=1e-5, so four rounds cost ~21.2.  Set the budget
# lower (e.g. 8.0) to watch training STOP when privacy runs out -- the
# tuner refuses to broadcast gradients it can no longer account for.
DP_EPSILON = 25.0
DP_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0     # L2 clip bound -> bounds gradient sensitivity
DP_NOISE_MULTIPLIER = 1.0  # sigma = max_grad_norm * noise_multiplier = 1.0

# Shapes of the toy "LoRA adapter" every node trains locally.
PARAM_SHAPES = [(8, 8), (8,)]


class Node:
    """One simulated device: private weights, private data, a mailbox."""

    def __init__(self, node_id: str, private_target: float, inbox: queue.Queue):
        self.node_id = node_id
        self.inbox = inbox
        # Private training target: this node's data pulls its weights toward
        # ``private_target``.  Peers never learn this value directly -- they
        # only ever see clipped, noised gradients.
        self.private_target = private_target
        # Every node starts from different random weights (heterogeneous).
        generator = torch.Generator().manual_seed(hash(node_id) % (2**31))
        self.weights = [
            torch.randn(shape, generator=generator) * 0.1 for shape in PARAM_SHAPES
        ]

    def lora_adapter(self):
        """Current local adapter parameters (what local training reads)."""
        return self.weights

    def local_train(self, steps: int):
        """Pretend local training: gradient of ||w - target||^2 / 2."""
        return [w - self.private_target for w in self.weights]

    def apply_gradients(self, grads, lr: float):
        """Apply the merged (averaged) gradient to the local adapter."""
        with torch.no_grad():
            for w, g in zip(self.weights, grads):
                w -= lr * g


def build_tuner(node: Node, outboxes: dict[str, queue.Queue]) -> FederatedFineTuner:
    """Wire a FederatedFineTuner to in-process mailboxes (stands in for gossip)."""

    def broadcast(peer_id, message):
        # Real deployments send this over DistLLM's P2P gossip protocol;
        # here we drop it straight into the peer's queue.
        outboxes[peer_id].put({"peer_id": node.node_id, **message})

    def receive(timeout: float = 30.0):
        try:
            return node.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    return FederatedFineTuner(
        node_id=node.node_id,
        lora_adapter=node.lora_adapter,
        apply_gradients=node.apply_gradients,
        gossip_broadcast=broadcast,
        gossip_receive=receive,
        local_steps=LOCAL_STEPS,
        num_rounds=NUM_ROUNDS,
        learning_rate=LEARNING_RATE,
        dp_epsilon=DP_EPSILON,
        dp_delta=DP_DELTA,
        dp_max_grad_norm=DP_MAX_GRAD_NORM,
        dp_noise_multiplier=DP_NOISE_MULTIPLIER,
        algorithm="fedavg",
    )


def main():
    # One mailbox per node; each node holds its peer's outbox.
    inbox_a, inbox_b = queue.Queue(), queue.Queue()
    outboxes = {"laptop-a": inbox_b, "laptop-b": inbox_a}

    # Deliberately heterogeneous private datasets (different targets).
    node_a = Node("laptop-a", private_target=+1.0, inbox=inbox_a)
    node_b = Node("laptop-b", private_target=-1.0, inbox=inbox_b)

    tuner_a = build_tuner(node_a, outboxes)
    tuner_b = build_tuner(node_b, outboxes)
    tuner_a.add_peer("laptop-b")
    tuner_b.add_peer("laptop-a")

    # RDP accountant: composes the per-round Gaussian-mechanism queries
    # with Renyi divergence -- tighter than summing naive epsilon bounds.
    accountant = RDPAccounting()
    sigma = DP_MAX_GRAD_NORM * DP_NOISE_MULTIPLIER

    print(f"Federated LoRA demo - {tuner_a.stats['algorithm']} + DP "
          f"(clip={DP_MAX_GRAD_NORM}, noise sigma={sigma}, delta={DP_DELTA})")
    print(f"Nodes: laptop-a (private target {node_a.private_target:+.0f}), "
          f"laptop-b (private target {node_b.private_target:+.0f})")
    print()
    print(f"{'round':>5} {'grads recv':>10} {'dp clips':>9} {'eps spent':>10}")
    print("-" * 38)

    for round_idx in range(NUM_ROUNDS):
        # Run both nodes' rounds concurrently, exactly like real gossip:
        # each broadcasts its DP-noised gradients, then waits for its peer.
        results: dict[str, dict] = {}

        def run_round(tuner, node, node_id):
            try:
                results[node_id] = tuner.train_round(node.local_train)
            except DPBudgetExhausted as e:
                # The tuner fails closed when the (epsilon, delta) budget
                # is spent: no more noised broadcasts.
                results[node_id] = {"stopped": str(e)}

        threads = [
            threading.Thread(target=run_round, args=(tuner_a, node_a, "laptop-a")),
            threading.Thread(target=run_round, args=(tuner_b, node_b, "laptop-b")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if any("stopped" in m for m in results.values()):
            print(f"round {round_idx + 1}: privacy budget exhausted -- "
                  f"training stopped fail-closed.")
            break

        # Each round exposes one Gaussian-noise query per node; compose them.
        accountant.add_query(sigma)
        spent = accountant.get_privacy_spent(DP_DELTA)
        grads_recv = sum(m["gradients_received"] for m in results.values())
        clips = tuner_a.stats["dp_clips"] + tuner_b.stats["dp_clips"]

        print(f"{round_idx + 1:>5} {grads_recv:>10} {clips:>9} "
              f"{spent['epsilon']:>10.4f}")

    print("-" * 38)
    print("\nFinal stats")
    for name, tuner in (("laptop-a", tuner_a), ("laptop-b", tuner_b)):
        s = tuner.stats
        print(f"  {name}: rounds={s['rounds_completed']} "
              f"local_steps={s['total_local_steps']} "
              f"peers_contacted={s['peers_contacted']} "
              f"dp_clips={s['dp_clips']} noise_added={s['dp_noise_added']}")

    spent = accountant.get_privacy_spent(DP_DELTA)
    print(f"\nTotal privacy spend after {NUM_ROUNDS} rounds: "
          f"epsilon={spent['epsilon']}, delta={spent['delta']} "
          f"(RDP composed over {spent['orders_used']} orders)")
    print("Only clipped, Gaussian-noised gradients crossed the wire; "
          "raw data stayed on each node.")


if __name__ == "__main__":
    main()
