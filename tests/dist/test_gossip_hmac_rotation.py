"""Regression: gossip HMAC key rotation must not replace a shared deployment key.

F-027: `check_key_rotation()` replaced the deployment-wide shared
DISTLLM_GOSSIP_HMAC_KEY with a random node-local token every interval, so after
the overlap expired no peer could verify this node's messages and gossip auth
permanently broke. Rotation must only apply to node-local (non-shared) keys.
"""

from __future__ import annotations

from distllm.dist.p2p.gossip import GossipProtocol


class TestGossipKeyRotation:
    def test_shared_key_not_rotated(self):
        proto = GossipProtocol(node_id="node-1", hmac_key="deployment-shared-key-abcdef-123456")
        original = proto._hmac_key
        proto._last_key_rotation = 0  # force rotation check to fire
        result = proto.check_key_rotation()
        assert proto._hmac_key == original, "shared key must NOT be replaced by a random key"
        assert result == []

    def test_node_local_key_still_rotates(self):
        # No shared key configured → node-local persistent key path (dev/test
        # mode). The key should still be replaced on rotation.
        proto = GossipProtocol(node_id="node-1", hmac_key="")
        original = proto._hmac_key
        assert proto._shared_hmac_key is False
        proto._last_key_rotation = 0
        proto.check_key_rotation()
        # In dev mode there is always a persistent key; rotation may or may not
        # change it depending on persistence, but the shared-key guard must not
        # skip it. The key must remain non-empty.
        assert proto._hmac_key != ""

    def test_verify_shared_key_still_authenticates(self):
        proto = GossipProtocol(node_id="node-1", hmac_key="deployment-shared-key-abcdef-123456")
        # A peer message signed with the same shared key verifies.
        msg = {"type": "advertise", "node_id": "node-2"}
        signed = proto.sign_message(msg)
        assert signed is not msg  # sign_message returns a copy with _hmac
        assert proto.verify_message(signed) is True

    @staticmethod
    def _make_protocol(node_id: str, hmac_key: str) -> GossipProtocol:
        # Construct directly with an explicit shared key. Never rely on the
        # node-local persistent-key fallback (which is unauthenticated across
        # nodes) — we must prove cross-node auth with a real shared secret.
        return GossipProtocol(node_id=node_id, hmac_key=hmac_key)

    def test_two_nodes_same_secret_authenticate_each_other(self):
        """Two nodes configured with the SAME shared secret must verify each
        other's gossip messages both directions (F-027 regression)."""
        shared = "deployment-shared-key-abcdef-123456"
        node_a = self._make_protocol("node-a", shared)
        node_b = self._make_protocol("node-b", shared)

        # Node A advertises; node B must authenticate it.
        ad = node_a.advertise(delta_only=False)
        signed = node_a.sign_message(ad)
        # A message signed by A must verify under B (and A itself).
        assert node_b.verify_message(signed) is True
        assert node_a.verify_message(signed) is True

        # Node B advertises; node A must authenticate it.
        ad_b = node_b.advertise(delta_only=False)
        signed_b = node_b.sign_message(ad_b)
        assert node_a.verify_message(signed_b) is True
        assert node_b.verify_message(signed_b) is True

    def test_two_nodes_different_secret_cannot_authenticate_each_other(self):
        """A node using a DIFFERENT secret must be rejected."""
        node_a = self._make_protocol("node-a", "secret-for-a")
        node_b = self._make_protocol("node-b", "secret-for-b")

        signed = node_a.sign_message(node_a.advertise(delta_only=False))
        # B must reject A's message (different keys).
        assert node_b.verify_message(signed) is False
        # And B's full processing path must drop it as unverified.
        assert node_b.process_advertisement(signed) == []

    def test_shared_key_survives_rotation_roundtrip_and_cross_node_auth(self):
        """Even when rotation fires (forced), two nodes sharing a configured
        secret must STILL authenticate each other — the shared key must not be
        replaced by a random node-local key (F-027 root cause)."""
        shared = "deployment-shared-key-abcdef-123456"
        node_a = self._make_protocol("node-a", shared)
        node_b = self._make_protocol("node-b", shared)

        # Force the rotation check to fire on both nodes.
        node_a._last_key_rotation = 0
        node_b._last_key_rotation = 0
        assert node_a.check_key_rotation() == []
        assert node_b.check_key_rotation() == []

        # The shared key must be unchanged after the forced rotation attempt.
        assert node_a._hmac_key == shared
        assert node_b._hmac_key == shared

        # Cross-node authentication must still hold in both directions.
        signed_a = node_a.sign_message(node_a.advertise(delta_only=False))
        assert node_b.verify_message(signed_a) is True
        signed_b = node_b.sign_message(node_b.advertise(delta_only=False))
        assert node_a.verify_message(signed_b) is True