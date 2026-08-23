"""Regression tests: PBFT message authenticity.

The previous implementation "signed" messages with
``sha256(node_id:digest:view)`` — every input is public, so any node could
forge another node's signature, and ``verify_signature`` was never called in
any handler. This suite verifies the fixed behavior:

- messages carry real Ed25519 signatures bound to ``sender:view:sequence:digest``,
- handlers reject forged, unsigned, unknown-sender, and replayed messages,
- the three-phase consensus still commits with all keys registered.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric import ed25519

from distllm.dist.byzantine import PBFTMessage, PBFTNode, PBFTPhase


def _cluster(node_ids: list[str]) -> dict[str, PBFTNode]:
    """Build a PBFTNode per member, each holding every member's public key."""
    keys = {nid: ed25519.Ed25519PrivateKey.generate() for nid in node_ids}
    pub = {nid: k.public_key() for nid, k in keys.items()}
    nodes: dict[str, PBFTNode] = {}
    for nid in node_ids:
        nodes[nid] = PBFTNode(
            nid, list(node_ids),
            signing_key=keys[nid],
            node_public_keys=pub,
        )
    return nodes


def _pre_prepare(
    node: PBFTNode, view: int = 0, sequence: int = 1, operation: dict | None = None,
) -> PBFTMessage:
    op = operation if operation is not None else {"op": "test"}
    digest = node._compute_digest(op)
    return PBFTMessage(
        phase=PBFTPhase.PRE_PREPARE,
        view=view, sequence=sequence, digest=digest,
        sender=node.node_id,
        operation=op,
        signature=node._sign_message(digest, view=view, sequence=sequence),
    )


class TestSignatureEnforcement:
    def test_legit_pre_prepare_accepted(self) -> None:
        """A validly signed PRE-PREPARE from the primary is accepted."""
        nodes = _cluster(["a", "b", "c", "d"])
        a, b = nodes["a"], nodes["b"]
        assert b.primary == "a"

        msg = _pre_prepare(a, sequence=1)
        assert b.handle_pre_prepare(msg) is True

    def test_forged_pre_prepare_rejected(self) -> None:
        """A message claiming sender=a but signed by attacker b is rejected."""
        nodes = _cluster(["a", "b", "c", "d"])
        a, b = nodes["a"], nodes["b"]
        assert b.primary == "a"

        forged = PBFTMessage(
            phase=PBFTPhase.PRE_PREPARE, view=0, sequence=1,
            digest="tampered", sender="a", operation={"op": "evil"},
            signature=b._sign_message("tampered", view=0, sequence=1),
        )
        assert b.handle_pre_prepare(forged) is False

    def test_unsigned_pre_prepare_rejected(self) -> None:
        """Messages without a signature fail closed."""
        nodes = _cluster(["a", "b", "c", "d"])
        b = nodes["b"]
        unsigned = PBFTMessage(
            phase=PBFTPhase.PRE_PREPARE, view=0, sequence=1,
            digest="x", sender="a", operation={"op": "x"},
        )
        assert b.handle_pre_prepare(unsigned) is False

    def test_unknown_sender_verification_fails(self) -> None:
        """A message from a node whose public key is not registered is rejected."""
        nodes = _cluster(["a", "b", "c", "d"])
        b = nodes["b"]
        eve = PBFTMessage(
            phase=PBFTPhase.PREPARE, view=0, sequence=1, digest="x",
            sender="eve", signature="AAAA",
        )
        assert b.verify_signature(eve) is False

    def test_signature_bound_to_view_and_sequence(self) -> None:
        """Replaying a valid signature against a different sequence fails."""
        nodes = _cluster(["a", "b", "c", "d"])
        a, b = nodes["a"], nodes["b"]

        msg = _pre_prepare(a, sequence=1)
        assert b.verify_signature(msg) is True

        replay = PBFTMessage(
            phase=PBFTPhase.PRE_PREPARE, view=0, sequence=2,
            digest=msg.digest, sender="a", operation={"op": "x"},
        )
        replay.signature = msg.signature  # reuse the seq=1 signature
        assert b.verify_signature(replay) is False

    def test_tampered_prepare_rejected(self) -> None:
        """A PREPARE whose digest does not match the accepted pre-prepare is
        rejected even if the signature itself is valid."""
        nodes = _cluster(["a", "b", "c", "d"])
        a, b, c = nodes["a"], nodes["b"], nodes["c"]

        pp = a.handle_request({"op": "register", "model": "x"})
        assert b.handle_pre_prepare(pp) is True

        tampered = PBFTMessage(
            phase=PBFTPhase.PREPARE, view=0, sequence=pp.sequence,
            digest="evil", sender="c",
            signature=c._sign_message("evil", view=0, sequence=pp.sequence),
        )
        assert b.handle_prepare(tampered) is False


class TestViewChangeSignatures:
    def test_view_change_signature_enforced(self) -> None:
        nodes = _cluster(["a", "b", "c", "d"])
        b, c = nodes["b"], nodes["c"]

        c.start_view_change()
        vc_msg = c._view_change_messages[1][0]
        assert vc_msg["signature"]

        b.handle_view_change(vc_msg)
        assert 1 in b._view_change_messages

        # Forging the payload (last_committed) invalidates the signature.
        forged = dict(vc_msg)
        forged["last_committed"] = int(forged.get("last_committed", 0)) + 1
        b._view_change_messages = {}
        b.handle_view_change(forged)
        assert 1 not in b._view_change_messages

    def test_new_view_signature_enforced(self) -> None:
        nodes = _cluster(["a", "b", "c", "d"])
        b, c = nodes["b"], nodes["c"]

        # Seed a view-change so the new primary can build a signed new_view.
        c._view_change_messages[1] = [{
            "type": "view_change", "node_id": "c", "new_view": 1,
            "last_committed": 0,
            "signature": c._sign_view_message("view_change", "c", 1, 0),
        }]
        # Set c as the new primary for view 1 (node_ids sorted => index 1).
        # b's valid new_view requires a valid signature.
        nv = {
            "type": "new_view", "new_view": 1, "sender": "c",
            "signature": c._sign_view_message("new_view", "c", 1),
            "checkpoint": {},
        }
        b.handle_new_view(nv)
        assert b._view == 1

        forged = dict(nv)
        forged["signature"] = "AAAA"
        assert forged.get("signature") != nv["signature"]
        b2 = _cluster(["a", "b", "c", "d"])["b"]
        b2.handle_new_view(forged)
        assert b2._view == 0  # unchanged


class TestSignedThreePhaseCommit:
    def test_commit_round_trip_with_keys(self) -> None:
        """The full pre-prepare -> prepare -> commit flow still commits with
        real signatures on every message."""
        keys = {nid: ed25519.Ed25519PrivateKey.generate() for nid in "abcd"}
        pub = {nid: k.public_key() for nid, k in keys.items()}
        committed: list[dict] = []
        nodes: dict[str, PBFTNode] = {}
        for nid in "abcd":
            nodes[nid] = PBFTNode(
                nid, list("abcd"),
                callback=lambda op, c=committed: c.append(op),
                signing_key=keys[nid], node_public_keys=pub,
            )
        a, b, c, d = nodes["a"], nodes["b"], nodes["c"], nodes["d"]
        assert a.is_primary

        # Phase 1: primary broadcasts a signed PRE-PREPARE.
        pp = a.handle_request({"op": "register", "model": "x"})
        assert pp is not None and pp.signature
        seq, digest = pp.sequence, pp.digest
        for n in (b, c, d):
            assert n.handle_pre_prepare(pp) is True

        # Phase 2: peers exchange signed PREPAREs.
        def mk_prep(nid: str) -> PBFTMessage:
            return PBFTMessage(
                phase=PBFTPhase.PREPARE, view=0, sequence=seq, digest=digest,
                sender=nid,
                signature=nodes[nid]._sign_message(digest, view=0, sequence=seq),
            )

        b_prep, c_prep, d_prep = mk_prep("b"), mk_prep("c"), mk_prep("d")
        assert b.handle_prepare(c_prep) is True
        assert b.handle_prepare(d_prep) is True
        assert c.handle_prepare(b_prep) is True
        assert c.handle_prepare(d_prep) is True
        assert d.handle_prepare(b_prep) is True
        assert d.handle_prepare(c_prep) is True

        # Each node now holds a signed COMMIT in its own log.
        b_commit = b._commit_log[seq][0]
        c_commit = c._commit_log[seq][0]
        d_commit = d._commit_log[seq][0]

        # Phase 3: commit quorum.
        assert b.handle_commit(c_commit) is True
        assert b.handle_commit(d_commit) is True

        assert b._committed_sequences == {seq}
        assert committed == [{"op": "register", "model": "x"}]
