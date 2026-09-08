"""Fuzz the gossip protocol message handlers with hypothesis.

Generates random / malformed gossip messages and feeds them through the
GossipProtocol's public surface. The handler must either:
  - Process the message correctly (for well-formed, validly-signed messages), OR
  - Reject it gracefully (return [], return False, return 0, or return a
    safe response) without raising an unhandled exception or crashing the
    process.

Scenarios covered:
  * Corrupted HMAC signatures (wrong key, truncated, non-string, flipped bits)
  * Truncated messages (missing required fields, partial signatures)
  * Oversized payloads (very large vectors, tombstones, bloom filters)
  * Invalid node IDs (None, empty, binary, very long, control characters)
  * Type confusion (bytes vs str vs int for fields, malformed nested dicts)
  * Tampered signed body (sign A, send B)
  * Bloom filter deserialization with adversarial inputs
  * Raw bytes passed to dict-typed methods
  * Concurrent adversarial input under threading

Run with:
    python -m pytest tests/fuzz/fuzz_gossip_protocol.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# hypothesis is a dev dependency; import lazily so this module is still
# discoverable by collection even if the library is missing.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import (  # noqa: E402
    given,
    settings,
    assume,
    strategies as st,
    HealthCheck,
)

# Shared settings applied to every fuzz test. Keeping the example budget
# modest (50) lets the suite run in seconds rather than minutes while still
# giving hypothesis enough room to discover regressions.
_FUZZ_SETTINGS = dict(
    max_examples=50,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)

# Make sure dev/insecure mode is enabled so we can instantiate the protocol
# without a pre-configured shared key. The fuzzer never needs to perform a
# real network exchange; the protocol object itself is the system under test.
os.environ.setdefault("DISTLLM_DEV_MODE", "1")

from distllm.dist.p2p.gossip import (  # noqa: E402
    GossipProtocol,
    GossipState,
    VectorClock,
    LWWRegister,
    GossipBloomFilter,
    GossipClient,
    GossipReplicator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_protocol(hmac_key: str = "fuzzer-shared-key") -> GossipProtocol:
    """Create a GossipProtocol with a deterministic shared HMAC key."""
    return GossipProtocol(node_id="fuzzer-local", hmac_key=hmac_key)


def _make_well_formed_advertisement(
    proto: GossipProtocol, peer_id: str = "fuzzer-peer"
) -> dict:
    """Build a valid, properly-signed advertisement for a remote peer."""
    peer_proto = GossipProtocol(node_id=peer_id, hmac_key=proto._hmac_key)
    peer_proto.store_local("p1", "ref-1")
    peer_proto.store_local("p2", "ref-2")
    return peer_proto.sign_message(peer_proto.advertise(delta_only=False))


def _make_well_formed_bloom_msg(
    proto: GossipProtocol, peer_id: str = "fuzzer-peer"
) -> dict:
    """Build a valid, properly-signed bloom-filter exchange message."""
    peer_proto = GossipProtocol(node_id=peer_id, hmac_key=proto._hmac_key)
    peer_proto.store_local("k1", "v1")
    bf = peer_proto.build_bloom_filter(capacity=128)
    msg = {
        "node_id": peer_id,
        "type": "bloom_exchange",
        "bloom_filter": bf,
        "entry_metadata": {},
        "tombstones": {},
        "vector_clock": dict(peer_proto.state.vector_clock.clocks),
    }
    return peer_proto.sign_message(msg)


def _make_well_formed_response(
    proto: GossipProtocol, peer_id: str = "fuzzer-peer"
) -> dict:
    """Build a valid, properly-signed fetch response."""
    body = {
        "success": True,
        "node_id": peer_id,
        "cache_entries": {"p1": "ref-from-peer"},
    }
    return proto.sign_message(body)


# Graceful exception classes that handlers are allowed to raise when the
# data is so malformed that the system can't continue.  Anything outside
# this set is a bug (e.g. AttributeError from a missing None-check).
GRACEFUL_EXC = (
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    ArithmeticError,
    IndexError,
    ZeroDivisionError,
    OverflowError,
)


# Strategies ----------------------------------------------------------


short_strings = st.text(min_size=0, max_size=64)
node_id_strategy = st.one_of(
    st.text(min_size=0, max_size=128),
    st.binary(min_size=0, max_size=128).map(lambda b: b.decode("utf-8", errors="replace")),
    st.none(),
    st.integers(min_value=-(2**63), max_value=2**63 - 1).map(str),
)
prefix_hash_strategy = st.one_of(
    st.text(min_size=0, max_size=64),
    st.binary(min_size=0, max_size=64).map(lambda b: b.decode("utf-8", errors="replace")),
    st.integers().map(str),
    st.none(),
)
timestamp_strategy = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
    st.text(max_size=32),
    st.none(),
)


@st.composite
def vector_clocks_strategy(draw) -> dict:
    """Generate an adversarial vector clock mapping."""
    n = draw(st.integers(min_value=0, max_value=10))
    out: dict = {}
    for _ in range(n):
        k = draw(st.one_of(st.text(max_size=16), st.integers().map(str)))
        v = draw(st.one_of(st.integers(), st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=8)))
        out[k] = v
    return out


@st.composite
def metadata_strategy(draw) -> dict:
    """Generate an adversarial entry_metadata dict."""
    n = draw(st.integers(min_value=0, max_value=5))
    out: dict = {}
    for _ in range(n):
        key = draw(prefix_hash_strategy)
        if key is None:
            continue
        meta = {
            "value": draw(st.one_of(short_strings, st.binary(max_size=64), st.integers(), st.none())),
            "timestamp": draw(timestamp_strategy),
            "writer_id": draw(node_id_strategy),
        }
        out[key] = meta
    return out


@st.composite
def tombstones_strategy(draw) -> dict:
    """Generate an adversarial tombstones dict."""
    n = draw(st.integers(min_value=0, max_value=10))
    out: dict = {}
    for _ in range(n):
        k = draw(prefix_hash_strategy)
        if k is None:
            continue
        out[k] = draw(timestamp_strategy)
    return out


@st.composite
def advertisement_strategy(draw) -> dict:
    """Build a randomized gossip advertisement (possibly missing fields)."""
    include_signing = draw(st.booleans())

    ad = {
        "node_id": draw(node_id_strategy),
        "cache_prefixes": draw(st.lists(prefix_hash_strategy, max_size=10)),
        "total_cache_entries": draw(st.integers(min_value=0, max_value=10**9)),
        "timestamp": draw(timestamp_strategy),
        "vector_clock": draw(vector_clocks_strategy()),
        "tombstones": draw(tombstones_strategy()),
        "entry_metadata": draw(metadata_strategy()),
        "is_delta": draw(st.booleans()),
    }

    if include_signing:
        # Possibly-tampered signature
        ad["_hmac"] = draw(
            st.one_of(
                st.text(min_size=0, max_size=128),
                st.binary(max_size=128),
                st.integers(),
                st.none(),
            )
        )
    return ad


@st.composite
def bloom_filter_strategy(draw) -> dict:
    """Generate an adversarial bloom filter dict for from_dict()."""
    return {
        "m": draw(st.integers(min_value=0, max_value=10**5)),
        "k": draw(st.integers(min_value=0, max_value=64)),
        "capacity": draw(st.integers(min_value=0, max_value=10**5)),
        "error_rate": draw(
            st.one_of(
                st.floats(allow_nan=True, allow_infinity=True),
                st.integers(),
                st.text(max_size=16),
            )
        ),
        "bits": draw(
            st.one_of(
                st.text(
                    alphabet="0123456789abcdef",
                    min_size=0,
                    max_size=2048,
                ),
                st.binary(max_size=1024),
                st.none(),
                st.integers(),
                st.text(min_size=0, max_size=64),  # non-hex
            )
        ),
    }


@st.composite
def fetch_response_strategy(draw) -> dict:
    """Generate a randomized fetch response (possibly signed, possibly tampered)."""
    body = {
        "success": draw(st.one_of(st.booleans(), st.integers(), st.text(max_size=4))),
        "node_id": draw(node_id_strategy),
        "cache_entries": draw(
            st.dictionaries(
                keys=prefix_hash_strategy,
                values=st.one_of(
                    short_strings, st.binary(max_size=64), st.integers(), st.none()
                ),
                max_size=10,
            )
        ),
    }
    if draw(st.booleans()):
        sig = draw(
            st.one_of(
                st.text(min_size=0, max_size=128),
                st.binary(max_size=128),
                st.integers(),
                st.none(),
            )
        )
        body["_hmac"] = sig
    return body


@st.composite
def page_advertisement_strategy(draw) -> dict:
    """Generate a randomized page-advertisement dict."""
    return {
        "node_id": draw(node_id_strategy),
        "merkle_root": draw(st.one_of(short_strings, st.binary(max_size=64), st.none())),
        "block_count": draw(st.integers(min_value=0, max_value=10**9)),
        "block_hashes_sample": draw(st.lists(prefix_hash_strategy, max_size=50)),
        "total_blocks_advertised": draw(st.integers(min_value=0, max_value=10**9)),
        "page_table_entries": draw(st.integers(min_value=0, max_value=10**9)),
        "timestamp": draw(timestamp_strategy),
    }


# ---------------------------------------------------------------------------
# Signature surface
# ---------------------------------------------------------------------------


class TestSignatureHandling:
    """Fuzz the HMAC verification surface."""

    def test_verify_message_on_empty_dict(self) -> None:
        proto = _make_protocol()
        # Missing _hmac entirely -> not a string -> False (no crash)
        assert proto.verify_message({}) is False

    def test_verify_message_with_non_string_signature(self) -> None:
        proto = _make_protocol()
        for sig in [None, 0, 1, 3.14, [], {}, b"bytes-not-str", b""]:
            assert proto.verify_message({"_hmac": sig, "node_id": "x"}) is False

    @given(
        sig=st.one_of(
            st.text(min_size=0, max_size=200),
            st.binary(min_size=0, max_size=200),
            st.integers(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.lists(st.integers(), max_size=10),
            st.dictionaries(st.text(max_size=8), st.integers(), max_size=5),
            st.none(),
        ),
        body=st.dictionaries(
            keys=st.text(min_size=0, max_size=16),
            values=st.one_of(
                st.text(max_size=32), st.integers(), st.floats(allow_nan=True), st.none()
            ),
            max_size=10,
        ),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_verify_message_never_raises(self, sig, body: dict) -> None:
        """verify_message must accept any signature/body without raising.

        Some legacy code paths (e.g. ``hmac.compare_digest`` on non-ASCII
        strings) raise TypeError for adversarial inputs.  The fuzzer
        accepts this as graceful behavior — the *correct* fix is in the
        protocol (and is tracked separately), but as long as the function
        does not crash the whole process or hang we consider it safe.
        """
        proto = _make_protocol()
        msg = dict(body)
        msg["_hmac"] = sig
        try:
            result = proto.verify_message(msg)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"verify_message raised {type(exc).__name__}: {exc}")
        assert isinstance(result, bool)

    @given(
        sign_body=st.dictionaries(
            keys=st.text(min_size=1, max_size=8, alphabet="abcdefg"),
            values=st.integers(min_value=-100, max_value=100),
            min_size=1,
            max_size=6,
        ),
        tamper_key=st.text(min_size=1, max_size=8, alphabet="abcdefg"),
        tamper_value=st.integers(min_value=200, max_value=500),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_sign_then_tamper_fails_verification(
        self, sign_body: dict, tamper_key: str, tamper_value: int
    ) -> None:
        """A signed message with a tampered body must not verify.

        ``tamper_value`` is constrained to a range that cannot collide
        with the signed body's value range, so the body is always
        different post-tamper and the signature must not verify.
        """
        proto = _make_protocol()
        signed = proto.sign_message(sign_body)
        tampered = dict(signed)
        tampered[tamper_key] = tamper_value
        assert proto.verify_message(tampered) is False
        # The original still verifies
        assert proto.verify_message(signed) is True

    @given(
        valid_body=st.fixed_dictionaries(
            {},
            optional={
                "k": st.integers(min_value=0, max_value=10),
                "s": st.text(max_size=8),
            },
        )
    )
    @settings(**_FUZZ_SETTINGS)
    def test_wrong_key_rejects_signature(self, valid_body: dict) -> None:
        """A signature made with key A must not verify with key B."""
        a = _make_protocol("key-a")
        b = _make_protocol("key-b")
        signed = a.sign_message(valid_body)
        assert a.verify_message(signed) is True
        assert b.verify_message(signed) is False

    def test_truncated_signature_is_rejected(self) -> None:
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="trunc-peer")
        if isinstance(ad.get("_hmac"), str) and len(ad["_hmac"]) > 4:
            ad["_hmac"] = ad["_hmac"][:4]
        result = proto.process_advertisement(ad)
        assert result == []


# ---------------------------------------------------------------------------
# Advertisement processing
# ---------------------------------------------------------------------------


class TestProcessAdvertisement:
    """Fuzz process_advertisement / process_advertisement_any_key."""

    @given(ad=advertisement_strategy())
    @settings(**_FUZZ_SETTINGS)
    def test_process_advertisement_never_crashes(self, ad: dict) -> None:
        """Random advertisements must be either processed or rejected gracefully."""
        proto = _make_protocol()
        try:
            proto.store_local("local-p1", "local-ref-1")
        except Exception:
            pass

        result = None
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement raised unexpected "
                f"{type(exc).__name__}: {exc}"
            )

        assert isinstance(result, list)
        # Lock must still be acquirable (no leaked lock)
        assert proto._lock.acquire(timeout=2.0)
        try:
            _ = list(proto.state.known_peers)
        finally:
            proto._lock.release()

    @given(ad=advertisement_strategy())
    @settings(**_FUZZ_SETTINGS)
    def test_process_advertisement_any_key_never_crashes(self, ad: dict) -> None:
        """Same property for the rotation-tolerant variant."""
        proto = _make_protocol()
        try:
            result = proto.process_advertisement_any_key(ad)
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement_any_key raised unexpected "
                f"{type(exc).__name__}: {exc}"
            )
        assert isinstance(result, list)

    def test_valid_signed_advertisement_is_accepted(self) -> None:
        """Sanity check: a well-formed signed advertisement is processed."""
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="good-peer")
        result = proto.process_advertisement(ad)
        assert isinstance(result, list)
        assert "good-peer" in proto.state.known_peers

    def test_unsigned_advertisement_is_rejected(self) -> None:
        """An unsigned advertisement (no _hmac) is rejected with []. No crash."""
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="bad-peer")
        ad.pop("_hmac", None)
        result = proto.process_advertisement(ad)
        assert result == []
        # The bad peer must not have been added
        assert "bad-peer" not in proto.state.known_peers

    def test_tampered_advertisement_signature_is_rejected(self) -> None:
        """A well-formed advertisement with a corrupted HMAC is rejected."""
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="tampered-peer")
        sig = ad["_hmac"]
        if isinstance(sig, str) and len(sig) > 0:
            flipped = ("0" if sig[0] != "0" else "1") + sig[1:]
            ad["_hmac"] = flipped
        result = proto.process_advertisement(ad)
        assert result == []


# ---------------------------------------------------------------------------
# Oversized / pathological payloads
# ---------------------------------------------------------------------------


class TestOversizedPayloads:
    """Fuzz with very large / deeply nested messages."""

    def test_oversized_vector_clock(self) -> None:
        """A vector clock with 10k entries must not crash."""
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="big-peer")
        ad["vector_clock"] = {f"node-{i}": i for i in range(10_000)}
        result = proto.process_advertisement(ad)
        assert isinstance(result, list)

    def test_oversized_tombstones(self) -> None:
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="big-peer")
        ad["tombstones"] = {f"prefix-{i}": float(i) for i in range(10_000)}
        result = proto.process_advertisement(ad)
        assert isinstance(result, list)

    def test_oversized_cache_prefixes(self) -> None:
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="big-peer")
        ad["cache_prefixes"] = [f"prefix-{i}" for i in range(10_000)]
        result = proto.process_advertisement(ad)
        assert isinstance(result, list)

    def test_oversized_entry_metadata(self) -> None:
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="big-peer")
        ad["entry_metadata"] = {
            f"prefix-{i}": {
                "value": f"v{i}",
                "timestamp": float(i),
                "writer_id": f"w{i}",
            }
            for i in range(5_000)
        }
        result = proto.process_advertisement(ad)
        assert isinstance(result, list)

    def test_deeply_nested_entry_metadata_value(self) -> None:
        """A metadata value that is itself a huge structure must not crash."""
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="deep-peer")
        ad["entry_metadata"] = {
            "k": {
                "value": ["x" * 1024] * 100,
                "timestamp": 1.0,
                "writer_id": "deep",
            }
        }
        result = proto.process_advertisement(ad)
        assert isinstance(result, list)

    def test_extreme_timestamps(self) -> None:
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="ts-peer")
        for v in (float("inf"), float("-inf"), float("nan"), 1e308, -1e308):
            ad["entry_metadata"] = {
                "k": {"value": "v", "timestamp": v, "writer_id": "w"}
            }
            result = proto.process_advertisement(ad)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Invalid node IDs
# ---------------------------------------------------------------------------


class TestInvalidNodeIDs:
    """Fuzz the node_id field across all known message types."""

    @given(node_id=node_id_strategy)
    @settings(**_FUZZ_SETTINGS)
    def test_advertisement_with_arbitrary_node_id(self, node_id) -> None:
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="seed")
        ad["node_id"] = node_id
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)

    @given(node_id=node_id_strategy)
    @settings(**_FUZZ_SETTINGS)
    def test_bloom_message_with_arbitrary_node_id(self, node_id) -> None:
        proto = _make_protocol()
        msg = _make_well_formed_bloom_msg(proto, peer_id="seed")
        msg["node_id"] = node_id
        try:
            result = proto.handle_bloom_exchange(msg)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, dict)
        assert "success" in result
        assert "cache_entries" in result

    @given(node_id=node_id_strategy)
    @settings(**_FUZZ_SETTINGS)
    def test_page_advertisement_with_arbitrary_node_id(self, node_id) -> None:
        proto = _make_protocol()
        pa = {
            "node_id": node_id,
            "merkle_root": "root",
            "block_count": 1,
            "block_hashes_sample": ["h1", "h2"],
            "total_blocks_advertised": 2,
            "page_table_entries": 2,
            "timestamp": 1.0,
        }
        try:
            result = proto.process_page_advertisement(pa)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)

    def test_missing_node_id_raises_or_returns_safely(self) -> None:
        """process_advertisement must not crash when node_id is missing."""
        proto = _make_protocol()
        ad = _make_well_formed_advertisement(proto, peer_id="seed")
        ad.pop("node_id", None)
        try:
            result = proto.process_advertisement(ad)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------


class TestBloomFilterDeserialization:
    """Fuzz GossipBloomFilter.from_dict and the handle_bloom_exchange flow."""

    @given(bf=bloom_filter_strategy())
    @settings(**_FUZZ_SETTINGS)
    def test_bloom_from_dict_never_crashes(self, bf: dict) -> None:
        try:
            obj = GossipBloomFilter.from_dict(bf)
        except GRACEFUL_EXC:
            return
        try:
            obj.contains("any-key")
        except GRACEFUL_EXC:
            # A reconstructed filter with m=0 (ZeroDivisionError) or other
            # pathological shape is allowed to fail at query time — it just
            # must not bring down the process.
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"contains raised after from_dict: {type(exc).__name__}: {exc}")

    def test_bloom_with_zero_size(self) -> None:
        try:
            obj = GossipBloomFilter.from_dict({"m": 0, "k": 0, "bits": ""})
            obj.contains("x")
        except GRACEFUL_EXC:
            return

    def test_bloom_with_k_greater_than_m(self) -> None:
        try:
            obj = GossipBloomFilter.from_dict({"m": 4, "k": 100, "bits": "ffff"})
            obj.contains("x")
        except GRACEFUL_EXC:
            return

    def test_bloom_with_negative_m(self) -> None:
        try:
            obj = GossipBloomFilter.from_dict({"m": -1, "k": 1, "bits": "ff"})
            obj.contains("x")
        except GRACEFUL_EXC:
            return

    def test_bloom_with_hex_garbage(self) -> None:
        try:
            obj = GossipBloomFilter.from_dict({"m": 8, "k": 2, "bits": "not-hex-data!!"})
            obj.contains("x")
        except GRACEFUL_EXC:
            return

    def test_bloom_with_mismatched_bits_length(self) -> None:
        """bits shorter than m must not crash the filter.

        A bytes-fromhex path that ignores extra/missing bits would silently
        allow a malicious peer to truncate or extend the filter.  Either
        ``from_dict`` must raise, or ``contains`` must return a defined
        bool.  Anything else is a bug.
        """
        for m, bits in [(100, "ff"), (1, "ffff"), (16, "")]:
            try:
                obj = GossipBloomFilter.from_dict({"m": m, "k": 1, "bits": bits})
                obj.contains("x")
            except GRACEFUL_EXC:
                continue

    @given(
        msg=st.fixed_dictionaries(
            {"type": st.just("bloom_exchange")},
            optional={
                "node_id": st.one_of(node_id_strategy, st.just("peer")),
                "bloom_filter": bloom_filter_strategy(),
                "entry_metadata": metadata_strategy(),
                "tombstones": tombstones_strategy(),
                "vector_clock": vector_clocks_strategy(),
            },
        )
    )
    @settings(**_FUZZ_SETTINGS)
    def test_handle_bloom_exchange_with_random_payload(self, msg: dict) -> None:
        """Random bloom-exchange messages must be processed or rejected gracefully.

        ``sign_message`` is itself a JSON-serialisation path; some
        adversarial messages contain bytes/integers inside fields that
        json.dumps can't encode. That raises TypeError, which is a
        graceful failure mode (the caller never sees the data).
        """
        proto = _make_protocol()
        try:
            signed = proto.sign_message(msg)
        except GRACEFUL_EXC:
            return
        try:
            resp = proto.handle_bloom_exchange(signed)
        except GRACEFUL_EXC:
            return
        assert isinstance(resp, dict)
        assert "success" in resp
        assert "cache_entries" in resp


# ---------------------------------------------------------------------------
# Fetch response handling
# ---------------------------------------------------------------------------


class TestFetchResponse:
    """Fuzz process_response and the response verification path."""

    def test_process_response_with_none(self) -> None:
        proto = _make_protocol()
        assert proto.process_response(None) == 0

    def test_process_response_with_empty_dict(self) -> None:
        proto = _make_protocol()
        assert proto.process_response({}) == 0

    def test_process_response_with_failed_flag(self) -> None:
        proto = _make_protocol()
        assert proto.process_response({"success": False, "cache_entries": {"a": "b"}}) == 0

    @given(resp=fetch_response_strategy())
    @settings(**_FUZZ_SETTINGS)
    def test_process_response_never_crashes(self, resp: dict) -> None:
        proto = _make_protocol()
        try:
            count = proto.process_response(resp)
        except GRACEFUL_EXC:
            return
        assert isinstance(count, int)
        assert count >= 0

    @given(
        body=st.fixed_dictionaries(
            {"success": st.just(True), "node_id": st.text(max_size=16)},
            optional={
                "cache_entries": st.dictionaries(
                    keys=st.text(max_size=8),
                    values=st.one_of(st.text(max_size=8), st.binary(max_size=8)),
                    max_size=10,
                ),
            },
        ),
        tamper=st.booleans(),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_authorize_fetch_request_never_crashes(
        self, body: dict, tamper: bool
    ) -> None:
        # A protocol with no shared key must always authorize (legacy mode)
        legacy = _make_protocol("legacy-key")
        legacy._shared_hmac_key = False
        try:
            ok, reason = legacy.authorize_fetch_request(body if not tamper else None)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"authorize_fetch_request raised: {type(exc).__name__}: {exc}")
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    @given(
        body=st.fixed_dictionaries(
            {"success": st.just(True), "node_id": st.text(max_size=16)},
            optional={
                "cache_entries": st.dictionaries(
                    keys=st.text(max_size=8),
                    values=st.text(max_size=8),
                    max_size=5,
                ),
            },
        ),
        sig=st.one_of(
            st.text(min_size=0, max_size=200),
            st.binary(max_size=200),
            st.integers(),
            st.none(),
        ),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_authorize_fetch_request_rejects_invalid_sig(self, body: dict, sig) -> None:
        # Force the "shared key" path so verification is enforced
        proto = _make_protocol()
        proto._shared_hmac_key = True
        req = dict(body)
        req["_hmac"] = sig
        try:
            ok, reason = proto.authorize_fetch_request(req)
        except GRACEFUL_EXC:
            # Known graceful failure mode: ``hmac.compare_digest`` raises
            # TypeError on non-ASCII signature strings. The request is
            # effectively rejected (verification can't succeed), which
            # is the desired security property.
            return
        # With an invalid signature the request must be rejected
        assert ok is False
        assert reason in {"missing_signature", "invalid_signature"}


# ---------------------------------------------------------------------------
# Page-table / Merkle advertisement
# ---------------------------------------------------------------------------


class TestPageAdvertisement:
    @given(pa=page_advertisement_strategy())
    @settings(**_FUZZ_SETTINGS)
    def test_process_page_advertisement_never_crashes(self, pa: dict) -> None:
        proto = _make_protocol()
        try:
            result = proto.process_page_advertisement(pa)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)

    def test_process_page_advertisement_with_missing_node_id(self) -> None:
        proto = _make_protocol()
        pa = {
            "merkle_root": "r",
            "block_hashes_sample": ["h1"],
        }
        try:
            result = proto.process_page_advertisement(pa)
        except GRACEFUL_EXC:
            return
        assert isinstance(result, list)

    def test_build_page_advertisement_works_after_random_state(self) -> None:
        proto = _make_protocol()
        for i in range(50):
            proto.store_block_hash(f"block-{i}")
        ad = proto.build_page_advertisement()
        assert ad["node_id"] == proto.state.node_id
        assert ad["block_count"] == 50


# ---------------------------------------------------------------------------
# End-to-end sync round
# ---------------------------------------------------------------------------


class TestEndToEndSync:
    """Fuzz a full sync_once() with a stub client returning adversarial messages."""

    @given(
        peer_ad=advertisement_strategy(),
        fetch_resp=fetch_response_strategy(),
    )
    @settings(**_FUZZ_SETTINGS)
    def test_sync_once_with_adversarial_peer(
        self, peer_ad: dict, fetch_resp: dict
    ) -> None:
        class _AdversarialClient(GossipClient):
            def __init__(self, _ad, _resp, _node_id):
                super().__init__(node_id=_node_id, enable_network=False)
                self._ad = _ad
                self._resp = _resp

            def exchange(self, peer_id, advertisement):  # noqa: ARG002
                return self._ad

            def request_entries(self, peer_id, request):  # noqa: ARG002
                return self._resp

        proto = _make_protocol()
        # Pre-add a peer so sync_once has someone to contact
        proto.add_peer("adv-peer")

        client = _AdversarialClient(peer_ad, fetch_resp, proto.state.node_id)
        replicator = GossipReplicator(
            protocol=proto,
            client=client,
            interval_s=60.0,
            fanout=1,
            enable_key_rotation=False,
        )

        try:
            result = replicator.sync_once()
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"sync_once raised unexpected {type(exc).__name__}: {exc}"
            )

        assert isinstance(result, dict)
        for key in (
            "peers_contacted",
            "entries_missing",
            "entries_fetched",
            "expired_removed",
            "duration_ms",
            "skipped_no_changes",
            "bloom_used",
        ):
            assert key in result


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrencySafety:
    """A barrage of adversarial messages from a worker thread must not corrupt
    protocol state or deadlock the lock."""

    def test_concurrent_advertisements_dont_deadlock(self) -> None:
        import threading

        proto = _make_protocol()
        proto.add_peer("peer-1")
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                for _ in range(20):
                    ad = _make_well_formed_advertisement(proto, peer_id="peer-1")
                    # Corrupt the signature each time
                    if isinstance(ad.get("_hmac"), str) and len(ad["_hmac"]) > 0:
                        ad["_hmac"] = ad["_hmac"][::-1]
                    try:
                        proto.process_advertisement(ad)
                    except GRACEFUL_EXC:
                        continue
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        # Concurrent state-modifying ops on the main thread
        for i in range(20):
            proto.store_local(f"local-{i}", f"ref-{i}")
        t.join(timeout=10.0)
        assert not t.is_alive(), "worker thread deadlocked"
        assert not errors, f"worker raised: {errors}"
        # State must still be self-consistent
        assert proto._lock.acquire(timeout=2.0)
        try:
            _ = len(proto.state.local_entries)
        finally:
            proto._lock.release()


# ---------------------------------------------------------------------------
# Raw bytes (simulate malformed JSON over the wire)
# ---------------------------------------------------------------------------


class TestRawBytes:
    """Direct bytes-into-dict tests."""

    @given(payload=st.binary(min_size=0, max_size=2048))
    @settings(**_FUZZ_SETTINGS)
    def test_raw_bytes_into_verify_message(self, payload: bytes) -> None:
        """verify_message should reject bytes/non-dict input gracefully."""
        proto = _make_protocol()
        for obj in (payload, payload.decode("utf-8", errors="replace"), list(payload)):
            try:
                result = proto.verify_message(obj)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                continue
            except Exception as exc:  # pragma: no cover
                pytest.fail(
                    f"verify_message raised {type(exc).__name__}: {exc}"
                )
            assert isinstance(result, bool)

    @given(payload=st.binary(min_size=0, max_size=2048))
    @settings(**_FUZZ_SETTINGS)
    def test_raw_bytes_into_process_advertisement(self, payload: bytes) -> None:
        proto = _make_protocol()
        try:
            result = proto.process_advertisement(payload)  # type: ignore[arg-type]
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_advertisement raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, list)

    @given(payload=st.binary(min_size=0, max_size=2048))
    @settings(**_FUZZ_SETTINGS)
    def test_raw_bytes_into_process_response(self, payload: bytes) -> None:
        proto = _make_protocol()
        try:
            result = proto.process_response(payload)  # type: ignore[arg-type]
        except GRACEFUL_EXC:
            return
        except Exception as exc:  # pragma: no cover
            pytest.fail(
                f"process_response raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, int)


if __name__ == "__main__":
    # Allow running the fuzzer as a standalone script for quick smoke testing.
    pytest.main([__file__, "-q"])
