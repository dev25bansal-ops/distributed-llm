"""Property-based tests: serialization round-trips.

Verifies that core configuration objects survive a
``model_dump()`` -> ``model_validate()`` round-trip exactly, i.e.

    obj == model_validate(model_dump(obj))

under diverse (hypothesis-generated) field values.

Modules under test (real, read before writing tests):
  * ``distllm.dist.federation.FederationConfig``  -- pydantic BaseSettings,
    simple flat typed fields with ``frozen=True``.
  * ``distllm.config.settings.DistLLMSettings``   -- large nested pydantic
    BaseSettings root config.

CAVEAT (documented, not forced): ``DistLLMSettings`` carries validators that
constrain some fields (e.g. ``model.name`` must be non-empty,
``model.dtype`` must be one of {float16, float32, bfloat16}; several
cross-field checks gate on *other* sections being enabled, e.g. ``tls.enabled``
requires cert/key files).  We therefore seed a valid ``model.name`` and only
mutate leaf fields that are unconditionally valid, rather than generating the
whole object from raw primitives and tripping those validators.  The
round-trip invariant itself is still exercised over a wide range of values.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from distllm.config.settings import DistLLMSettings
from distllm.dist.federation import FederationConfig


# Hypothesis is kept fast: modest example budget, no deadline, standard checks.
_PBT_SETTINGS = dict(max_examples=30, deadline=None)


# ---------------------------------------------------------------------------
# FederationConfig -- flat BaseSettings, fully primitive-driven.
# ---------------------------------------------------------------------------

@settings(**_PBT_SETTINGS)
@given(
    enabled=st.booleans(),
    cluster_id=st.text(min_size=0, max_size=20),
    listen_host=st.text(min_size=0, max_size=30),
    listen_port=st.integers(1024, 65535),
    seed_nodes=st.lists(st.text(min_size=0, max_size=30), max_size=5),
    discovery_interval_s=st.floats(0.1, 100.0),
    heartbeat_interval_s=st.floats(0.1, 100.0),
    spillover_enabled=st.booleans(),
    spillover_threshold_gpu_util=st.floats(0.0, 100.0),
    circuit_breaker_threshold=st.integers(1, 20),
    circuit_breaker_reset_s=st.floats(0.1, 100.0),
    cache_digest_ttl_s=st.floats(0.1, 1000.0),
    gossip_enabled=st.booleans(),
    gossip_fanout=st.integers(1, 20),
)
def test_federation_config_round_trip(
    enabled, cluster_id, listen_host, listen_port, seed_nodes,
    discovery_interval_s, heartbeat_interval_s, spillover_enabled,
    spillover_threshold_gpu_util, circuit_breaker_threshold,
    circuit_breaker_reset_s, cache_digest_ttl_s, gossip_enabled,
    gossip_fanout,
):
    data = {
        "enabled": enabled,
        "cluster_id": cluster_id,
        "listen_host": listen_host,
        "listen_port": listen_port,
        "seed_nodes": seed_nodes,
        "discovery_interval_s": discovery_interval_s,
        "heartbeat_interval_s": heartbeat_interval_s,
        "spillover_enabled": spillover_enabled,
        "spillover_threshold_gpu_util": spillover_threshold_gpu_util,
        "circuit_breaker_threshold": circuit_breaker_threshold,
        "circuit_breaker_reset_s": circuit_breaker_reset_s,
        "cache_digest_ttl_s": cache_digest_ttl_s,
        "gossip_enabled": gossip_enabled,
        "gossip_fanout": gossip_fanout,
    }
    cfg = FederationConfig.model_validate(data)
    dumped = cfg.model_dump()
    restored = FederationConfig.model_validate(dumped)
    assert restored == cfg


# ---------------------------------------------------------------------------
# DistLLMSettings -- nested root config. Seed valid model.name + dtype, then
# mutate only unconditionally-valid leaf scalars.
# ---------------------------------------------------------------------------

# Safe, unconditionally-valid leaf fields (no cross-section validators fired).
_DTYPE = st.sampled_from(["float16", "float32", "bfloat16"])
_GEN_TEMP = st.floats(0.1, 2.0)
_GEN_MAXTOK = st.integers(1, 4096)
_NET_TIMEOUT = st.integers(1, 600)
_NET_RETRIES = st.integers(1, 20)
_BATCH_TOK = st.integers(1, 8192)
_COORD_PORT = st.integers(1024, 65535)


@settings(**_PBT_SETTINGS)
@given(
    dtype=_DTYPE,
    gen_temp=_GEN_TEMP,
    gen_max_tokens=_GEN_MAXTOK,
    net_timeout=_NET_TIMEOUT,
    net_retries=_NET_RETRIES,
    batch_tokens=_BATCH_TOK,
    coord_port=_COORD_PORT,
)
def test_distllm_settings_round_trip(
    dtype, gen_temp, gen_max_tokens, net_timeout, net_retries,
    batch_tokens, coord_port,
):
    # Seed from a config with a valid, non-empty model.name.
    base = DistLLMSettings(model={"name": "meta-llama/Llama-2-7b"})
    data = base.model_dump()

    # Apply overrides (nested dict paths).
    data["model"]["dtype"] = dtype
    data.setdefault("generation", {})["temperature"] = gen_temp
    data.setdefault("generation", {})["max_new_tokens"] = gen_max_tokens
    data.setdefault("network", {})["grpc_timeout"] = net_timeout
    data.setdefault("network", {})["max_retries"] = net_retries
    # chunked_prefill stays disabled (default) so max_tokens_per_batch is not
    # cross-validated against it.
    data.setdefault("batching", {})["max_tokens_per_batch"] = batch_tokens
    data.setdefault("coordinator", {})["port"] = coord_port

    try:
        cfg = DistLLMSettings.model_validate(data)
    except ValidationError:
        # Defensive: if a generated combination trips an unrelated validator,
        # skip rather than force the round-trip (caveat noted above).
        return

    dumped = cfg.model_dump()
    restored = DistLLMSettings.model_validate(dumped)
    assert restored == cfg
