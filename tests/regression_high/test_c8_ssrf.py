"""Regression tests for HIGH fix C8: SSRF allow_private_hosts=True.

``safe_urlopen`` already defaults to *blocking* private/loopback addresses
(fail-closed). The bug was the call sites in cross-cluster forwarding and P2P
discovery hardcoding ``allow_private_hosts=True``. Those now go through
``_federation_allows_private()``, which is False unless the operator explicitly
sets ``DISTLLM_FEDERATION_ALLOW_PRIVATE=1``.
"""

from __future__ import annotations

import os

import pytest

from distllm.security.utils import safe_urlopen, validate_http_url


def test_safe_urlopen_blocks_private_by_default():
    # 127.0.0.1 is loopback -> must raise with default (fail-closed) settings.
    with pytest.raises(ValueError):
        validate_http_url("http://127.0.0.1:8080/health", allow_private_hosts=False)
    with pytest.raises(ValueError):
        # safe_urlopen calls validate with allow_private_hosts defaulting False
        validate_http_url("http://10.0.0.5:8080/x", allow_private_hosts=False)


def test_federation_allows_private_opt_in():
    from distllm.dist import cross_cluster
    from distllm.dist.p2p import discovery

    os.environ.pop("DISTLLM_FEDERATION_ALLOW_PRIVATE", None)
    assert cross_cluster._federation_allows_private() is False
    assert discovery._federation_allows_private() is False

    os.environ["DISTLLM_FEDERATION_ALLOW_PRIVATE"] = "1"
    try:
        assert cross_cluster._federation_allows_private() is True
        assert discovery._federation_allows_private() is True
    finally:
        os.environ.pop("DISTLLM_FEDERATION_ALLOW_PRIVATE", None)


def test_cross_cluster_no_longer_passes_allow_private_true():
    # Static check that the insecure literal is gone from the call sites.
    src = (cross_cluster_path := __import__("pathlib").Path(__file__).resolve().parents[2])
    text = (src / "src/distllm/dist/cross_cluster.py").read_text()
    text2 = (src / "src/distllm/dist/p2p/discovery.py").read_text()
    assert "allow_private_hosts=True" not in text
    assert "allow_private_hosts=True" not in text2
    assert "_federation_allows_private()" in text
    assert "_federation_allows_private()" in text2
