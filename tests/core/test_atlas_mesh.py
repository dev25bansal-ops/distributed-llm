"""Regression tests for the AtlasMesh / LPSolverRouter request key.

Fixes audit issue B4: ``AtlasMesh._req_key`` produced ``model:prompt:max``
while ``LPSolverRouter._req_key`` always appended ``:idx``, so every
``scores.get(key, 0.5)`` in the solver missed and the multi-objective
score collapsed to the constant 0.5 default in every routing mode
(latency/cost/reliability/carbon scorer dead, bandit training reward
corrupted).  The two classes now share ONE canonical request key.

Regression strategy:
  * Build a scorer row (request -> composite score) with the same key format
    AtlasMesh uses, then compute the key both places and assert equality
    (fails pre-fix).
  * Route through the real mesh and assert the solver surfaces the real
    composite score rather than the 0.5 default (fails pre-fix).
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_am = load_module("distllm/core/atlas_mesh.py")
AtlasMesh = _am.AtlasMesh
LPSolverRouter = _am.LPSolverRouter
RoutingRequest = _am.RoutingRequest
ClusterGraph = _am.ClusterGraph
LatencyCostReliabilityScorer = _am.LatencyCostReliabilityScorer


def _make_request(**overrides: object) -> RoutingRequest:
    fields = dict(model="llama-70b", prompt_length=512, max_tokens=128)
    fields.update(overrides)
    return RoutingRequest(**fields)


def _two_cluster_mesh() -> AtlasMesh:
    """One clearly-good cluster (c1) and one clearly-bad cluster (c2)."""
    mesh = AtlasMesh()
    mesh.cluster_graph.add_cluster(
        "c1", region="us-east-1", provider="aws",
        cost_per_token=0.0, latency_baseline=10.0,
        reliability_history=1.0, carbon_intensity=0.0,
    )
    mesh.cluster_graph.add_cluster(
        "c2", region="us-east-2", provider="aws",
        cost_per_token=0.1, latency_baseline=5000.0,
        reliability_history=0.0, carbon_intensity=400.0,
    )
    return mesh


class TestCanonicalRequestKey:
    """The single canonical request key is shared by both classes."""

    def test_atlas_and_solver_keys_match(self):
        req = _make_request()
        assert AtlasMesh._req_key(req) == LPSolverRouter._req_key(req)

    def test_solver_key_ignores_batch_idx(self):
        """LPSolverRouter._req_key(req, i) must equal AtlasMesh._req_key(req)
        for every batch index — no stray ``:idx`` suffix."""
        req = _make_request()
        canonical = AtlasMesh._req_key(req)
        for idx in (0, 1, 7):
            assert LPSolverRouter._req_key(req, idx) == canonical

    def test_key_has_expected_shape(self):
        req = _make_request(model="llama-70b", prompt_length=512, max_tokens=128)
        assert AtlasMesh._req_key(req) == "llama-70b:512:128"
        assert LPSolverRouter._req_key(req) == "llama-70b:512:128"

    def test_scores_dict_lookup_hits(self):
        """A scorer row keyed by the AtlasMesh key must be found by the
        solver's lookups (greedy path uses default idx=0)."""
        req = _make_request()
        cluster = _two_cluster_mesh().cluster_graph.get_cluster("c1")
        assert cluster is not None
        scorer = LatencyCostReliabilityScorer()
        scores = {(AtlasMesh._req_key(req), cluster.id): scorer.score(cluster, req)}
        # Greedy uses self._req_key(req) with default idx → must hit.
        assert (LPSolverRouter._req_key(req), cluster.id) in scores
        assert scores[(LPSolverRouter._req_key(req), cluster.id)] == scorer.score(
            cluster, req
        )


class TestScoresHitRouting:
    """The composite score must flow through routing (not a flat 0.5)."""

    def test_greedy_assignment_reports_real_score(self):
        mesh = _two_cluster_mesh()
        req = _make_request()
        scores = {}
        for cluster in mesh.cluster_graph.all_clusters():
            key = (mesh._req_key(req), cluster.id)
            scores[key] = mesh.scorer.score(cluster, req)
        assignments = mesh.lp_router.solve(
            [req], mesh.cluster_graph.all_clusters(), scores
        )
        assert len(assignments) == 1
        assignment = assignments[0]
        assert assignment.cluster.id == "c1"
        assert assignment.score > 0.5
        assert assignment.score != 0.5

    def test_route_batch_reports_real_score(self):
        mesh = _two_cluster_mesh()
        assignments = mesh.route_batch([_make_request()])
        assert len(assignments) == 1
        assignment = assignments[0]
        assert assignment.cluster.id == "c1"
        assert assignment.score > 0.5
        assert assignment.score != 0.5

    def test_multi_request_scipy_batch_reports_real_scores(self):
        """Even with the scipy LP path (batch > 1), scores must be the real
        composite values, not the 0.5 default."""
        mesh = _two_cluster_mesh()
        reqs = [
            _make_request(model="llama-70b", prompt_length=512, max_tokens=128),
            _make_request(model="gpt-4", prompt_length=256, max_tokens=64),
        ]
        assignments = mesh.route_batch(reqs)
        assert len(assignments) == 2
        for assignment in assignments:
            assert assignment.cluster.id == "c1"
            assert assignment.score > 0.5
            assert assignment.score != 0.5
