"""Test M-1, M-2, M-3 fixes."""
import sys, os, time, threading, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# M-1: ModelStore
print("=== M-1: ModelStore ===")
from distllm.dist.model_store import ModelStore
store = ModelStore(cache_dir='/tmp/distllm_test_cache')
p = store.model_path('test/model', 'main')
print(f"  Model path: {p}")
print(f"  Has layers (0-3): {store.has_layers('test/model', 0, 3)}")
print(f"  Cached models: {store.list_cached_models()}")
store.save_layer_manifest('test/model', 32)
cached = store.list_cached_models()
print(f"  After save manifest: {len(cached)} model(s)")
assert len(cached) >= 1, "Model should be cached"
assert cached[0]['total_layers'] == 32
print("  PASSED")

# M-2: Cluster key auth
print("\n=== M-2: Cluster key auth ===")
from distllm.dist.node_client import create_node_client
from distllm.dist.node_service import NodeServer
from distllm.dist import node_pb2
import torch

class MockAuthNode:
    def __init__(self):
        self.node_id = 'auth_test'
        self.start_layer = 0
        self.end_layer = 3
        self.total_layers = 8
        self.is_first = True
        self.is_last = True
        self.partitioner = type('obj', (object,), {'layers': list(range(4))})()
    def _get_device(self):
        return 'cpu'
    def forward_fn(self, **kw):
        return torch.randn(1, 5, 32000), None

mock = MockAuthNode()
server = NodeServer(mock, port=15081, cluster_key='test-secret')
t = threading.Thread(target=lambda: server.start(use_tls=False), daemon=True)
t.start()
time.sleep(1)

client_ok = create_node_client('127.0.0.1', 15081, timeout_s=3.0, cluster_key='test-secret')
r = client_ok.stub.HealthCheck(node_pb2.HealthCheckRequest(node_id='test'))
print(f"  Correct key health: {r.healthy} (expect True)")
assert r.healthy, "Auth with correct key should succeed"

client_bad = create_node_client('127.0.0.1', 15081, timeout_s=3.0, cluster_key='wrong')
r2 = client_bad.stub.HealthCheck(node_pb2.HealthCheckRequest(node_id='test'))
print(f"  Wrong key health: {r2.healthy} (expect False)")
assert not r2.healthy, "Auth with wrong key should fail"

client_none = create_node_client('127.0.0.1', 15081, timeout_s=3.0)
r3 = client_none.stub.HealthCheck(node_pb2.HealthCheckRequest(node_id='test'))
print(f"  No key health: {r3.healthy} (expect False)")
assert not r3.healthy, "Auth with no key should fail"

client_ok2 = create_node_client('127.0.0.1', 15081, timeout_s=3.0, cluster_key='test-secret')
r4 = client_ok2.stub.ForwardPass(node_pb2.ForwardPassRequest(request_id='test'))
print(f"  Correct key ForwardPass: success={r4.success} (expect True)")
assert r4.success, "ForwardPass with correct key should succeed"

client_bad2 = create_node_client('127.0.0.1', 15081, timeout_s=3.0, cluster_key='wrong')
r5 = client_bad2.stub.ForwardPass(node_pb2.ForwardPassRequest(request_id='test'))
print(f"  Wrong key ForwardPass: success={r5.success} (expect False)")
assert not r5.success, "ForwardPass with wrong key should fail"

server.stop()
print("  PASSED")

# M-3: Cluster nodes API endpoint
print("\n=== M-3: Cluster nodes API ===")
from distllm.api.server import app
routes_found = [r.path for r in app.routes if hasattr(r, 'path') and 'cluster/nodes' in r.path]
print(f"  /api/cluster/nodes endpoint exists: {len(routes_found) > 0}")
assert len(routes_found) > 0, "Cluster nodes endpoint should be registered"
print("  PASSED")

print("\n=== ALL THREE FIXES VERIFIED ===")
