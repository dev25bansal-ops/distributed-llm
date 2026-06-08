package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// mockServer returns an httptest.Server that mimics the DistLLM API.
func mockServer() *httptest.Server {
	mux := http.NewServeMux()

	// Health
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":        "ok",
			"version":       "0.4.0",
			"workers":       2,
			"models_loaded": []string{"meta-llama/Llama-2-7b"},
			"num_gpus":      2,
		})
	})

	// Models
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "meta-llama/Llama-2-7b", "loaded": true, "memory_mb": 13000},
				},
			})
		case http.MethodPost:
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
		}
	})

	// Single model
	mux.HandleFunc("/v1/models/meta-llama/Llama-2-7b", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "meta-llama/Llama-2-7b",
				"loaded": true,
			})
		case http.MethodDelete:
			w.WriteHeader(http.StatusOK)
		}
	})

	// Load/unload
	mux.HandleFunc("/v1/models/load", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
	})
	mux.HandleFunc("/v1/models/unload", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
	})

	// Nodes
	mux.HandleFunc("/api/cluster/nodes", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]interface{}{
			{
				"node_id":          "node-1",
				"host":             "10.0.1.10",
				"gpu_name":         "NVIDIA A100",
				"gpu_memory_total": 81920,
				"gpu_memory_free":  40960,
				"healthy":          true,
				"status":           "online",
			},
		})
	})

	// Admin nodes
	mux.HandleFunc("/admin/v1/nodes", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
	})
	mux.HandleFunc("/admin/v1/nodes/10.0.1.10:50051", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPut:
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
		case http.MethodDelete:
			w.WriteHeader(http.StatusOK)
		}
	})

	// Federation
	mux.HandleFunc("/api/federation/config", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok"})
	})
	mux.HandleFunc("/api/federation/status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"cluster_id": "test-cluster",
			"peers": []map[string]interface{}{
				{"cluster_id": "peer-1", "host": "10.0.2.10", "port": 50060, "region": "us-west-2"},
			},
		})
	})

	return httptest.NewServer(mux)
}

func testProviderConfig(server *httptest.Server) *ProviderConfig {
	return &ProviderConfig{
		Endpoint: server.URL,
		APIKey:   "",
		Client:   server.Client(),
	}
}

func testAccProviders(server *httptest.Server) map[string]*schema.Provider {
	p := Provider()
	p.ConfigureContextFunc = func(ctx context.Context, d *schema.ResourceData) (interface{}, diag.Diagnostics) {
		return testProviderConfig(server), nil
	}
	return map[string]*schema.Provider{
		"distllm": p,
	}
}

// ---------------------------------------------------------------------------
// Unit tests — resource CRUD
// ---------------------------------------------------------------------------

func TestResourceModel_Create(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceModel().Schema, map[string]interface{}{
		"name": "meta-llama/Llama-2-7b",
	})

	diags := resourceModelCreate(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("create failed: %v", diags)
	}
	if d.Id() != "meta-llama/Llama-2-7b" {
		t.Fatalf("expected id 'meta-llama/Llama-2-7b', got %q", d.Id())
	}
	if d.Get("status").(string) != "loaded" {
		t.Fatalf("expected status 'loaded', got %q", d.Get("status"))
	}
}

func TestResourceModel_Read(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceModel().Schema, map[string]interface{}{})
	d.SetId("meta-llama/Llama-2-7b")

	diags := resourceModelRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("read failed: %v", diags)
	}
	if d.Get("name") != "meta-llama/Llama-2-7b" {
		t.Fatalf("expected name 'meta-llama/Llama-2-7b', got %q", d.Get("name"))
	}
}

func TestResourceModel_Delete(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceModel().Schema, map[string]interface{}{})
	d.SetId("meta-llama/Llama-2-7b")

	diags := resourceModelDelete(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("delete failed: %v", diags)
	}
	if d.Id() != "" {
		t.Fatalf("expected empty id after delete, got %q", d.Id())
	}
}

func TestResourceDeployment_Create(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceDeployment().Schema, map[string]interface{}{
		"model_name": "meta-llama/Llama-2-7b",
		"dtype":      "float16",
	})

	diags := resourceDeploymentCreate(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("create failed: %v", diags)
	}
	if d.Get("status").(string) != "deployed" {
		t.Fatalf("expected status 'deployed', got %q", d.Get("status"))
	}
}

func TestResourceNode_Create(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceNode().Schema, map[string]interface{}{
		"host": "10.0.1.10",
		"port": 50051,
		"role": "worker",
	})

	diags := resourceNodeCreate(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("create failed: %v", diags)
	}
	expectedID := "10.0.1.10:50051"
	if d.Id() != expectedID {
		t.Fatalf("expected id %q, got %q", expectedID, d.Id())
	}
}

func TestResourceFederation_Create(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceFederation().Schema, map[string]interface{}{
		"cluster_id":          "test-cluster",
		"listen_port":         50060,
		"spillover_enabled":   true,
		"spillover_threshold": 80.0,
		"seed_nodes":          []interface{}{"10.0.1.10:50060"},
	})

	diags := resourceFederationCreate(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("create failed: %v", diags)
	}
	if d.Id() != "test-cluster" {
		t.Fatalf("expected id 'test-cluster', got %q", d.Id())
	}
}

// ---------------------------------------------------------------------------
// Data source tests
// ---------------------------------------------------------------------------

func TestDataSourceClusterStatus_Read(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, dataSourceClusterStatus().Schema, map[string]interface{}{})

	diags := dataSourceClusterStatusRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("read failed: %v", diags)
	}
	if !d.Get("healthy").(bool) {
		t.Fatal("expected healthy=true")
	}
	if d.Get("version").(string) != "0.4.0" {
		t.Fatalf("expected version '0.4.0', got %q", d.Get("version"))
	}
}

func TestDataSourceModels_Read(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, dataSourceModels().Schema, map[string]interface{}{})

	diags := dataSourceModelsRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("read failed: %v", diags)
	}
	models := d.Get("models").([]interface{})
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
}

func TestDataSourceNodes_Read(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, dataSourceNodes().Schema, map[string]interface{}{})

	diags := dataSourceNodesRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("read failed: %v", diags)
	}
	nodes := d.Get("nodes").([]interface{})
	if len(nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(nodes))
	}
	node := nodes[0].(map[string]interface{})
	if node["host"] != "10.0.1.10" {
		t.Fatalf("expected host '10.0.1.10', got %q", node["host"])
	}
}

// ---------------------------------------------------------------------------
// Import tests
// ---------------------------------------------------------------------------

func TestResourceModel_Import(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceModel().Schema, map[string]interface{}{})
	d.SetId("meta-llama/Llama-2-7b")

	diags := resourceModelRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("import read failed: %v", diags)
	}
	if d.Id() == "" {
		t.Fatal("expected non-empty id after import")
	}
}

func TestResourceDeployment_Import(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceDeployment().Schema, map[string]interface{}{})
	d.SetId("meta-llama/Llama-2-7b")

	diags := resourceDeploymentRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("import read failed: %v", diags)
	}
}

func TestResourceFederation_Import(t *testing.T) {
	server := mockServer()
	defer server.Close()

	config := testProviderConfig(server)
	d := schema.TestResourceDataRaw(t, resourceFederation().Schema, map[string]interface{}{})
	d.SetId("test-cluster")

	diags := resourceFederationRead(context.Background(), d, config)
	if diags.HasError() {
		t.Fatalf("import read failed: %v", diags)
	}
}
