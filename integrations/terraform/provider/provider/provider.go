package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

func Provider() *schema.Provider {
	return &schema.Provider{
		Schema: map[string]*schema.Schema{
			"endpoint": {
				Type:        schema.TypeString,
				Required:    true,
				DefaultFunc: schema.EnvDefaultFunc("DISTLLM_ENDPOINT", "http://localhost:8000"),
				Description: "DistLLM API endpoint",
			},
			"api_key": {
				Type:        schema.TypeString,
				Optional:    true,
				DefaultFunc: schema.EnvDefaultFunc("DISTLLM_API_KEY", ""),
				Description: "API key for DistLLM",
			},
			"timeout": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     120,
				Description: "HTTP request timeout in seconds",
			},
		},
		ResourcesMap: map[string]*schema.Resource{
			"distllm_model":      resourceModel(),
			"distllm_deployment": resourceDeployment(),
			"distllm_node":       resourceNode(),
			"distllm_federation": resourceFederation(),
		},
		DataSourcesMap: map[string]*schema.Resource{
			"distllm_cluster_status": dataSourceClusterStatus(),
			"distllm_models":         dataSourceModels(),
			"distllm_nodes":          dataSourceNodes(),
		},
		ConfigureContextFunc: providerConfigure,
	}
}

func providerConfigure(ctx context.Context, d *schema.ResourceData) (interface{}, diag.Diagnostics) {
	endpoint := d.Get("endpoint").(string)
	apiKey := d.Get("api_key").(string)
	timeout := d.Get("timeout").(int)

	client := &http.Client{
		Timeout: time.Duration(timeout) * time.Second,
	}

	config := &ProviderConfig{
		Endpoint: endpoint,
		APIKey:   apiKey,
		Client:   client,
	}

	return config, nil
}

type ProviderConfig struct {
	Endpoint string
	APIKey   string
	Client   *http.Client
}

// ---------------------------------------------------------------------------
// Resource: distllm_model
// ---------------------------------------------------------------------------

func resourceModel() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceModelCreate,
		ReadContext:   resourceModelRead,
		UpdateContext: resourceModelUpdate,
		DeleteContext: resourceModelDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},
		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Model name (e.g., meta-llama/Llama-2-7b)",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Model load status",
			},
			"loaded": {
				Type:        schema.TypeBool,
				Computed:    true,
				Description: "Whether the model is loaded",
			},
		},
	}
}

func resourceModelCreate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	name := d.Get("name").(string)

	payload := fmt.Sprintf(`{"model": %q}`, name)
	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/v1/models", config.Endpoint),
		strings.NewReader(payload))
	if err != nil {
		return diag.FromErr(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to load model %s: HTTP %d", name, resp.StatusCode)
	}

	d.SetId(name)
	d.Set("status", "loaded")
	d.Set("loaded", true)

	return nil
}

func resourceModelRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	name := d.Id()

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/v1/models/%s", config.Endpoint, name), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		d.SetId("")
		return nil
	}
	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to read model %s: HTTP %d", name, resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return diag.FromErr(err)
	}

	d.Set("name", result["id"])
	d.Set("status", "loaded")
	d.Set("loaded", true)
	return nil
}

func resourceModelUpdate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	return resourceModelCreate(ctx, d, m)
}

func resourceModelDelete(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	name := d.Id()

	req, err := http.NewRequestWithContext(ctx, "DELETE",
		fmt.Sprintf("%s/v1/models/%s", config.Endpoint, name), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	d.SetId("")
	return nil
}

// ---------------------------------------------------------------------------
// Resource: distllm_deployment
// ---------------------------------------------------------------------------

func resourceDeployment() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceDeploymentCreate,
		ReadContext:   resourceDeploymentRead,
		UpdateContext: resourceDeploymentUpdate,
		DeleteContext: resourceDeploymentDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},
		Schema: map[string]*schema.Schema{
			"model_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Model name to deploy",
			},
			"dtype": {
				Type:        schema.TypeString,
				Optional:    true,
				Default:     "float16",
				Description: "Model precision/quantization",
			},
			"num_nodes": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     1,
				Description: "Number of nodes for distributed deployment",
			},
			"replicas": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     1,
				Description: "Number of model replicas",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Deployment status",
			},
			"endpoint": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "API endpoint for the deployed model",
			},
			"loaded_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "When the model was loaded",
			},
		},
	}
}

func resourceDeploymentCreate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	modelName := d.Get("model_name").(string)
	dtype := d.Get("dtype").(string)

	payload := fmt.Sprintf(`{"model": %q, "dtype": %q}`, modelName, dtype)
	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/v1/models/load", config.Endpoint),
		strings.NewReader(payload))
	if err != nil {
		return diag.FromErr(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to deploy model %s: HTTP %d", modelName, resp.StatusCode)
	}

	d.SetId(modelName)
	d.Set("status", "deployed")
	d.Set("endpoint", fmt.Sprintf("%s/v1/completions", config.Endpoint))
	d.Set("loaded_at", time.Now().Format(time.RFC3339))

	return nil
}

func resourceDeploymentRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	modelName := d.Id()

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/v1/models/%s", config.Endpoint, modelName), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		d.SetId("")
		return nil
	}
	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to read deployment %s: HTTP %d", modelName, resp.StatusCode)
	}

	d.Set("status", "deployed")
	d.Set("endpoint", fmt.Sprintf("%s/v1/completions", config.Endpoint))

	return nil
}

func resourceDeploymentUpdate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	return resourceDeploymentCreate(ctx, d, m)
}

func resourceDeploymentDelete(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	modelName := d.Id()

	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/v1/models/unload", config.Endpoint),
		strings.NewReader(fmt.Sprintf(`{"model": %q}`, modelName)))
	if err != nil {
		return diag.FromErr(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	d.SetId("")
	return nil
}

// ---------------------------------------------------------------------------
// Resource: distllm_node
// ---------------------------------------------------------------------------

func resourceNode() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceNodeCreate,
		ReadContext:   resourceNodeRead,
		UpdateContext: resourceNodeUpdate,
		DeleteContext: resourceNodeDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},
		Schema: map[string]*schema.Schema{
			"host": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Node hostname or IP",
			},
			"port": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Node port",
			},
			"role": {
				Type:        schema.TypeString,
				Optional:    true,
				Default:     "worker",
				Description: "Node role (worker, coordinator, prefill, decode)",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Node status (online, offline, draining)",
			},
			"gpu_name": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "GPU type",
			},
			"gpu_memory_total": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Total GPU memory in MB",
			},
			"healthy": {
				Type:     schema.TypeBool,
				Computed: true,
			},
		},
	}
}

func resourceNodeCreate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	host := d.Get("host").(string)
	port := d.Get("port").(int)
	role := d.Get("role").(string)

	payload := fmt.Sprintf(`{"host": %q, "port": %d, "role": %q}`, host, port, role)
	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/admin/v1/nodes", config.Endpoint),
		strings.NewReader(payload))
	if err != nil {
		return diag.FromErr(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return diag.Errorf("failed to add node %s:%d: HTTP %d", host, port, resp.StatusCode)
	}

	d.SetId(fmt.Sprintf("%s:%d", host, port))
	d.Set("status", "online")
	d.Set("healthy", true)

	return nil
}

func resourceNodeRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	id := d.Id()
	parts := strings.SplitN(id, ":", 2)
	if len(parts) != 2 {
		d.SetId("")
		return nil
	}
	targetHost := parts[0]

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/api/cluster/nodes", config.Endpoint), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		d.SetId("")
		return nil
	}

	var nodes []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&nodes); err != nil {
		return diag.FromErr(err)
	}

	for _, node := range nodes {
		host, _ := node["host"].(string)
		if host == targetHost {
			d.Set("host", node["host"])
			d.Set("status", node["status"])
			d.Set("gpu_name", node["gpu_name"])
			d.Set("healthy", node["healthy"])
			if mem, ok := node["gpu_memory_total"].(float64); ok {
				d.Set("gpu_memory_total", int(mem))
			}
			return nil
		}
	}

	d.SetId("")
	return nil
}

func resourceNodeUpdate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	host := d.Get("host").(string)
	port := d.Get("port").(int)
	role := d.Get("role").(string)

	payload := fmt.Sprintf(`{"host": %q, "port": %d, "role": %q}`, host, port, role)
	req, err := http.NewRequestWithContext(ctx, "PUT",
		fmt.Sprintf("%s/admin/v1/nodes/%s:%d", config.Endpoint, host, port),
		strings.NewReader(payload))
	if err != nil {
		return diag.FromErr(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to update node %s:%d: HTTP %d", host, port, resp.StatusCode)
	}

	return nil
}

func resourceNodeDelete(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	host := d.Get("host").(string)
	port := d.Get("port").(int)

	req, err := http.NewRequestWithContext(ctx, "DELETE",
		fmt.Sprintf("%s/admin/v1/nodes/%s:%d", config.Endpoint, host, port), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	d.SetId("")
	return nil
}

// ---------------------------------------------------------------------------
// Data Source: distllm_cluster_status
// ---------------------------------------------------------------------------

func dataSourceClusterStatus() *schema.Resource {
	return &schema.Resource{
		ReadContext: dataSourceClusterStatusRead,
		Schema: map[string]*schema.Schema{
			"healthy": {
				Type:     schema.TypeBool,
				Computed: true,
			},
			"workers": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"models_loaded": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"version": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"num_gpus": {
				Type:     schema.TypeInt,
				Computed: true,
			},
		},
	}
}

func dataSourceClusterStatusRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/health", config.Endpoint), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	d.SetId(time.Now().Format(time.RFC3339))

	healthy := resp.StatusCode == http.StatusOK
	d.Set("healthy", healthy)

	if healthy {
		var result map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err == nil {
			if v, ok := result["version"]; ok {
				d.Set("version", v)
			}
			if w, ok := result["workers"]; ok {
				d.Set("workers", w)
			}
			if m, ok := result["models_loaded"]; ok {
				d.Set("models_loaded", m)
			}
			if m, ok := result["num_gpus"]; ok {
				d.Set("num_gpus", m)
			}
		} else {
			d.Set("workers", 1)
			d.Set("models_loaded", []string{})
			d.Set("version", "0.4.0")
			d.Set("num_gpus", 0)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Data Source: distllm_models
// ---------------------------------------------------------------------------

func dataSourceModels() *schema.Resource {
	return &schema.Resource{
		ReadContext: dataSourceModelsRead,
		Schema: map[string]*schema.Schema{
			"models": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"id":        {Type: schema.TypeString, Computed: true},
						"loaded":    {Type: schema.TypeBool, Computed: true},
						"memory_mb": {Type: schema.TypeInt, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceModelsRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/v1/models", config.Endpoint), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to list models: HTTP %d", resp.StatusCode)
	}

	d.SetId(time.Now().Format(time.RFC3339))

	var models []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return diag.FromErr(err)
	}

	var result []map[string]interface{}
	for _, m := range models {
		entry := map[string]interface{}{
			"id":     m["id"],
			"loaded": m["loaded"],
		}
		if mem, ok := m["memory_mb"]; ok {
			entry["memory_mb"] = mem
		}
		result = append(result, entry)
	}

	d.Set("models", result)
	return nil
}

// ---------------------------------------------------------------------------
// Data Source: distllm_nodes
// ---------------------------------------------------------------------------

func dataSourceNodes() *schema.Resource {
	return &schema.Resource{
		ReadContext: dataSourceNodesRead,
		Schema: map[string]*schema.Schema{
			"nodes": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"node_id":          {Type: schema.TypeString, Computed: true},
						"host":             {Type: schema.TypeString, Computed: true},
						"gpu_name":         {Type: schema.TypeString, Computed: true},
						"gpu_memory_total": {Type: schema.TypeInt, Computed: true},
						"gpu_memory_free":  {Type: schema.TypeInt, Computed: true},
						"healthy":          {Type: schema.TypeBool, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceNodesRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)

	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/api/cluster/nodes", config.Endpoint), nil)
	if err != nil {
		return diag.FromErr(err)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to list nodes: HTTP %d", resp.StatusCode)
	}

	d.SetId(time.Now().Format(time.RFC3339))

	var nodes []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&nodes); err != nil {
		return diag.FromErr(err)
	}

	var result []map[string]interface{}
	for _, n := range nodes {
		entry := map[string]interface{}{
			"node_id":  n["node_id"],
			"host":     n["host"],
			"gpu_name": n["gpu_name"],
			"healthy":  n["healthy"],
		}
		if mem, ok := n["gpu_memory_total"].(float64); ok {
			entry["gpu_memory_total"] = int(mem)
		}
		if mem, ok := n["gpu_memory_free"].(float64); ok {
			entry["gpu_memory_free"] = int(mem)
		}
		result = append(result, entry)
	}

	d.Set("nodes", result)
	return nil
}

// ---------------------------------------------------------------------------
// Resource: distllm_federation
// ---------------------------------------------------------------------------

func resourceFederation() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceFederationCreate,
		ReadContext:   resourceFederationRead,
		UpdateContext: resourceFederationUpdate,
		DeleteContext: resourceFederationDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},
		Schema: map[string]*schema.Schema{
			"cluster_id": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Unique cluster identifier",
			},
			"seed_nodes": {
				Type:        schema.TypeList,
				Optional:    true,
				Description: "Seed nodes for federation discovery",
				Elem:        &schema.Schema{Type: schema.TypeString},
			},
			"listen_port": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     50060,
				Description: "Federation listen port",
			},
			"spillover_enabled": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     true,
				Description: "Enable request spillover to peers",
			},
			"spillover_threshold": {
				Type:        schema.TypeFloat,
				Optional:    true,
				Default:     80.0,
				Description: "GPU utilization threshold for spillover",
			},
			"peers": {
				Type:        schema.TypeList,
				Computed:    true,
				Description: "Discovered peer clusters",
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"cluster_id": {Type: schema.TypeString, Computed: true},
						"host":       {Type: schema.TypeString, Computed: true},
						"port":       {Type: schema.TypeInt, Computed: true},
						"region":     {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func resourceFederationCreate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	clusterID := d.Get("cluster_id").(string)

	payload := map[string]interface{}{
		"cluster_id":         clusterID,
		"listen_port":        d.Get("listen_port").(int),
		"spillover_enabled":  d.Get("spillover_enabled").(bool),
		"spillover_threshold": d.Get("spillover_threshold").(float64),
	}

	if seeds, ok := d.Get("seed_nodes").([]interface{}); ok {
		seedList := make([]string, len(seeds))
		for i, s := range seeds {
			seedList[i] = s.(string)
		}
		payload["seed_nodes"] = seedList
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/api/federation/config", config.Endpoint),
		strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return diag.Errorf("failed to create federation config: HTTP %d", resp.StatusCode)
	}

	d.SetId(clusterID)
	return resourceFederationRead(ctx, d, m)
}

func resourceFederationRead(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)

	req, _ := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/api/federation/status", config.Endpoint), nil)
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		d.SetId("")
		return nil
	}

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to read federation status: HTTP %d", resp.StatusCode)
	}

	var status map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return diag.FromErr(err)
	}

	d.Set("cluster_id", status["cluster_id"])
	if peers, ok := status["peers"].([]interface{}); ok {
		peerList := make([]map[string]interface{}, len(peers))
		for i, p := range peers {
			peer := p.(map[string]interface{})
			peerList[i] = map[string]interface{}{
				"cluster_id": peer["cluster_id"],
				"host":       peer["host"],
				"port":       peer["port"],
				"region":     peer["region"],
			}
		}
		d.Set("peers", peerList)
	}

	return nil
}

func resourceFederationUpdate(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	config := m.(*ProviderConfig)
	clusterID := d.Get("cluster_id").(string)

	payload := map[string]interface{}{
		"cluster_id":         clusterID,
		"listen_port":        d.Get("listen_port").(int),
		"spillover_enabled":  d.Get("spillover_enabled").(bool),
		"spillover_threshold": d.Get("spillover_threshold").(float64),
	}

	if seeds, ok := d.Get("seed_nodes").([]interface{}); ok {
		seedList := make([]string, len(seeds))
		for i, s := range seeds {
			seedList[i] = s.(string)
		}
		payload["seed_nodes"] = seedList
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "PUT",
		fmt.Sprintf("%s/api/federation/config", config.Endpoint),
		strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	if config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+config.APIKey)
	}

	resp, err := config.Client.Do(req)
	if err != nil {
		return diag.FromErr(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return diag.Errorf("failed to update federation config: HTTP %d", resp.StatusCode)
	}

	return resourceFederationRead(ctx, d, m)
}

func resourceFederationDelete(ctx context.Context, d *schema.ResourceData, m interface{}) diag.Diagnostics {
	d.SetId("")
	return nil
}
