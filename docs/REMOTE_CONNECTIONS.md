# Remote & Mobile Connections

Connect DistLLM to cloud notebooks, other laptops, and mobile devices.

---

## Multi-Device Setup (All Devices Simultaneously)

All devices can connect to the **same coordinator** at the same time:

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR LAPTOP (Coordinator)                     │
│                    192.168.1.59 / 100.68.122.100                │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │   API Server :8000   │    │  gRPC Server :50050  │          │
│  │   (HTTP/REST)        │    │  (Worker connections) │          │
│  └──────────────────────┘    └──────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │Laptop 2│  │MacBook │  │Colab   │  │Kaggle  │  │ Phone  │
   │Worker  │  │Worker  │  │Client  │  │Client  │  │Client  │
   │gRPC    │  │gRPC    │  │HTTP    │  │HTTP    │  │HTTP    │
   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
```

### Device Roles

| Device | Role | Protocol | Purpose |
|--------|------|----------|---------|
| **Laptop 2** | Worker | gRPC :50050 | Adds GPU compute to the cluster |
| **MacBook** | Worker | gRPC :50050 | Adds GPU compute to the cluster |
| **Colab** | Client | HTTP :8000 | Sends inference requests |
| **Kaggle** | Client | HTTP :8000 | Sends inference requests |
| **Phone** | Client | HTTP :8000 | Sends inference requests |

### Step-by-Step: Connect All Devices

#### Step 1: Start Coordinator (this laptop)

```powershell
$env:API_KEY = "dev"
distllm system api -m Qwen/Qwen2.5-3B -l -p 8000 --host 0.0.0.0
```

#### Step 2: Expose for Colab/Kaggle/Phone (ngrok)

```powershell
# Install ngrok
winget install ngrok

# Expose port 8000
ngrok http 8000
# Note the URL: https://abc123.ngrok-free.app
```

#### Step 3: Connect Laptop 2 (same Wi-Fi)

```powershell
$env:API_KEY = "dev"
distllm cluster join --coordinator 192.168.1.59 --port 50050
```

#### Step 4: Connect MacBook (Tailscale)

```bash
# Install Tailscale on MacBook
# Login to same account as this laptop
export API_KEY="dev"
distllm cluster join --coordinator 100.68.122.100 --port 50050
```

#### Step 5: Use from Colab

```python
import openai
client = openai.OpenAI(
    base_url="https://abc123.ngrok-free.app/v1",
    api_key="dev",
)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B",
    messages=[{"role": "user", "content": "Hello from Colab!"}],
)
print(response.choices[0].message.content)
```

#### Step 6: Use from Kaggle

Same as Colab — paste the same code in a Kaggle notebook cell.

#### Step 7: Use from Phone

- **Option A**: Open `https://abc123.ngrok-free.app` in phone browser
- **Option B**: Install Tailscale on phone, use `http://100.68.122.100:8000`

### Important Notes

- **Workers** (laptops, MacBook) connect via **gRPC :50050** — they add GPU power
- **Clients** (Colab, Kaggle, phone) connect via **HTTP :8000** — they send requests
- All devices share the **same API key** (`API_KEY` env var)
- The coordinator handles all routing automatically
- More workers = faster inference (pipeline parallelism)
- More clients = more concurrent requests (load balancing)

---

## Google Colab / Kaggle

Colab and Kaggle run in Google's cloud — they can't directly reach your laptop.
Use **ngrok** to expose your API publicly.

### Step 1: Expose your API (on your laptop)

```powershell
# Install ngrok
winget install ngrok

# Authenticate (get token from https://dashboard.ngrok.com)
ngrok config add-authtoken YOUR_TOKEN

# Expose port 8000
ngrok http 8000
```

You'll get a public URL like `https://abc123.ngrok-free.app`

### Step 2: Use in Google Colab

```python
# In a Colab notebook cell:
import openai

client = openai.OpenAI(
    base_url="https://abc123.ngrok-free.app/v1",
    api_key="dev",  # Your API key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B",
    messages=[{"role": "user", "content": "Explain distributed computing"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### Step 3: Use in Kaggle

Same as Colab — paste the same code in a Kaggle notebook cell.

### Important Notes

- ngrok free tier has rate limits (20 connections/minute)
- The URL changes each time you restart ngrok (use `ngrok http --domain=your-domain 8000` for a fixed URL)
- Set `API_KEY` on your laptop for security

---

## MacBook

### Option A: Same Wi-Fi Network

```bash
# On MacBook
pip install distributed-llm

# Set the same API key as the coordinator
export API_KEY="dev"

# Connect as worker
distllm cluster join --coordinator 192.168.1.59 --port 50050

# Or just use the API
curl -H "Authorization: Bearer dev" http://192.168.1.59:8000/v1/models
```

### Option B: Tailscale (works from anywhere)

```bash
# 1. Install Tailscale on MacBook
#    https://tailscale.com/download/mac

# 2. Login to the same Tailscale account as your laptop

# 3. Your MacBook gets a Tailscale IP (e.g., 100.x.x.x)

# 4. Connect
export API_KEY="dev"
distllm cluster join --coordinator 100.68.122.100 --port 50050

# 5. Use the API
curl -H "Authorization: Bearer dev" http://100.68.122.100:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B","messages":[{"role":"user","content":"Hello!"}]}'
```

### Option C: Run DistLLM on MacBook (Apple Silicon)

```bash
# Install
pip install distributed-llm

# Start with llama.cpp backend (works on M1/M2/M3)
distllm system api --model Qwen/Qwen2.5-3B --local --port 8000 --no-auth

# Or connect as worker to your laptop's coordinator
distllm cluster join --coordinator 192.168.1.59 --port 50050
```

---

## Phone (iOS / Android)

### Option A: Tailscale + HTTP Client App

1. Install **Tailscale** on your phone (App Store / Play Store)
2. Login to the same account
3. Install an HTTP client app:
   - iOS: "HTTP Bot", "RestClient", or "Paw"
   - Android: "HTTP Request", "REST Client", or "Postman"
4. Make a request:

```
POST http://100.68.122.100:8000/v1/chat/completions
Authorization: Bearer dev
Content-Type: application/json

{
  "model": "Qwen/Qwen2.5-3B",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 100
}
```

### Option B: ngrok (public URL)

```powershell
# On your laptop
ngrok http 8000
```

Then open `https://abc123.ngrok-free.app` in your phone's browser.
The DistLLM web UI will load and you can chat directly.

### Option C: Build a Mobile App

The DistLLM API is OpenAI-compatible, so any mobile LLM app works:

| App | iOS | Android | Setup |
|-----|-----|---------|-------|
| **OpenCat** | ✅ | ✅ | Set base URL to your API |
| **ChatBox** | ✅ | ✅ | Add custom API endpoint |
| **Mochi Diffusion** | ✅ | ❌ | For image models |

---

## Quick Reference

| Device | Best Method | Speed | Setup Time |
|--------|-------------|-------|------------|
| Google Colab | ngrok | Medium | 5 min |
| Kaggle | ngrok | Medium | 5 min |
| MacBook (same WiFi) | Direct IP | Fast | 2 min |
| MacBook (remote) | Tailscale | Fast | 10 min |
| Phone (same WiFi) | Direct IP | Fast | 5 min |
| Phone (remote) | Tailscale | Fast | 10 min |
| Phone (public) | ngrok | Medium | 5 min |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Colab can't connect | Check ngrok is running, URL is correct |
| MacBook can't reach laptop | Check both on same Wi-Fi, or use Tailscale |
| Phone shows "connection refused" | Check firewall, ensure `--host 0.0.0.0` |
| Slow on phone | Use smaller model (0.5B-1.5B) |
| ngrok URL changed | Restart ngrok or use paid fixed domain |
