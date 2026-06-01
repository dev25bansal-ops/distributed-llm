# 2-Laptop Distributed Cluster Setup Guide

## Network Info
- **This laptop (Coordinator)**: 192.168.1.59
- **Other laptop (Worker)**: ???? (fill in after checking)

---

## Step 1: On THIS Laptop (Coordinator)

### 1.1 Install DistLLM
```bash
cd D:\distributed-llm
pip install -e . --no-deps
```

### 1.2 Start the Coordinator
```bash
# Start coordinator with a small model for testing
distllm run --model roneneldan/TinyStories-1M --local --port 8000
```

### 1.3 Verify it's running
```bash
curl http://localhost:8000/health
```

---

## Step 2: On the OTHER Laptop (Worker)

### 2.1 Prerequisites
- Python 3.10+
- pip
- Same Wi-Fi network as coordinator

### 2.2 Install DistLLM
```bash
pip install distributed-llm
# OR clone and install from source:
git clone https://github.com/distributed-llm/distributed-llm.git
cd distributed-llm
pip install -e . --no-deps
```

### 2.3 Connect as Worker
```bash
# Replace 192.168.1.59 with coordinator's IP
distllm-node --coordinator 192.168.1.59:50050 --port 50051
```

### 2.4 Verify Connection
```bash
# On coordinator laptop, check connected nodes
curl http://localhost:8000/admin/v1/nodes
```

---

## Step 3: Test Distributed Inference

### 3.1 Send a request
```bash
curl http://192.168.1.59:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "roneneldan/TinyStories-1M",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### 3.2 Check cluster status
```bash
curl http://192.168.1.59:8000/health
```

---

## Troubleshooting

### "Connection refused"
- Check firewall: `netsh advfirewall firewall add rule name="DistLLM" dir=in action=allow protocol=tcp localport=50050,8000`
- Check both laptops are on same Wi-Fi network

### "Model not found"
- Model downloads on first use — wait for download to complete
- Check disk space: models need 1-10GB depending on size

### "CUDA not available"
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Check GPU: `nvidia-smi`
