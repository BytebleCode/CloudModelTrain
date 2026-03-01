# Running on RunPod

## Fastest Path: One Command

SSH into your RunPod pod (or use the web terminal), upload the project, and run:

```bash
cd /workspace/CloudModelTrain
bash runpod/go.sh
```

That's it. `go.sh` handles everything:
1. Installs Python deps + Flash Attention 2
2. Downloads real Python datasets from HuggingFace (18k–455k records per agent)
3. Verifies GPU
4. Trains all 6 agents sequentially

Second run onwards is instant (deps + datasets are cached).

### Train a specific agent
```bash
bash runpod/go.sh code_writer
bash runpod/go.sh security_auditor --flash_attn
bash runpod/go.sh code_writer --lr 1e-4 --epochs 5
```

### Train all agents
```bash
bash runpod/go.sh --all --flash_attn
```

---

## Step-by-Step (Manual)

### 1. Create a Pod
- Go to [RunPod](https://runpod.io) → Pods → Deploy
- GPU: **A100 80GB** (or A100 40GB for 7B models)
- Template: **RunPod PyTorch 2.2** (CUDA 12.1)
- Volume: 50GB+ (model weights + checkpoints)
- Deploy

### 2. Upload the project
```bash
# Option A: rsync from local machine
rsync -avz --exclude='outputs' --exclude='data_cache' --exclude='datasets' \
  . root@<pod_ip>:/workspace/CloudModelTrain/

# Option B: clone from git
ssh root@<pod_ip>
cd /workspace && git clone <your-repo-url> CloudModelTrain
```

### 3. Run
```bash
ssh root@<pod_ip>
cd /workspace/CloudModelTrain
bash runpod/go.sh code_writer --flash_attn
```

### 4. Download results
```bash
rsync -avz root@<pod_ip>:/workspace/CloudModelTrain/outputs/ ./outputs/
```

---

## Custom Docker Template (Zero-Setup Pods)

Pre-bake everything into a Docker image so pods need zero setup:

```bash
docker build -t <your-dockerhub>/cloudmodeltrain:latest -f runpod/Dockerfile .
docker push <your-dockerhub>/cloudmodeltrain:latest
```

On RunPod:
- Pods → Deploy → Change Template → Custom
- Image: `<your-dockerhub>/cloudmodeltrain:latest`
- Docker command: `bash runpod/go.sh code_writer --flash_attn`

Datasets are pre-built in the image — pod starts training immediately.

---

## Serverless Endpoint (API-Triggered)

For programmatic training without SSH:

### Deploy
1. Add `runpod` to requirements.txt
2. Build + push Docker image
3. RunPod → Serverless → New Endpoint
4. Handler path: `runpod/serverless_handler.py`
5. GPU: A100 80GB, Max workers: 1

### Trigger
```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint_id>/run" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"agent": "code_writer", "flash_attn": true}}'
```

### Check status
```bash
curl "https://api.runpod.ai/v2/<endpoint_id>/status/<job_id>" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"
```

---

## Recommended RunPod Settings

| Setting | Value | Notes |
|---------|-------|-------|
| GPU | A100 80GB SXM | Best price/perf for QLoRA |
| Volume | 50–100 GB | Model weights (~15GB) + checkpoints |
| Template | PyTorch 2.2+ | CUDA 12.1+ required |
| Container disk | 20 GB | pip packages |
| Spot instance | Yes | 60–70% cheaper; resume with `--resume` on preemption |

## Cost Estimates

| Agent | Seq Len | ~Time (7B, 1K samples) | ~Cost (A100 @ $1.64/hr spot) |
|-------|---------|------------------------|------------------------------------|
| code_writer | 4096 | ~20 min | ~$0.55 |
| test_generator | 2048 | ~12 min | ~$0.33 |
| static_reviewer | 2048 | ~15 min | ~$0.41 |
| security_auditor | 4096 | ~20 min | ~$0.55 |
| performance_optimizer | 4096 | ~20 min | ~$0.55 |
| docs_generator | 2048 | ~12 min | ~$0.33 |
| **All 6 agents** | — | **~1.5 hrs** | **~$2.70** |

*Scales linearly with dataset size. Costs based on RunPod spot A100 80GB.*

## Spot Instance Recovery

If your spot pod gets preempted:

```bash
# Re-deploy, then resume from last checkpoint:
bash runpod/go.sh code_writer --resume outputs/code_writer/<run>/checkpoint-400
```

Checkpoints are saved every 200 steps by default.
