# CloudModelTrain — Multi-Agent QLoRA Fine-Tuning Framework

Fine-tune specialized coding-agent LoRA adapters on a single A100 GPU. Each agent role (code_writer, test_generator, security_auditor, etc.) gets its own adapter, dataset, and config — all hyper-specialized on Python.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build real datasets from HuggingFace (one-time)
python scripts/build_datasets.py

# 3. Train an agent
python train.py --agent code_writer

# 4. Or use accelerate for bf16 mixed precision
bash scripts/launch.sh code_writer
```

## Datasets

All datasets are pulled from real, publicly available HuggingFace sources — no synthetic data.

| Agent | HuggingFace Source | ~Records | Content |
|-------|-------------------|----------|---------|
| `code_writer` | `iamtarun/python_code_instructions_18k_alpaca` | 18k | Python instruction → code pairs |
| `test_generator` | `KAKA22/CodeRM-UnitTest` / `mbpp` fallback | 17k+ | Code → unit test generation |
| `static_reviewer` | `HuggingFaceH4/Code-Feedback` (Python-filtered) | ~15k | Multi-turn code review chats |
| `security_auditor` | `CyberNative/Code_Vulnerability_Security_DPO` (Python) | ~1k+ | Vulnerable → secure code pairs |
| `performance_optimizer` | Code-Feedback (optimization) + MBPP | ~2k+ | Performance analysis + optimization |
| `docs_generator` | `Nan-Do/code-search-net-python` | 100k+ | Python function → docstring pairs |

### Building Datasets

```bash
# Build all agents
python scripts/build_datasets.py

# Build one agent
python scripts/build_datasets.py --agent code_writer

# Limit size (for testing)
python scripts/build_datasets.py --agent code_writer --max_samples 500

# List available builders
python scripts/build_datasets.py --list
```

Datasets are written to `datasets/<agent>/train.jsonl`.

## CLI Reference

```bash
# Basic training
python train.py --agent code_writer

# Named run
python train.py --agent test_generator --run_name v2_longer

# Resume from checkpoint
python train.py --agent security_auditor --resume outputs/security_auditor/20240101_120000/checkpoint-400

# Override model / hyperparams
python train.py --agent code_writer --model codellama/CodeLlama-13b-hf --lr 1e-4 --epochs 5

# Enable Flash Attention 2
python train.py --agent code_writer --flash_attn

# Validate dataset only (no training)
python train.py --agent code_writer --validate_only

# Skip dataset pull (data already cached)
python train.py --agent code_writer --skip_pull
```

## Supported Agents

| Agent | Description | Default Seq Len |
|-------|-------------|----------------|
| `code_writer` | Generates production Python code from specs | 4096 |
| `test_generator` | Creates pytest suites for given code | 2048 |
| `static_reviewer` | Reviews Python code for quality/bugs/style | 2048 |
| `security_auditor` | Identifies security vulnerabilities with CWE refs | 4096 |
| `performance_optimizer` | Optimizes Python code for speed/memory | 4096 |
| `docs_generator` | Generates docstrings and API documentation | 2048 |

## Project Structure

```
CloudModelTrain/
├── train.py                        # CLI entrypoint
├── configs/
│   ├── base.yaml                   # Base config (all agents inherit)
│   └── agents/*.yaml               # Per-agent overrides
├── registry/
│   └── agents.yaml                 # Agent -> dataset URI mapping
├── scripts/
│   ├── build_datasets.py           # Pull real data from HuggingFace
│   └── launch.sh                   # accelerate wrapper
├── src/
│   ├── config.py                   # Config loading + merging
│   ├── data/
│   │   ├── pull.py                 # S3/local dataset sync
│   │   ├── prepare.py              # Tokenization + packing
│   │   └── validate.py             # Integrity checks
│   └── train/
│       ├── run.py                  # QLoRA training loop
│       └── callbacks.py            # Early stopping, cost tracking
├── runpod/                         # RunPod deployment scripts
│   ├── setup.sh                    # One-time pod setup
│   ├── train_agent.sh              # Train one agent
│   ├── train_all.sh                # Train all agents
│   ├── Dockerfile                  # Custom RunPod template
│   ├── serverless_handler.py       # API-triggered training
│   └── README_RUNPOD.md            # RunPod deployment guide
├── datasets/                       # Built by build_datasets.py
│   └── <agent>/train.jsonl
└── requirements.txt
```

## Output Structure

```
outputs/<agent_name>/<run_name>/
├── adapter/                    # LoRA adapter weights (safetensors)
├── tokenizer/                  # Tokenizer files
├── config_snapshot.yaml        # Exact config used (reproducibility)
├── checkpoint-*/               # Intermediate checkpoints
└── runs/                       # TensorBoard logs
```

## Dataset Format

### Instruction format (JSONL) — code_writer, test_generator, security_auditor, performance_optimizer, docs_generator
```json
{"instruction": "Generate Python unit tests...", "input": "def add(a, b): ...", "output": "def test_add(): ..."}
```

### Chat format (JSONL) — static_reviewer
```json
{"messages": [{"role": "user", "content": "Review this Python code..."}, {"role": "assistant", "content": "Issues found:..."}]}
```

## Adding a New Agent (1-minute recipe)

1. Add a builder function in `scripts/build_datasets.py`

2. Add entry to `registry/agents.yaml`:
   ```yaml
   my_new_agent:
     description: "What it does"
     dataset_uri: "datasets/my_new_agent/train.jsonl"
     eval_uri: ""
     format: "instruction"
     fields:
       instruction: "instruction"
       input: "input"
       output: "output"
     sha256: ""
   ```

3. Optionally add `configs/agents/my_new_agent.yaml` with overrides:
   ```yaml
   training:
     max_seq_length: 4096
     learning_rate: 1.5e-4
   ```

4. Build + Train:
   ```bash
   python scripts/build_datasets.py --agent my_new_agent
   python train.py --agent my_new_agent
   ```

## Loading Adapters for Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("outputs/code_writer/run1/tokenizer")
model = PeftModel.from_pretrained(base_model, "outputs/code_writer/run1/adapter")
```

## RunPod Deployment

See [runpod/README_RUNPOD.md](runpod/README_RUNPOD.md) for full deployment guide.

```bash
# On RunPod A100 pod:
bash runpod/setup.sh
python scripts/build_datasets.py
bash runpod/train_all.sh --flash_attn
```
