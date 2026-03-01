# Dockerfile for building a custom RunPod template with all deps + datasets pre-baked.
#
# Usage:
#   docker build -t my-training-image -f runpod/Dockerfile .
#   docker push <your-registry>/cloudmodeltrain:latest
#   # Then use as custom RunPod template — zero setup, just train.
#
# Based on RunPod's PyTorch image (CUDA 12.1 + Python 3.10).

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace/CloudModelTrain

# System deps
RUN apt-get update -qq && \
    apt-get install -y -qq git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Flash Attention 2 (cached layer — big speed boost on A100)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy project
COPY . .

# Build datasets into the image so pods start training immediately
RUN python scripts/build_datasets.py && \
    touch .setup_done .datasets_done

# Create output dirs
RUN mkdir -p data_cache outputs

# Default: show usage
CMD ["bash", "-c", "echo 'Usage: bash runpod/go.sh [agent_name] [--flash_attn]' && echo '' && python train.py --help"]
