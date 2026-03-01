# Pre-built training image: all deps + flash-attn + datasets baked in.
# Pods start training immediately — zero setup time.
#
# Build:
#   docker build -t <your-dockerhub>/cloudmodeltrain:latest -f runpod/Dockerfile .
#   docker push <your-dockerhub>/cloudmodeltrain:latest
#
# On RunPod: use as custom template, then just run:
#   cd /workspace/CloudModelTrain && bash runpod/go.sh code_writer --flash_attn

FROM runpod/pytorch:2.4.1-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace/CloudModelTrain

# Redirect caches to /workspace (persists on RunPod volumes)
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
ENV PIP_CACHE_DIR=/workspace/.cache/pip
ENV TMPDIR=/workspace/tmp

# System deps
RUN apt-get update -qq && \
    apt-get install -y -qq git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Upgrade torch first (base for everything else)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade torch torchvision torchaudio

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade transformers accelerate peft bitsandbytes

# Flash Attention 2 (pre-compiled = no 15 min wait on pod startup)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy project
COPY . .

# Build datasets into image
RUN mkdir -p /workspace/tmp && \
    python scripts/build_datasets.py && \
    touch .setup_done_v6 .datasets_done

# Create dirs
RUN mkdir -p data_cache outputs /workspace/.cache/huggingface /workspace/.cache/pip /workspace/tmp

CMD ["bash", "-c", "echo 'Ready. Run: bash runpod/go.sh code_writer --flash_attn' && bash"]
