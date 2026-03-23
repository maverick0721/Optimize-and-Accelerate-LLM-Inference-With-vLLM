# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt ./

# Override at build time if needed, for example:
# --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url ${TORCH_INDEX_URL} torch torchvision torchaudio \
    && python -m pip install -r requirements.txt

COPY . .

RUN chmod +x run.sh

# Deterministic defaults from the project scripts.
CMD ["python", "run_benchmark.py", "--seed", "42", "--max-new-tokens", "100", "--output", "vllm_vs_hf_results.csv", "--quiet"]
