# Сервис инференса Qwen2-VL (FastAPI + PEFT LoRA)
# Сборка для CPU:        docker build -t docstruct-inference .
# Сборка для NVIDIA GPU: docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t docstruct-inference .
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN pip install --no-cache-dir \
        torch==2.9.1 \
        torchvision==0.24.1 \
        --extra-index-url ${TORCH_INDEX}

COPY pyproject.toml ./
COPY src/model/ ./src/model/
COPY src/configs/ ./src/configs/

RUN pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "python-multipart" \
        "Pillow>=10.0" \
        "PyYAML" \
        "transformers==5.3.0" \
        "accelerate>=0.33.0" \
        "peft==0.20.0" \
        "bitsandbytes>=0.43.0" \
        "qwen-vl-utils==0.0.14" \
        "sentencepiece" \
        "einops" \
    && pip install --no-cache-dir -e .

COPY adapters/ ./adapters/

ENV HF_HOME=/cache/huggingface \
    CONFIG_PATH=/app/src/configs/qwen2_2b.yaml \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]