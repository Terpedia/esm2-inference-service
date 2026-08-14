# CUDA 12.1 is compatible with Cloud Run's current L4 driver (CUDA 12.2 capability).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    PORT=8080

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t6_8M_UR50D', revision='c731040fcd8d73dceaa04b0a8e6329b345b0f5df', allow_patterns=['config.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','*.safetensors'])"

ENV HF_HUB_OFFLINE=1

USER 65532:65532
EXPOSE 8080
CMD ["terpedia-esm2"]
