# vLLM Inference Server Docker Configuration
# Uses the official vLLM OpenAI-compatible server

FROM vllm/vllm-openai:latest

# Environment variables for model configuration
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENV GPU_MEMORY_UTILIZATION=0.85
ENV MAX_MODEL_LEN=8192
ENV DTYPE=float16

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_NAME}", \
     "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--dtype", "${DTYPE}", \
     "--host", "0.0.0.0", \
     "--port", "8000"]

