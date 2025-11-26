# Qdrant Vector Database Docker Configuration
# This file is optional - we use the official image directly in docker-compose
# But can be used for custom configurations

FROM qdrant/qdrant:latest

# Expose the default port
EXPOSE 6333
EXPOSE 6334

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6333/health || exit 1

# Default storage path
VOLUME ["/qdrant/storage"]

