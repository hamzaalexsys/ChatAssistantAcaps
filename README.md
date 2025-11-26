# Atlas-RAG: ACAPS Documentation Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot system designed for ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale) internal documentation. The system provides accurate, citation-backed answers from internal documents with a **Zero Hallucination** policy.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![License](https://img.shields.io/badge/License-Internal-red)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [Data Ingestion](#data-ingestion)
8. [Module Reference](#module-reference)
9. [API Reference](#api-reference)
10. [Customization Guide](#customization-guide)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

---

## Overview

Atlas-RAG is a RAG-based chatbot that:

- **Answers questions** based ONLY on ingested documents (Zero Hallucination)
- **Provides citations** with links to source documents
- **Supports French and English** queries
- **Includes guardrails** against jailbreaks, toxicity, and off-topic queries
- **Handles greetings** conversationally without database lookup

### Key Features

| Feature | Description |
|---------|-------------|
| **Multilingual Embeddings** | BAAI/bge-m3 model for French/English/Arabic support |
| **Vector Search** | Qdrant for fast similarity search |
| **LLM Generation** | Qwen2.5-7B via vLLM for response generation |
| **Guardrails** | Input/output validation to prevent misuse |
| **Smart Ingestion** | Only re-indexes changed/new files |
| **Citation Links** | Deterministic URL generation for sources |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                    http://localhost:3000                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend API (FastAPI)                       │
│                    http://localhost:8080                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Guardrails  │  │ RAG Engine  │  │ Conversational Router   │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│  Qdrant Vector  │ │   vLLM      │ │  BGE-M3         │
│    Database     │ │  Inference  │ │  Embeddings     │
│  :6333          │ │  :8000      │ │  (local)        │
└─────────────────┘ └─────────────┘ └─────────────────┘
```

### Data Flow

1. **User Query** → Frontend sends question to Backend API
2. **Input Guardrails** → Check for jailbreaks, toxicity, off-topic
3. **Greeting Detection** → If greeting, respond conversationally (skip RAG)
4. **Query Embedding** → BGE-M3 embeds the question
5. **Vector Search** → Qdrant finds similar document chunks
6. **Context Assembly** → Top-K results assembled as context
7. **LLM Generation** → Qwen generates answer from context
8. **Output Guardrails** → Check for hallucinations
9. **Response** → Answer + citations returned to user

---

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose** (for containerized deployment)
- **Python 3.12+** (for local development)
- **Node.js 18+** (for frontend development)
- **NVIDIA GPU** with CUDA (for vLLM inference server)

### One-Command Start (Docker)

```bash
# Clone and start all services
docker-compose up -d

# Ingest documents
docker exec atlas-api python -m data_ingestion.pipeline --recreate

# Access the UI
open http://localhost:3000
```

---

## Installation

### Option 1: Docker Compose (Recommended)

```bash
# Copy environment file
cp env.example .env

# Edit .env with your configuration
nano .env

# Start all services
docker-compose up -d
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Start vLLM (requires GPU)
docker run -d --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct --api-key secret

# Start Backend API
python -m uvicorn backend_api.app.main:app --host 0.0.0.0 --port 8080

# Start Frontend (separate terminal)
cd frontend_ui
npm install
npm run dev
```

---

## Configuration

### Environment Variables

Create a `.env` file from `env.example`:

```bash
# vLLM Inference Server
VLLM_URL=http://localhost:8000/v1
VLLM_API_KEY=secret
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=atlas_knowledge

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# RAG Configuration
SIMILARITY_THRESHOLD=0.40    # BGE-M3 scores are typically 0.4-0.7
TOP_K_RESULTS=5
MAX_CONTEXT_LENGTH=8192

# Generation Settings
TEMPERATURE=0.0              # 0 = deterministic, no creativity

# Security
ENABLE_GUARDRAILS=true
LOG_LEVEL=INFO
DEBUG=false
```

### Key Configuration Notes

| Parameter | Default | Notes |
|-----------|---------|-------|
| `SIMILARITY_THRESHOLD` | 0.40 | BGE-M3 produces scores in 0.4-0.7 range. Don't set above 0.5 |
| `TEMPERATURE` | 0.0 | Keep at 0 for factual, deterministic answers |
| `TOP_K_RESULTS` | 5 | Number of document chunks to retrieve |
| `ENABLE_GUARDRAILS` | true | Disable only for testing |

---

## Running the Application

### Start Services

```bash
# Start all services (Docker)
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f api
```

### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend UI | http://localhost:3000 | Chat interface |
| Backend API | http://localhost:8080 | REST API |
| API Docs | http://localhost:8080/docs | Swagger UI |
| Qdrant UI | http://localhost:6333/dashboard | Vector DB dashboard |
| vLLM | http://localhost:8000 | LLM inference |

---

## LLM Options

Atlas-RAG supports multiple LLM backends:

### Option 1: Ollama (Recommended for CPU)

Ollama runs LLMs locally without requiring a GPU (though slower).

```bash
# Install Ollama (https://ollama.ai)
# Windows: Download installer from website

# Pull the model
ollama pull qwen2.5:7b

# Verify it's running
curl http://localhost:11434/api/tags
```

**Configuration:**
```bash
VLLM_URL=http://localhost:11434/v1
VLLM_API_KEY=ollama
VLLM_MODEL=qwen2.5:7b
```

### Option 2: vLLM (Requires NVIDIA GPU)

```bash
# Start vLLM container (requires NVIDIA GPU + Docker)
docker run -d --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct --api-key secret
```

### Option 3: OpenAI API

```bash
VLLM_URL=https://api.openai.com/v1
VLLM_API_KEY=sk-your-api-key
VLLM_MODEL=gpt-4o-mini
```

### Fallback Mode

If no LLM is available, the system returns retrieved document context with a disclaimer.

---

## Data Ingestion

### Document Format

Documents must be **Markdown files** with YAML frontmatter:

```markdown
---
title: Document Title
base_url: https://portal.acaps.ma/docs/example
---

# Main Heading

Introduction paragraph.

## Section 1

Content of section 1.

### Subsection 1.1

Detailed content...
```

### Ingestion Commands

```bash
# Ingest all documents (skip unchanged files)
python -m data_ingestion.pipeline

# Force re-ingest all documents
python -m data_ingestion.pipeline --recreate

# Ingest a single file
python -m data_ingestion.pipeline --file data_ingestion/documents/new_doc.md

# Ingest from custom directory
python -m data_ingestion.pipeline --source-dir /path/to/docs

# Check pipeline status
python -m data_ingestion.pipeline --status

# Verbose output
python -m data_ingestion.pipeline -v
```

### Ingestion Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Markdown   │────▶│    Parser    │────▶│   Embedder   │────▶│    Qdrant    │
│    Files     │     │  (Chunking)  │     │   (BGE-M3)   │     │   (Upsert)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Check if    │
                    │ file changed │
                    │  (mtime)     │
                    └──────────────┘
```

### Smart Ingestion Features

- **Skip unchanged files**: Compares file modification time with stored timestamp
- **Detect updates**: Re-indexes files that have been modified
- **Header inclusion**: Embeds `header_path + text` for better retrieval
- **Incremental updates**: Use `process_single_file()` for single file updates

---

## File Watcher Service

The File Watcher automatically re-indexes documents when they are created, modified, or deleted.

### Start the File Watcher

```bash
# Watch the default documents directory
python -m data_ingestion.file_watcher

# Watch a custom directory
python -m data_ingestion.file_watcher --watch-dir /path/to/docs

# With custom debounce time (seconds)
python -m data_ingestion.file_watcher --debounce 5.0

# Verbose output
python -m data_ingestion.file_watcher -v
```

### How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    File Watcher Service                       │
│                                                               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  Watchdog   │────▶│  Debounce   │────▶│  Pipeline   │    │
│  │  Observer   │     │  (2 sec)    │     │  Ingestion  │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│         │                                       │            │
│         │ File Events                           │ Upsert     │
│         ▼                                       ▼            │
│  ┌─────────────┐                        ┌─────────────┐     │
│  │  Documents  │                        │   Qdrant    │     │
│  │  Directory  │                        │   Vector DB │     │
│  └─────────────┘                        └─────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Events Handled

| Event | Action |
|-------|--------|
| File Created | Parse + Embed + Upsert new document |
| File Modified | Delete old vectors + Re-index |
| File Deleted | Remove vectors from Qdrant |
| File Moved | Delete old + Index at new location |

### Run as Background Service

```bash
# Windows (PowerShell)
Start-Process -NoNewWindow python -ArgumentList "-m", "data_ingestion.file_watcher"

# Linux/Mac
nohup python -m data_ingestion.file_watcher > watcher.log 2>&1 &
```

---

## Module Reference

### 1. Data Ingestion (`data_ingestion/`)

| File | Purpose |
|------|---------|
| `parser.py` | Parses Markdown with YAML frontmatter, splits by headers |
| `embedder.py` | Generates embeddings using BGE-M3 model |
| `vector_store.py` | Qdrant operations: upsert, search, delete |
| `pipeline.py` | Orchestrates the full ingestion flow |
| `config.py` | Ingestion configuration |

### 2. Backend API (`backend_api/app/`)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application, endpoints |
| `engine.py` | RAG engine: retrieval + generation |
| `models.py` | Pydantic request/response models |
| `config.py` | API configuration (pydantic-settings) |
| `guardrails/` | Input/output validation |

### 3. Frontend UI (`frontend_ui/`)

| File | Purpose |
|------|---------|
| `src/App.tsx` | Main React application |
| `src/index.css` | Tailwind CSS styles |
| `vite.config.ts` | Vite build configuration |

---

## API Reference

### POST `/chat`

Send a question and receive an answer with citations.

**Request:**
```json
{
  "question": "Quelles sont les conditions pour l'autorisation d'exercice?",
  "conversation_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "Selon l'Article 5, les conditions sont...",
  "citations": [
    {
      "title": "Article 5 - Autorisation d'Exercice",
      "url": "https://portal.acaps.ma/legal/loi-assurance#article-5",
      "score": 0.66,
      "snippet": "Toute entreprise souhaitant exercer..."
    }
  ],
  "confidence": 0.66,
  "conversation_id": "uuid",
  "metadata": {
    "retrieval_count": 5,
    "model": "Qwen/Qwen2.5-7B-Instruct"
  }
}
```

### GET `/health`

Check system health.

### POST `/ingest`

Trigger document ingestion (admin endpoint).

### GET `/stats`

Get vector store statistics.

---

## Customization Guide

### Changing the Embedding Model

Edit `backend_api/app/config.py` and `data_ingestion/config.py`:

```python
# For a different model
embedding_model: str = "intfloat/multilingual-e5-large"
embedding_dimension: int = 1024  # Check model's dimension
```

**Note:** After changing the model, you MUST re-ingest all documents:
```bash
python -m data_ingestion.pipeline --recreate
```

### Changing the LLM

Edit `docker-compose.yml` or `.env`:

```yaml
# In docker-compose.yml
command: >
  --model mistralai/Mistral-7B-Instruct-v0.2
  --api-key ${VLLM_API_KEY:-secret}
```

### Adding New Document Types

Edit `data_ingestion/parser.py` to support new formats:

```python
class PDFParser:
    """Parser for PDF documents."""
    
    def parse_file(self, file_path: str) -> List[DocumentChunk]:
        # Implement PDF parsing
        pass
```

### Customizing Guardrails

Edit `backend_api/app/guardrails/guardrails.py`:

```python
# Add new jailbreak patterns
JAILBREAK_PATTERNS = [
    r"ignore.*instructions",
    r"your_new_pattern_here",
]

# Add domain-specific off-topic patterns
OFF_TOPIC_PATTERNS = [
    r"(?:what's|what is).*weather",
    r"your_new_pattern_here",
]
```

### Customizing the Conversational Router

Edit `backend_api/app/engine.py`:

```python
def _is_greeting(self, text: str) -> bool:
    greeting_keywords = [
        "bonjour", "salut", "hello", "hi",
        # Add more greetings
        "marhaba", "ahlan",
    ]
    return any(keyword in text.lower() for keyword in greeting_keywords)
```

### Adjusting Retrieval Quality

1. **Lower threshold** for more results (may include less relevant):
   ```
   SIMILARITY_THRESHOLD=0.35
   ```

2. **Increase top_k** for more context:
   ```
   TOP_K_RESULTS=10
   ```

3. **Add reranking** (if enabled):
   ```
   USE_RERANKER=true
   RERANKER_MODEL=BAAI/bge-reranker-v2-m3
   ```

---

## Testing

### Run All Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=backend_api --cov=data_ingestion

# Run specific test file
python -m pytest tests/test_backend_api.py -v

# Run tests matching pattern
python -m pytest tests/ -k "mock" -v
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_data_ingestion.py` | Parser, Embedder, VectorStore, Pipeline |
| `test_backend_api.py` | RAG Engine, API endpoints |
| `test_guardrails.py` | Input/Output validation |

### Manual Testing

```bash
# Test RAG with real Qdrant (no vLLM needed)
python test_rag_real.py

# Test Qdrant directly
python test_qdrant_direct.py
```

---

## Troubleshooting

### "I cannot find relevant information"

**Cause:** Similarity threshold too high or documents not ingested.

**Fix:**
1. Check Qdrant has documents: http://localhost:6333/dashboard
2. Lower `SIMILARITY_THRESHOLD` to 0.35-0.40
3. Re-ingest with `--recreate` flag

### "Article 5 not found" but document exists

**Cause:** Generic queries like "Article 5" don't match well semantically.

**Fix:**
- Use more specific queries: "Quelles sont les conditions pour l'autorisation d'exercice?"
- The system works best with natural language questions

### Embeddings take too long

**Cause:** BGE-M3 is a large model (~2GB).

**Fix:**
1. Use GPU: Set `device="cuda"` in embedder
2. Use smaller model: `BAAI/bge-small-en-v1.5`
3. Reduce batch size in `embedder.py`

### vLLM connection refused

**Cause:** vLLM server not running or wrong URL.

**Fix:**
1. Check vLLM is running: `docker ps | grep vllm`
2. Check logs: `docker logs atlas-vllm`
3. Verify URL in `.env`: `VLLM_URL=http://localhost:8000/v1`

### PowerShell command errors

**Important:** Don't use `&&` in PowerShell. Use `;` instead:

```powershell
# Wrong
cd project && python script.py

# Correct
cd project; python script.py
```

---

## Project Structure

```
ChatAssistantAcaps/
├── backend_api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # API configuration
│   │   ├── engine.py          # RAG engine
│   │   ├── guardrails/        # Input/output validation
│   │   ├── main.py            # FastAPI app
│   │   └── models.py          # Pydantic models
│   ├── Dockerfile
│   └── requirements.txt
├── data_ingestion/
│   ├── __init__.py
│   ├── config.py              # Ingestion config
│   ├── documents/             # Source Markdown files
│   ├── embedder.py            # BGE-M3 embeddings
│   ├── parser.py              # Markdown parser
│   ├── pipeline.py            # Orchestration
│   └── vector_store.py        # Qdrant operations
├── frontend_ui/
│   ├── src/
│   │   ├── App.tsx            # React app
│   │   ├── index.css          # Styles
│   │   └── main.tsx           # Entry point
│   ├── Dockerfile
│   └── package.json
├── inference_server/
│   ├── config.json
│   └── Dockerfile
├── vector_db/
│   ├── Dockerfile
│   └── qdrant_config.yaml
├── tests/
│   ├── test_backend_api.py
│   ├── test_data_ingestion.py
│   └── test_guardrails.py
├── docker-compose.yml
├── requirements.txt
├── env.example
└── README.md
```

---

## License

Internal use only - ACAPS

---

## Support

For issues or questions, contact the development team.

