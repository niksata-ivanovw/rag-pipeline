# RAG Backend API

A production-grade, modular Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **PostgreSQL (pgvector)**, **Ollama (`bge-m3` & `qwen3.5:9b`)**, **LlamaIndex**, and **MinIO Object Storage**.

---

## 🏗 Architecture & Overview

The project is structured according to the official **FastAPI "Bigger Applications"** convention with strict separation between API routing, dependency injection, and core business logic:

* **Object Storage (MinIO):** Stores the raw, unmodified `.docx` and `.pdf` files (S3-compatible).
* **Vector Database (PostgreSQL + pgvector):** Stores parsed text chunks, rich metadata (filename, page, chunk index, section), and vector embeddings.
* **Embeddings (Ollama `bge-m3`):** High-quality multilingual embeddings (1024 dimensions) with startup pre-warming.
* **LLM Generation (Ollama `qwen3.5:9b` via LlamaIndex):** Grounded question-answering with strict context constraints.
* **Chunking Engine:** Structure-aware, regex-driven chunking with heading/paragraph heuristics and boundary preservation.

---

## 📂 Project Structure

```text
rag-backend/
├── app/
│   ├── main.py                  # Entry point, CORS, lifespan startup/shutdown, routing
│   ├── dependencies.py          # Shared dependencies (DB pool, Ollama models, MinIO client)
│   │
│   ├── schemas/                 # Pydantic data validation schemas
│   │   ├── query.py             # QueryRequest, QueryResponse, ChunkSource
│   │   └── upload.py            # UploadResponse
│   │
│   ├── routers/                 # Thin HTTP controllers
│   │   ├── query.py             # POST /query
│   │   └── upload.py            # POST /upload
│   │
│   └── services/                # Decoupled business logic
│       ├── parsers/             # DOCX and PDF block parsers
│       ├── chunking.py          # Structure-first & regex chunking algorithm
│       ├── retriever.py         # pgvector cosine similarity search
│       ├── storage_service.py   # MinIO S3 object storage operations
│       ├── ingestion_service.py # Parse -> chunk -> embed -> pgvector pipeline
│       └── rag_service.py       # Context assembly & LLM generation via LlamaIndex
│
├── scripts/                     # Standalone CLI tools & utilities
│   ├── batch_ingest.py          # Batch indexing of local document directories
│   ├── test_search.py           # Quick CLI semantic search verification
│   └── print_content.py         # Document parsing debug utility
│
├── requirements.txt             # Project dependencies
└── README.md
```

---

## ⚙️ Prerequisites & Infrastructure

### 1. Ollama Models
Ensure [Ollama](https://ollama.com/) is installed and running, then pull the required models:
```bash
ollama pull bge-m3
ollama pull qwen3.5:9b
```

### 2. Docker Containers

* **PostgreSQL with pgvector (Port 5433):**
  ```bash
  docker run -d --name pgvector-rag \
    -e POSTGRES_DB=rag_database \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=secret \
    -p 5433:5432 \
    ankane/pgvector
  ```

* **MinIO Object Storage (Ports 9000 & 9001):**
  ```bash
  docker run -d --name rag-minio \
    -p 9000:9000 \
    -p 9001:9001 \
    -e MINIO_ROOT_USER=minioadmin \
    -e MINIO_ROOT_PASSWORD=minioadmin \
    minio/minio server /data --console-address ":9001"
  ```
  * **MinIO Web Console:** [http://localhost:9001](http://localhost:9001) (User: `minioadmin` / Pass: `minioadmin`)

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd rag-backend

# Activate your virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the FastAPI Application
```bash
uvicorn app.main:app --reload --port 8000
```
During startup, the application pre-warms the `bge-m3` embedding model and `qwen3.5:9b` in memory to eliminate cold-start latency for the first request.

* **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 API Endpoints

### 1. `POST /query`
Performs semantic search against `document_chunks` using cosine distance (`<=>`), feeds retrieved context to the LLM, and returns the generated answer with source citations.

* **Request Body:**
  ```json
  {
    "question": "Кои са образите в 'Аз искам да те помня все така'?",
    "top_k": 3
  }
  ```

* **Response Example:**
  ```json
  {
    "question": "Кои са образите в 'Аз искам да те помня все така'?",
    "answer": "Основните образи са на лирическия аз и любимата, представени в мига на неизбежна раздяла...",
    "sources": [
      {
        "content": "Аз искам да те помня все така: бездомна, безнадеждна и унила...",
        "similarity_score": 0.8732,
        "file_name": "az_iskam_da_te_pomnq_vse_taka_dimcho_debelqnov.docx",
        "page": 1,
        "chunk_index": 0,
        "section": "1. Образи и мотиви"
      }
    ]
  }
  ```

### 2. `POST /upload`
Uploads a `.docx` or `.pdf` file via `multipart/form-data`.
1. Uploads original binary to MinIO bucket `rag-documents`.
2. Extracts structured text blocks.
3. Chunks text using regex & structural boundaries.
4. Computes embeddings via `bge-m3`.
5. Indexes chunks and metadata into PostgreSQL `document_chunks`.

* **Response Example:**
  ```json
  {
    "filename": "obqva.docx",
    "minio_bucket": "rag-documents",
    "chunks_indexed": 12,
    "status": "completed",
    "message": "File successfully stored in MinIO and indexed into pgvector."
  }
  ```

---

## 🛠 CLI Scripts

* **Batch Ingest Local Documents:**
  ```bash
  python scripts/batch_ingest.py --dir path/to/folder
  ```
* **CLI Semantic Search Test:**
  ```bash
  python scripts/test_search.py
  ```
