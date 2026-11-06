# add project root to PYTHONPATH so src modules are visible
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from starlette.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent))

from app.dependencies import DatabasePool, OllamaLLMClient
from app.routers import query, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag.startup")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("initializing database connection pool and clients"
                "")
    app.state.db_pool = DatabasePool()
    app.state.embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
    app.state.llm_client = Ollama(model="qwen3.5:9b", base_url="http://localhost:11434", request_timeout=120.0)

    logger.info("warming up embedding model 'bge-m3' in Ollama memory...")
    start_time = time.perf_counter()
    try:
        _ = app.state.embed_model.get_text_embedding("warmup probe")
        _ = app.state.llm_client.complete("Hi", max_tokens=1)
        duration = time.perf_counter() - start_time
        logger.info(f"Embedding model 'bge-m3' and LLM model 'qwen3.5:9b' warmed up and ready in {duration:.2f}s.")
    except Exception as e:
        logger.warning(f"Embedding warmup warning: {e}. Ollama may still load lazily on first query.")

    logger.info("All dependencies initialized successfully! API is ready to serve requests.")

    yield

    # SHUTDOWN
    logger.info("Shutting down: closing database connection pool...")
    app.state.db_pool.close_all()
    logger.info("All connections closed.")

app = FastAPI(
    title="RAG Backend API",
    description="FastAPI Interface for local RAG pipeline with pgvector and Ollama",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(upload.router)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "app": "RAG Backend API"}

@app.get("/")
def root():
    return {
        "message": "The API is running",
        "docs": "http://127.0.0.1:8000/docs"
    }