# add project root to PYTHONPATH so src modules are visible
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from llama_index.embeddings.ollama import OllamaEmbedding
from starlette.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent))

from app.dependencies import DatabasePool, OllamaLLMClient
from app.routers import query, upload

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting: initializing connections and models...")
    app.state.db_pool = DatabasePool()
    app.state.embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
    app.state.llm_client = OllamaLLMClient(model_name="qwen3.5:9b", base_url="http://localhost:11434")
    print("all resources loaded successfully")
    yield
    print("stopping connections and models...")
    app.state.db_pool.close_all()

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