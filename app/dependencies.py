import httpx
import psycopg2.pool
from fastapi import Request
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

DB_CONFIG = dict(
    dbname='rag_database',
    user='postgres',
    password='secret',
    host='127.0.0.1',
    port='5433'
)

class DatabasePool:
    def __init__(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)

    def get_connection(self):
        conn = self.pool.getconn()
        return conn

    def release_connection(self, conn):
        self.pool.putconn(conn)

    def close_all(self):
        self.pool.closeall()

class OllamaLLMClient:
    """Direct client to the local Ollama for answer generation"""
    def __init__(self, model_name: str = "qwen3.5:9b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": -1
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

def get_db_pool(request: Request) -> DatabasePool:
    return request.app.state.db_pool

def get_embed_model(request: Request) -> OllamaEmbedding:
    return request.app.state.embed_model

def get_llm_client(request: Request) -> Ollama:
    return request.app.state.llm_client