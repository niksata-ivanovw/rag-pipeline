from typing import List

from llama_index.embeddings.ollama import OllamaEmbedding
from pgvector.psycopg2 import register_vector

from app.dependencies import DatabasePool
from app.schemas.query import ChunkSource


class VectorRetrieverService:
    def __init__(self, db_pool: DatabasePool, embed_model: OllamaEmbedding):
        self.db_pool = db_pool
        self.embed_model = embed_model

    def search(self, query_text: str, top_k: int = 3) -> List[ChunkSource]:
        # generate vector using bge-m3
        query_vector = self.embed_model.get_text_embedding(query_text)

        conn = self.db_pool.get_connection()
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                search_query = """
                    SELECT content, 1 - (embedding <=> %s::vector) AS similarity_score, metadata
                    FROM document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(search_query, (query_vector, query_vector, top_k))
                rows = cur.fetchall()

                sources = []
                for row in rows:
                    content, score, meta = row[0].strip(), float(row[1]), row[2] or {}
                    sources.append(
                        ChunkSource(
                            content=content,
                            similarity_score=score,
                            file_name=meta.get("file_name"),
                            page=meta.get("page"),
                            chunk_index=meta.get("chunk_index"),
                            section=meta.get("section"),
                            metadata=meta
                        )
                    )

                return sources
        finally:
            self.db_pool.release_connection(conn)