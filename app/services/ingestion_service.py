import os
import tempfile

import psycopg2.extras
from llama_index.embeddings.ollama import OllamaEmbedding
from pgvector.psycopg2 import register_vector

from app.dependencies import DatabasePool
from app.services.chunking import chunk_blocks
from app.services.parsers import read_blocks


class IngestionService:
    def __init__(self, db_pool: DatabasePool, embed_model: OllamaEmbedding):
        self.db_pool = db_pool
        self.embed_model = embed_model

    def ingest_bytes(self, filename: str, file_bytes: bytes) -> int:
        suffix = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # get blocks through parsers
            blocks = read_blocks(tmp_path)

            # generate chunks
            chunks = chunk_blocks(
                blocks,
                self.embed_model.get_text_embedding_batch,
                file_name=filename,
                percentile=90,
                min_chars=300,
                max_chars=1200
            )

            if not chunks:
                return 0

            # vectorize
            vectors = self.embed_model.get_text_embedding_batch([c.embed_text for c in chunks])

            # write in PostgreSQL
            conn = self.db_pool.get_connection()
            try:
                register_vector(conn)
                with conn.cursor() as cur:
                    # delete old chunks for the same file
                    cur.execute("DELETE FROM document_chunks WHERE metadata->>'file_name' = %s", (filename,))

                    rows = [
                        (1, c.content, psycopg2.extras.Json(c.metadata), vec) for c, vec in zip(chunks, vectors)
                    ]
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO document_chunks (document_id, content, metadata, embedding) VALUES %s",
                        rows,
                    )
                conn.commit()
                return len(chunks)
            finally:
                self.db_pool.release_connection(conn)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)