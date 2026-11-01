"""
main.py - ingest every .docx / .pdf in a folder into the document_chunks table.

    python main.py                  # real run: Ollama bge-m3 + PostgreSQL
    python main.py --dry-run        # chunk + embed, print a summary, write nothing to the DB
    python main.py --dry-run --stub # no Ollama either (mechanics only)

Layout: src/main.py, src/chunking.py, src/parsers/{__init__,common,docx_parser,pdf_parser}.py
"""
import os
import sys

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from chunking import chunk_blocks, _stub_embed
from parsers import read_blocks, SUPPORTED

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")
DB = dict(dbname="rag_database", user="postgres", password="secret", host="127.0.0.1", port="5433")


def get_embed_fn(stub: bool):
    if stub:
        return _stub_embed
    from llama_index.embeddings.ollama import OllamaEmbedding  # imported lazily

    model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
    return model.get_text_embedding_batch


def ingest_file(cur, path: str, document_id: int, embed_fn, dry_run: bool) -> int:
    file_name = os.path.basename(path)

    blocks = read_blocks(path)  # parsing
    chunks = chunk_blocks(  # chunking
        blocks, embed_fn, file_name=file_name, percentile=90, min_chars=300, max_chars=1200
    )
    if not chunks:
        print(f"{file_name}: no content, skipped")
        return 0

    # Embed the text WITH title/section context; store the clean content.
    vectors = embed_fn([c.embed_text for c in chunks])
    print(f"{file_name}: {len(chunks)} chunks, vector size {len(vectors[0])}")

    if dry_run:
        return len(chunks)

    # Re-running must not duplicate rows: remove this file's old chunks first.
    cur.execute("DELETE FROM document_chunks WHERE metadata->>'file_name' = %s", (file_name,))

    rows = [
        (document_id, c.content, psycopg2.extras.Json(c.metadata), vec)
        for c, vec in zip(chunks, vectors)
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO document_chunks (document_id, content, metadata, embedding) VALUES %s",
        rows,
    )
    return len(chunks)


def main():
    dry_run = "--dry-run" in sys.argv
    embed_fn = get_embed_fn("--stub" in sys.argv)
    files = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(SUPPORTED))

    conn = None if dry_run else psycopg2.connect(**DB)
    try:
        cur = None
        if conn:
            register_vector(conn)
            cur = conn.cursor()

        total = 0
        for document_id, name in enumerate(files, start=1):  # one id per file instead of always 1
            total += ingest_file(cur, os.path.join(DOCS_DIR, name), document_id, embed_fn, dry_run)

        if conn:
            conn.commit()  # all files or nothing
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            print(f"done: {total} chunks written, table now has {cur.fetchone()[0]} rows")
        else:
            print(f"dry run: {total} chunks would be written")
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()