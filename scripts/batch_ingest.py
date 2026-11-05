"""
scripts/batch_ingest.py

CLI script to batch index documents from a folder (e.g., documents/)
using the production IngestionService.

Usage:
    python scripts/batch_ingest.py
    python scripts/batch_ingest.py --dir path/to/other_folder
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so the 'app' package can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.embeddings.ollama import OllamaEmbedding
from app.dependencies import DatabasePool
from app.services.ingestion_service import IngestionService

SUPPORTED_EXTENSIONS = (".docx", ".pdf")
DEFAULT_DOCS_DIR = PROJECT_ROOT / "documents"


def batch_ingest(docs_dir: Path):
    if not docs_dir.exists():
        print(f"Error: Directory '{docs_dir}' does not exist!")
        return

    # Find all supported files
    files = sorted(
        [f for f in docs_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if not files:
        print(f"No .docx or .pdf files found in '{docs_dir}'.")
        return

    print(f"Found {len(files)} file(s) to index in '{docs_dir.name}':")
    for f in files:
        print(f"  - {f.name}")
    print("-" * 50)

    # Initialize shared resources
    print("Initializing database connection pool and embedding model...")
    db_pool = DatabasePool()
    embed_model = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")

    # Instantiate the production IngestionService
    service = IngestionService(db_pool=db_pool, embed_model=embed_model)

    total_chunks = 0
    successful_files = 0

    try:
        for idx, file_path in enumerate(files, start=1):
            filename = file_path.name
            print(f"[{idx}/{len(files)}] Processing {filename}...", end=" ", flush=True)

            try:
                # Read file bytes and pass directly to IngestionService
                file_bytes = file_path.read_bytes()
                chunks_count = service.ingest_bytes(filename=filename, file_bytes=file_bytes)

                print(f"Done! ({chunks_count} chunks)")
                total_chunks += chunks_count
                successful_files += 1

            except Exception as e:
                print(f"ERROR: {e}")

        print("=" * 50)
        print(f"Successfully processed: {successful_files}/{len(files)} files.")
        print(f"Total chunks written to DB: {total_chunks}")

    finally:
        # Always close connection pool resources
        db_pool.close_all()
        print("Database connections closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingestion of documents into the RAG database.")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_DOCS_DIR),
        help="Path to documents directory (default: documents/)",
    )
    args = parser.parse_args()

    batch_ingest(Path(args.dir))