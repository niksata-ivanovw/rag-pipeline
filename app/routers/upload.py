from fastapi import APIRouter, UploadFile, Depends, File, HTTPException
from llama_index.embeddings.ollama import OllamaEmbedding

from app.dependencies import DatabasePool, get_db_pool, get_embed_model
from app.schemas.upload import UploadResponse
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("", response_model=UploadResponse)
async def handle_upload(
        file: UploadFile = File(...),
        db_pool: DatabasePool = Depends(get_db_pool),
        embed_model: OllamaEmbedding = Depends(get_embed_model),
):
    if not file.filename.lower().endswith((".docx", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf are supported.")

    file_bytes = await file.read()
    ingestion = IngestionService(db_pool, embed_model)
    chunks_count = ingestion.ingest_bytes(file.filename, file_bytes)

    return UploadResponse(
        filename=file.filename,
        chunks_indexed=chunks_count,
        message=f"{file.filename} has been successfully processed and {chunks_count} chunks were created."
    )