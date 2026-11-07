from fastapi import APIRouter, UploadFile, Depends, File, HTTPException
from app.schemas.upload import UploadResponse
from app.dependencies import get_db_pool, get_embed_model, get_storage_service, DatabasePool
from app.services.ingestion_service import IngestionService
from app.services.storage_service import MinioStorageService
from llama_index.embeddings.ollama import OllamaEmbedding

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("", response_model=UploadResponse)
async def handle_upload(
        file: UploadFile = File(...),
        db_pool: DatabasePool = Depends(get_db_pool),
        embed_model: OllamaEmbedding = Depends(get_embed_model),
        storage_service: MinioStorageService = Depends(get_storage_service)
):
    if not file.filename.lower().endswith((".docx", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf are supported.")

    file_bytes = await file.read()

    try:
        storage_service.upload_file(
            filename=file.filename,
            file_bytes=file_bytes,
            content_type=file.content_type or "application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with MinIO: {str(e)}")

    try:
        ingestion = IngestionService(db_pool, embed_model)
        chunks_count = ingestion.ingest_bytes(file.filename, file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with pgvector: {str(e)}")

    return UploadResponse(
        filename=file.filename,
        minio_bucket=storage_service.bucket_name,
        chunks_indexed=chunks_count,
        status="completed",
        message=f"{file.filename} has been successfully processed and {chunks_count} chunks were created."
    )