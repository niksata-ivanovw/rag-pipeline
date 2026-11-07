from pydantic import BaseModel

class UploadResponse(BaseModel):
    filename: str
    minio_bucket: str
    chunks_indexed: int
    status: str
    message: str