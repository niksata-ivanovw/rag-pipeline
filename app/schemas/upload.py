from pydantic import BaseModel

class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str