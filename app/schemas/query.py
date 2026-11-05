from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ChunkSource(BaseModel):
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, example="Колко точки получава човекът на позиция 4?")
    top_k: int = Field(default=3, ge=1, le=10, description="Брой Резултати от търсенето")

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[ChunkSource]