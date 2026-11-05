from fastapi import APIRouter, Depends, HTTPException
from llama_index.embeddings.ollama import OllamaEmbedding

from app.dependencies import DatabasePool, get_db_pool, OllamaLLMClient, get_embed_model, get_llm_client
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag_service import RAGService
from app.services.retriever import VectorRetrieverService

router = APIRouter(prefix="/query", tags=["Query"])

@router.post("", response_model=QueryResponse)
def handle_query(
        request: QueryRequest,
        db_pool: DatabasePool = Depends(get_db_pool),
        embed_model: OllamaEmbedding = Depends(get_embed_model),
        llm_client: OllamaLLMClient = Depends(get_llm_client),
):
    try:
        retriever = VectorRetrieverService(db_pool, embed_model)
        rag_service = RAGService(retriever, llm_client)

        answer, sources = rag_service.answer(request.question, top_k=request.top_k)
        return QueryResponse(question=request.question, answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while processing the question: {str(e)}")