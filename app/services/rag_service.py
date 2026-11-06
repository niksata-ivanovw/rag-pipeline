from typing import List, Tuple

from llama_index.llms.ollama import Ollama
from app.schemas.query import ChunkSource
from app.services.retriever import VectorRetrieverService


class RAGService:
    def __init__(self, retriever: VectorRetrieverService, llm_client: Ollama):
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[ChunkSource]]:
        # find most relevant chunks
        sources = self.retriever.search(question, top_k=top_k)

        if not sources:
            return "Didn't find any relevant documents in the database.", []

        # form context
        context = "\n\n---\n\n".join(
            [f"Източник: {s.file_name or s.metadata.get('file_name', 'документ')}\n{s.content}"
             for s in sources
             ])

        prompt = f"""You are a precise assistant. Answer the user's question SOLELY based on the context provided below.
        If the answer is not contained in the context, state that the information is missing.
        CONTEXT:
        {context}
        QUESTION:
        {question}
        ANSWER:"""

        # generate the answer using llm
        response = self.llm_client.complete(prompt)
        answer_text = response.text

        return answer_text, sources