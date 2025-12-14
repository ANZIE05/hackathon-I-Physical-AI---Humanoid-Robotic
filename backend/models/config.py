from pydantic import BaseModel
from typing import Optional


class RAGConfiguration(BaseModel):
    """
    Configuration for RAG system
    """
    model_provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7
    enable_rag: bool = True


class QueryRequest(BaseModel):
    """
    Request model for RAG queries
    """
    query: str
    selected_text: Optional[str] = None
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.7


class QueryResponse(BaseModel):
    """
    Response model for RAG queries
    """
    query: str
    response: str
    sources: List[str]
    confidence: float