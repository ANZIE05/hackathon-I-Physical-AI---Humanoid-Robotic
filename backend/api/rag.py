from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..models.config import QueryRequest, QueryResponse, RAGConfiguration
from ..services.rag_service import RAGService
from ..services.vector_store import VectorStore
from ..services.chunking import ContentChunker
import os


router = APIRouter()


def get_rag_service():
    """
    Dependency to get the RAG service instance
    """
    # Initialize services
    vector_store = VectorStore()
    chunker = ContentChunker()
    rag_service = RAGService(vector_store, chunker)
    return rag_service


@router.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Process a RAG query and return a response
    """
    try:
        response = rag_service.query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/index-textbook")
async def index_textbook(rag_service: RAGService = Depends(get_rag_service)):
    """
    Index the entire textbook content
    """
    try:
        indexed_count = rag_service.index_textbook()
        return {"message": f"Successfully indexed {indexed_count} content chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing textbook: {str(e)}")


@router.get("/config")
async def get_config():
    """
    Get the current RAG configuration
    """
    config = RAGConfiguration()
    return config


@router.post("/config")
async def update_config(config: RAGConfiguration):
    """
    Update the RAG configuration
    """
    # In a real implementation, this would update the configuration
    # For now, we'll just return the provided config
    return config


@router.get("/sources")
async def get_sources(rag_service: RAGService = Depends(get_rag_service)):
    """
    Get a list of all indexed sources
    """
    # This would require implementing a method to retrieve all sources from the vector store
    # For now, return a placeholder implementation
    vector_store = VectorStore()

    # Get all points to extract unique sources
    scroll_results = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        limit=10000
    )

    sources = set()
    for point in scroll_results[0]:
        source = point.payload.get("source", "unknown")
        sources.add(source)

    return {"sources": list(sources)}