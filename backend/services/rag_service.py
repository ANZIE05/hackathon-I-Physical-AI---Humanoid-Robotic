from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .chunking import ContentChunker
from ..models.config import QueryRequest, QueryResponse
import openai
import os


class RAGService:
    """
    Service for RAG (Retrieval-Augmented Generation) functionality
    """

    def __init__(self, vector_store: VectorStore, chunker: ContentChunker):
        self.vector_store = vector_store
        self.chunker = chunker

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query and return a response based on retrieved content
        """
        # If selected_text is provided, only use that text for answering
        if request.selected_text:
            # Generate response based only on selected text
            response = self._generate_response_from_selected_text(
                request.query,
                request.selected_text,
                request.temperature
            )
            sources = ["selected_text"]
            confidence = 1.0  # High confidence since using exact selected text
        else:
            # Retrieve relevant chunks from the vector store
            retrieved_chunks = self.vector_store.search(
                query=request.query,
                top_k=request.top_k
            )

            if not retrieved_chunks:
                return QueryResponse(
                    query=request.query,
                    response="I couldn't find any relevant content to answer your question.",
                    sources=[],
                    confidence=0.0
                )

            # Extract content from retrieved chunks
            retrieved_content = []
            sources = []
            for chunk in retrieved_chunks:
                retrieved_content.append(chunk["content"])
                if chunk["source"] not in sources:
                    sources.append(chunk["source"])

            # Combine retrieved content
            context = "\n\n".join(retrieved_content)

            # Generate response based on retrieved context
            response = self._generate_response_from_context(
                request.query,
                context,
                request.temperature
            )

            # Calculate average score as confidence
            avg_score = sum([chunk["score"] for chunk in retrieved_chunks]) / len(retrieved_chunks)
            confidence = min(avg_score, 1.0)  # Ensure confidence is between 0 and 1

        return QueryResponse(
            query=request.query,
            response=response,
            sources=sources,
            confidence=confidence
        )

    def _generate_response_from_context(self, query: str, context: str, temperature: float = 0.7) -> str:
        """
        Generate a response using OpenAI based on the provided context
        """
        prompt = f"""
        You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        Use the following context to answer the question. If the context doesn't contain
        relevant information, say so clearly.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Answer questions based only on the provided context. Be accurate and helpful."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _generate_response_from_selected_text(self, query: str, selected_text: str, temperature: float = 0.7) -> str:
        """
        Generate a response using OpenAI based only on the selected text
        """
        prompt = f"""
        You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        Answer the question based ONLY on the following selected text. Do not use
        any other knowledge or information beyond what is provided here.

        Selected Text:
        {selected_text}

        Question: {query}

        Answer:
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Answer questions based ONLY on the provided selected text. Do not use any external knowledge. Be accurate and helpful."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def index_textbook(self) -> int:
        """
        Index the entire textbook content
        """
        from .indexing import ContentIndexer
        indexer = ContentIndexer(self.vector_store, self.chunker)
        indexed_count = indexer.index_textbook_content()
        return indexed_count