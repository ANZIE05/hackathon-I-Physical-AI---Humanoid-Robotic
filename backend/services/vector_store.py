import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np


class VectorStore:
    """
    Service for interacting with vector storage for RAG
    """

    def __init__(self, collection_name: str = "textbook_content"):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Qdrant client (using local for now, can be configured for cloud)
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(url=qdrant_url)

        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Ensure the collection exists in Qdrant
        """
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Create the collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

    def add_chunk(self, content: str, source: str, metadata: Dict[str, Any] = None):
        """
        Add a content chunk to the vector store
        """
        # Generate embedding for the content
        embedding = self.embedding_model.encode(content).tolist()

        # Create a unique ID for this chunk
        chunk_id = str(uuid.uuid4())

        # Prepare the payload
        payload = {
            "content": content,
            "source": source,
            "metadata": metadata or {}
        }

        # Add to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant content chunks based on the query
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "content": result.payload["content"],
                "source": result.payload["source"],
                "metadata": result.payload["metadata"],
                "score": result.score
            })

        return results

    def remove_source(self, source: str):
        """
        Remove all chunks associated with a particular source
        """
        # Find all points with this source
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source)
                    )
                ]
            ),
            limit=10000  # Adjust as needed
        )

        # Extract IDs to delete
        ids_to_delete = [point.id for point in scroll_results[0]]

        if ids_to_delete:
            # Delete the points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids_to_delete
                )
            )

    def clear_collection(self):
        """
        Clear all content from the collection
        """
        # Get all point IDs
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000  # Adjust as needed
        )

        # Extract IDs to delete
        ids_to_delete = [point.id for point in scroll_results[0]]

        if ids_to_delete:
            # Delete all points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids_to_delete
                )
            )