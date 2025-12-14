import unittest
from unittest.mock import Mock, patch
from services.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Mock the Qdrant client
        with patch('services.vector_store.QdrantClient') as mock_client:
            self.mock_qdrant_client = Mock()
            mock_client.return_value = self.mock_qdrant_client

            # Create VectorStore instance
            self.vector_store = VectorStore()

    @patch('services.vector_store.SentenceTransformer')
    def test_add_chunk(self, mock_sentence_transformer):
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value = mock_model

        # Call add_chunk
        self.vector_store.add_chunk("test content", "test_source", {"type": "test"})

        # Verify that upsert was called
        self.mock_qdrant_client.upsert.assert_called_once()

    @patch('services.vector_store.SentenceTransformer')
    def test_search(self, mock_sentence_transformer):
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value = mock_model

        # Mock search results
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.payload = {
            "content": "test content",
            "source": "test_source",
            "metadata": {"type": "test"}
        }
        mock_result.score = 0.9
        self.mock_qdrant_client.search.return_value = [mock_result]

        # Call search
        results = self.vector_store.search("test query", top_k=5)

        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "test content")
        self.assertEqual(results[0]["source"], "test_source")
        self.assertEqual(results[0]["score"], 0.9)


if __name__ == '__main__':
    unittest.main()