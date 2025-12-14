import unittest
from unittest.mock import Mock, patch
from models.config import QueryRequest
from services.rag_service import RAGService


class TestRAGService(unittest.TestCase):
    def setUp(self):
        # Mock the vector store and chunker
        self.mock_vector_store = Mock()
        self.mock_chunker = Mock()

        # Create RAGService with mocks
        with patch('services.rag_service.openai'):
            self.rag_service = RAGService(self.mock_vector_store, self.mock_chunker)

    def test_query_with_selected_text(self):
        # Create a query request with selected text
        request = QueryRequest(
            query="What is ROS 2?",
            selected_text="ROS 2 is the next generation of Robot Operating System.",
            top_k=5,
            temperature=0.7
        )

        # Mock the internal method
        self.rag_service._generate_response_from_selected_text = Mock(return_value="ROS 2 response")

        # Call the query method
        result = self.rag_service.query(request)

        # Verify the result
        self.assertEqual(result.query, "What is ROS 2?")
        self.assertEqual(result.response, "ROS 2 response")
        self.assertEqual(result.sources, ["selected_text"])

    def test_query_without_selected_text(self):
        # Create a query request without selected text
        request = QueryRequest(
            query="What is Physical AI?",
            selected_text=None,
            top_k=3,
            temperature=0.5
        )

        # Mock the vector store search
        mock_chunks = [
            {
                "content": "Physical AI content",
                "source": "test_source_1",
                "score": 0.9
            },
            {
                "content": "More Physical AI content",
                "source": "test_source_2",
                "score": 0.8
            }
        ]
        self.mock_vector_store.search.return_value = mock_chunks

        # Mock the internal method
        self.rag_service._generate_response_from_context = Mock(return_value="Physical AI response")

        # Call the query method
        result = self.rag_service.query(request)

        # Verify the result
        self.assertEqual(result.query, "What is Physical AI?")
        self.assertEqual(result.response, "Physical AI response")
        self.assertIn("test_source_1", result.sources)
        self.assertIn("test_source_2", result.sources)

    def test_query_no_results(self):
        # Create a query request
        request = QueryRequest(
            query="What is unknown topic?",
            selected_text=None,
            top_k=5,
            temperature=0.7
        )

        # Mock the vector store to return no results
        self.mock_vector_store.search.return_value = []

        # Call the query method
        result = self.rag_service.query(request)

        # Verify the result
        self.assertIn("couldn't find any relevant content", result.response)


if __name__ == '__main__':
    unittest.main()