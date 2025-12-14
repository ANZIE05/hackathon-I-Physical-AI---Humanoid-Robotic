import unittest
from services.chunking import ContentChunker


class TestContentChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = ContentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunk_text_basic(self):
        text = "This is a sample text. " * 10  # Creates a text longer than 100 chars
        chunks = self.chunker.chunk_text(text, "test_source")

        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)

        # Each chunk should have content, source, and metadata
        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('source', chunk)
            self.assertIn('metadata', chunk)
            self.assertEqual(chunk['source'], "test_source")

    def test_chunk_text_short(self):
        text = "Short text"
        chunks = self.chunker.chunk_text(text, "test_source")

        # Should create a single chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['content'], text)

    def test_chunk_markdown(self):
        markdown_content = """
# Header 1
This is content under header 1.

## Header 2
This is content under header 2.

### Header 3
This is content under header 3.
"""
        chunks = self.chunker.chunk_markdown(markdown_content, "test_markdown_source")

        # Should create chunks based on headers
        self.assertGreater(len(chunks), 0)

        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('source', chunk)
            self.assertIn('metadata', chunk)


if __name__ == '__main__':
    unittest.main()