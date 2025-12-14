from typing import List, Dict, Any
from .chunking import ContentChunker
from .vector_store import VectorStore
import os
import glob
import markdown
from pathlib import Path


class ContentIndexer:
    """
    Service for indexing textbook content into the vector store
    """

    def __init__(self, vector_store: VectorStore, chunker: ContentChunker):
        self.vector_store = vector_store
        self.chunker = chunker

    def index_textbook_content(self, content_dir: str = "docusaurus/docs") -> int:
        """
        Index all textbook content from the docs directory
        """
        indexed_count = 0

        # Find all markdown files in the content directory
        md_files = glob.glob(f"{content_dir}/**/*.md", recursive=True)

        for file_path in md_files:
            try:
                # Read the markdown file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract the relative path for the source identifier
                source = os.path.relpath(file_path, content_dir)

                # Chunk the content
                chunks = self.chunker.chunk_markdown(content, source)

                # Add chunks to vector store
                for chunk in chunks:
                    self.vector_store.add_chunk(
                        content=chunk['content'],
                        source=chunk['source'],
                        metadata=chunk['metadata']
                    )
                    indexed_count += 1

                print(f"Indexed {len(chunks)} chunks from {source}")

            except Exception as e:
                print(f"Error indexing {file_path}: {str(e)}")

        return indexed_count

    def index_single_document(self, content: str, source: str) -> int:
        """
        Index a single document
        """
        chunks = self.chunker.chunk_markdown(content, source)

        for chunk in chunks:
            self.vector_store.add_chunk(
                content=chunk['content'],
                source=chunk['source'],
                metadata=chunk['metadata']
            )

        return len(chunks)

    def update_document(self, content: str, source: str) -> int:
        """
        Update an existing document in the index
        """
        # First remove the old document
        self.vector_store.remove_source(source)

        # Then add the new version
        return self.index_single_document(content, source)

    def delete_document(self, source: str):
        """
        Remove a document from the index
        """
        self.vector_store.remove_source(source)