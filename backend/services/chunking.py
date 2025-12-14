from typing import List, Dict, Any
import re


class ContentChunker:
    """
    Service for chunking textbook content for RAG system
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        Split text into chunks of specified size with overlap
        """
        # Split text by paragraphs first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0

        for paragraph in paragraphs:
            # Estimate token count (roughly 1 token ~ 4 characters for English)
            paragraph_tokens = len(paragraph) // 4

            # If adding this paragraph would exceed chunk size
            if current_chunk_tokens + paragraph_tokens > self.chunk_size:
                # Save the current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'source': source,
                        'metadata': {'type': 'paragraph'}
                    })

                # Start a new chunk with overlap from the previous chunk
                if self.chunk_overlap > 0 and current_chunk:
                    # Take the last part of the current chunk for overlap
                    overlap_tokens = min(self.chunk_overlap, len(current_chunk))
                    current_chunk = current_chunk[-overlap_tokens:] + paragraph
                else:
                    current_chunk = paragraph
                current_chunk_tokens = len(current_chunk) // 4
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + paragraph
                current_chunk_tokens += paragraph_tokens

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'source': source,
                'metadata': {'type': 'paragraph'}
            })

        return chunks

    def chunk_markdown(self, markdown_content: str, source: str = "") -> List[Dict[str, Any]]:
        """
        Chunk markdown content by headers and sections
        """
        # Split by markdown headers (h1, h2, h3, etc.)
        header_pattern = r'(^|\n)(#{1,6})\s+(.+?)(?=\n|$)'
        sections = re.split(header_pattern, markdown_content)

        chunks = []
        current_section = ""

        # Process the split sections
        for i, section in enumerate(sections):
            # Check if this is a header
            if section.strip().startswith('#'):
                if current_section.strip():
                    # Chunk the accumulated content
                    header_chunks = self.chunk_text(current_section.strip(), source)
                    chunks.extend(header_chunks)

                # Start new section with header
                if i + 1 < len(sections):
                    current_section = section + " " + sections[i + 1]
                else:
                    current_section = section
            else:
                current_section += section

        # Process the last section
        if current_section.strip():
            header_chunks = self.chunk_text(current_section.strip(), source)
            chunks.extend(header_chunks)

        return chunks