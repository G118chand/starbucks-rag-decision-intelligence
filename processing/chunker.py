"""Text chunking utilities for splitting LangChain Documents into retrieval-ready chunks."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits Documents into overlapping chunks while preserving all metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Initialise the chunker with a RecursiveCharacterTextSplitter.

        Args:
            chunk_size: Maximum character length per chunk.
            chunk_overlap: Number of characters shared between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of Documents into smaller chunks, preserving and extending metadata.

        Each chunk inherits all metadata from its source Document and gains two extra fields:
        ``chunk_index`` (0-based position within the source) and ``chunk_total`` (total splits
        produced from that source).

        Args:
            documents: Source Documents to split.

        Returns:
            Flat list of chunk Documents ordered by source then position.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            splits = self.splitter.split_text(doc.page_content)
            total = len(splits)
            for idx, text in enumerate(splits):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_total": total,
                }
                all_chunks.append(Document(page_content=text, metadata=chunk_metadata))

        logger.info(
            "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
            len(documents),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    def chunk_stats(self, chunks: List[Document]) -> dict:
        """Compute descriptive statistics over a list of chunk Documents.

        Args:
            chunks: Chunk Documents as returned by :meth:`chunk_documents`.

        Returns:
            Dictionary with keys ``count``, ``avg_size``, ``min_size``, ``max_size``.
        """
        if not chunks:
            return {"count": 0, "avg_size": 0, "min_size": 0, "max_size": 0}

        sizes = [len(c.page_content) for c in chunks]
        return {
            "count": len(sizes),
            "avg_size": round(sum(sizes) / len(sizes), 1),
            "min_size": min(sizes),
            "max_size": max(sizes),
        }
