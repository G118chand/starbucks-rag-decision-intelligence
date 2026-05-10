"""Unit tests for processing.chunker.DocumentChunker."""

from typing import List

import pytest
from langchain.schema import Document

from processing.chunker import DocumentChunker


def test_chunk_size_within_bounds(sample_documents: List[Document]) -> None:
    """All produced chunks must fit within 125 % of the requested chunk_size."""
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_documents(sample_documents)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) <= 250, (
            f"Chunk exceeds 250 chars: {len(chunk.page_content)}"
        )


def test_metadata_preserved(sample_documents: List[Document]) -> None:
    """Every chunk must retain the 'source' key from its parent document."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_documents(sample_documents)
    for chunk in chunks:
        assert "source" in chunk.metadata, (
            f"Chunk missing 'source' in metadata: {chunk.metadata}"
        )


def test_chunk_stats_has_all_keys(sample_documents: List[Document]) -> None:
    """chunk_stats() must return all four expected keys."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_documents(sample_documents)
    stats = chunker.chunk_stats(chunks)
    for key in ("count", "avg_size", "min_size", "max_size"):
        assert key in stats, f"Missing key '{key}' in stats: {stats}"


def test_empty_input_returns_empty() -> None:
    """chunk_documents with an empty list must return an empty list."""
    chunker = DocumentChunker()
    result = chunker.chunk_documents([])
    assert result == []


def test_chunk_index_in_metadata(sample_documents: List[Document]) -> None:
    """Each chunk must carry chunk_index and chunk_total metadata fields."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_documents(sample_documents)
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata, (
            f"Missing chunk_index: {chunk.metadata}"
        )
        assert "chunk_total" in chunk.metadata, (
            f"Missing chunk_total: {chunk.metadata}"
        )
