"""Structural tests for chunk documents produced by the ingestion + chunking pipeline."""

import os
import sys
from typing import List

import pytest
from langchain.schema import Document

# project root on path (conftest.py also sets this, but explicit here for clarity)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_sample_chunks_not_empty(sample_chunks: List[Document]) -> None:
    """The chunker must produce at least one chunk from the sample corpus."""
    assert len(sample_chunks) > 0


def test_all_chunks_have_text(sample_chunks: List[Document]) -> None:
    """Every chunk must contain non-empty page_content."""
    for chunk in sample_chunks:
        assert len(chunk.page_content) > 0, (
            f"Empty page_content in chunk: {chunk.metadata}"
        )


def test_all_chunks_have_source_metadata(sample_chunks: List[Document]) -> None:
    """Every chunk must carry a 'source' key in its metadata."""
    for chunk in sample_chunks:
        assert "source" in chunk.metadata, (
            f"Missing 'source' metadata in chunk: {chunk.metadata}"
        )


def test_chunk_total_consistent(sample_chunks: List[Document]) -> None:
    """Every chunk must carry chunk_total >= 1 from its parent document."""
    for chunk in sample_chunks:
        assert "chunk_total" in chunk.metadata, (
            f"Missing 'chunk_total' in chunk: {chunk.metadata}"
        )
        assert chunk.metadata["chunk_total"] >= 1, (
            f"chunk_total must be >= 1, got {chunk.metadata['chunk_total']}"
        )
