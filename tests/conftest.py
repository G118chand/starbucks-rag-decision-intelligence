"""Shared pytest fixtures for the Starbucks RAG platform test suite."""

import os
import sys
from typing import List

import pytest

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain.schema import Document
from processing.chunker import DocumentChunker


@pytest.fixture
def sample_documents() -> List[Document]:
    """Five representative enterprise documents covering the main data domains."""
    return [
        Document(
            page_content=(
                "Pacific Northwest sales declined 12% YoY. Seattle stores wait time 5.8 min. "
                "Competitor Dutch Bros opened nearby causing traffic loss."
            ),
            metadata={
                "source": "q3_report.md",
                "region": "Pacific Northwest",
                "data_type": "unstructured",
            },
        ),
        Document(
            page_content=(
                "Loyalty member churn highest in stores with wait over 6 minutes. "
                "34 percent of churned members had 2 or more negative experience tags "
                "in last 60 days. Gold tier is 41 percent of revenue."
            ),
            metadata={
                "source": "churn_analysis.md",
                "data_type": "unstructured",
            },
        ),
        Document(
            page_content=(
                "Cold Brew Concentrate stockout 412 incidents. Oat Milk 384 incidents. "
                "Total OOS revenue impact 94000 dollars. West Coast port delays increased "
                "lead time from 3 to 7 days."
            ),
            metadata={
                "source": "inventory_ops.md",
                "data_type": "unstructured",
            },
        ),
        Document(
            page_content=(
                "Gold tier members 41 percent revenue, 6.2 visits per month. "
                "Birthday reward redemption 67 percent highest of all offer types. "
                "Gamification challenges lift retention 22 percent."
            ),
            metadata={
                "source": "loyalty_report.md",
                "data_type": "unstructured",
            },
        ),
        Document(
            page_content=(
                "Austin TX plus 23 percent YoY revenue, average wait time 3.1 minutes, "
                "mobile order rate 71 percent. Nashville plus 19 percent. "
                "Charlotte plus 15 percent."
            ),
            metadata={
                "source": "benchmarks.md",
                "data_type": "unstructured",
            },
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents: List[Document]) -> List[Document]:
    """Chunked version of sample_documents using a small chunk size for test speed."""
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    return chunker.chunk_documents(sample_documents)
