"""One-shot script to ingest data, chunk it, embed it, and write the FAISS index.

Usage:
    python scripts/build_index.py
"""

import logging
import sys
import time
from pathlib import Path

# ── make project root importable ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")

from embeddings.embedder import EmbeddingEngine
from ingestion.data_loader import DataIngestionPipeline
from processing.chunker import DocumentChunker
from vectorstore.faiss_store import FAISSVectorStore


def main() -> None:
    """Run the full index-build pipeline and report timings at each stage."""
    total_start = time.perf_counter()
    print("\n" + "=" * 60)
    print("  Building FAISS Index — Starbucks RAG Platform")
    print("=" * 60)

    # ── 1. ingest ─────────────────────────────────────────────────────────────
    print("\n[1/4] Ingesting documents …")
    t0 = time.perf_counter()
    pipeline = DataIngestionPipeline()
    docs = pipeline.load_all()
    elapsed = time.perf_counter() - t0
    print(f"      Loaded {len(docs):,} documents  ({elapsed:.1f}s)")

    # ── 2. chunk ──────────────────────────────────────────────────────────────
    print("\n[2/4] Chunking documents …")
    t0 = time.perf_counter()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(docs)
    stats = chunker.chunk_stats(chunks)
    elapsed = time.perf_counter() - t0
    print(f"      Chunks : {stats['count']:,}")
    print(f"      Avg size : {stats['avg_size']} chars")
    print(f"      Min/Max  : {stats['min_size']} / {stats['max_size']} chars")
    print(f"      Time     : {elapsed:.1f}s")

    # ── 3. embed ──────────────────────────────────────────────────────────────
    print("\n[3/4] Loading embedding model …")
    t0 = time.perf_counter()
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    info = embedder.get_model_info()
    elapsed = time.perf_counter() - t0
    print(f"      Model      : {info['model_name']}")
    print(f"      Dimensions : {info['dimensions']}")
    print(f"      Load time  : {elapsed:.1f}s")

    # ── 4. build index ────────────────────────────────────────────────────────
    print(f"\n[4/4] Embedding {len(chunks):,} chunks and building FAISS index …")
    print("      (this may take 30–90 seconds on first run)")
    t0 = time.perf_counter()
    store = FAISSVectorStore(index_dir=str(PROJECT_ROOT / "vectorstore" / "faiss_index"))
    store.build_index(chunks, embedder)
    elapsed = time.perf_counter() - t0

    index_stats = store.get_stats()
    print(f"      Vectors    : {index_stats['total_vectors']:,}")
    print(f"      Dimensions : {index_stats['dimensions']}")
    print(f"      Index size : {index_stats['index_size_mb']} MB")
    print(f"      Time       : {elapsed:.1f}s")

    # ── summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print("  Index build complete!")
    print(f"  Total time : {total_elapsed:.1f}s")
    print(f"  Index path : {PROJECT_ROOT / 'vectorstore' / 'faiss_index'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
