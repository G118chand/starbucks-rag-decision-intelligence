"""FAISS vector store — build, persist, and query dense embeddings."""

import json
import logging
import os
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Coerce a value to a JSON-serialisable Python primitive.

    Handles numpy scalars and booleans that would otherwise cause json.dumps to raise.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class FAISSVectorStore:
    """Wraps a FAISS IndexFlatL2 with disk persistence and document metadata storage.

    The index and accompanying metadata are saved under *index_dir* so they can be
    rebuilt once and reloaded on every subsequent startup.
    """

    def __init__(self, index_dir: str = "vectorstore/faiss_index") -> None:
        """Initialise store paths; the index is not loaded until build or load is called.

        Args:
            index_dir: Directory where ``index.faiss`` and ``metadata.json`` are written.
        """
        self.index_dir: str = index_dir
        self.index: faiss.Index | None = None
        self.documents: List[Document] = []

    # ── build ─────────────────────────────────────────────────────────────────

    def build_index(self, chunks: List[Document], embedder: Any) -> None:
        """Embed *chunks*, build a flat L2 FAISS index, and persist to disk.

        Args:
            chunks:   Documents whose ``page_content`` will be embedded.
            embedder: Any object exposing ``embed_documents(texts) -> List[List[float]]``.
        """
        logger.info("Building FAISS index for %d chunks", len(chunks))

        texts: List[str] = [doc.page_content for doc in chunks]
        raw_embeddings: List[List[float]] = embedder.embed_documents(texts)

        vectors: np.ndarray = np.array(raw_embeddings, dtype=np.float32)
        dim: int = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.documents = chunks

        # ── persist ──────────────────────────────────────────────────────────
        os.makedirs(self.index_dir, exist_ok=True)

        index_path = os.path.join(self.index_dir, "index.faiss")
        faiss.write_index(self.index, index_path)

        metadata_list = [
            {
                "text": doc.page_content,
                "metadata": {k: _json_safe(v) for k, v in doc.metadata.items()},
            }
            for doc in chunks
        ]
        meta_path = os.path.join(self.index_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata_list, fh, ensure_ascii=False)

        logger.info(
            "FAISS index built: %d vectors, %d dimensions", self.index.ntotal, dim
        )

    # ── load ──────────────────────────────────────────────────────────────────

    def load_index(self, embedder: Any) -> None:  # noqa: ARG002  (embedder reserved for future sub-class compatibility)
        """Load a previously saved FAISS index and its metadata from disk.

        Args:
            embedder: Unused here; kept in the signature for interface consistency
                      (sub-classes may need it to warm the model before searches).

        Raises:
            FileNotFoundError: If the index file does not exist at *index_dir*.
        """
        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run scripts/build_index.py first."
            )

        self.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata_list: List[dict] = json.load(fh)

        self.documents = [
            Document(page_content=item["text"], metadata=item["metadata"])
            for item in metadata_list
        ]
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)

    # ── search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self, query: str, embedder: Any, top_k: int = 5
    ) -> List[dict]:
        """Return the *top_k* most similar documents to *query*.

        Distances from ``IndexFlatL2`` are converted to similarity scores via
        ``score = 1 / (1 + distance)`` so that higher is better.

        Args:
            query:   Natural-language search query.
            embedder: Object with ``embed_query(str) -> List[float]``.
            top_k:   Number of results to return.

        Returns:
            List of dicts with keys ``text``, ``metadata``, and ``score``,
            sorted by score descending.
        """
        query_vec: np.ndarray = np.array(
            [embedder.embed_query(query)], dtype=np.float32
        )
        distances, indices = self.index.search(query_vec, top_k)

        results: List[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            results.append(
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(1.0 / (1.0 + float(dist)), 6),
                }
            )
        return results

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return diagnostic information about the loaded index.

        Returns:
            Dict with ``total_vectors``, ``dimensions``, and ``index_size_mb``,
            or ``{"status": "not loaded"}`` when no index is in memory.
        """
        if self.index is None:
            return {"status": "not loaded"}

        index_path = os.path.join(self.index_dir, "index.faiss")
        size_bytes = (
            os.path.getsize(index_path) if os.path.exists(index_path) else 0
        )
        return {
            "total_vectors": self.index.ntotal,
            "dimensions": self.index.d,
            "index_size_mb": round(size_bytes / (1024 * 1024), 2),
        }
