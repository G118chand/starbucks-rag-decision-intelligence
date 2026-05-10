"""Hybrid retriever combining dense FAISS semantic search with sparse BM25 keyword search."""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses FAISS vector search (semantic) and BM25 (keyword) for robust retrieval.

    The default weighting is 70 % semantic + 30 % keyword, which generally favours
    meaning over exact term overlap while still surfacing documents that share
    important domain vocabulary with the query.
    """

    def __init__(
        self,
        faiss_store: Any,
        embedder: Any,
        chunks: List[Document],
    ) -> None:
        """Build the BM25 corpus index over *chunks*.

        Args:
            faiss_store: A ``FAISSVectorStore`` instance (or any object with
                         ``similarity_search(query, embedder, top_k)``).
            embedder:    Embedding engine forwarded to FAISS searches.
            chunks:      The same chunk list used to build the FAISS index.
        """
        self.faiss_store = faiss_store
        self.embedder = embedder
        self.chunks = chunks
        self.chunk_texts: List[str] = [doc.page_content for doc in chunks]

        tokenized_corpus: List[List[str]] = [
            text.lower().split() for text in self.chunk_texts
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # fast O(1) text → metadata lookup; duplicate texts resolve to last writer
        self._text_to_metadata: Dict[str, dict] = {
            doc.page_content: doc.metadata for doc in chunks
        }

        logger.info("HybridRetriever initialized: %d chunks", len(chunks))

    # ── private helpers ───────────────────────────────────────────────────────

    def _normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalise *scores* to the [0, 1] range.

        Args:
            scores: Raw scores (any range).

        Returns:
            Normalised scores.  Returns all-ones when every score is identical.
        """
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        span = max_s - min_s
        return [(s - min_s) / span for s in scores]

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """Retrieve the *top_k* most relevant chunks for *query*.

        Pipeline:
        1. FAISS similarity search → semantic scores (``top_k × 2`` candidates).
        2. BM25 scores over the full corpus, normalised to [0, 1], top ``top_k × 2`` kept.
        3. Union of both candidate sets; each text scored as
           ``0.7 × semantic + 0.3 × keyword``.
        4. Sorted descending, top *top_k* returned.

        Args:
            query:  Natural-language query string.
            top_k:  Number of results to return.

        Returns:
            List of dicts with keys ``text``, ``metadata``, ``semantic_score``,
            ``keyword_score``, and ``combined_score``, sorted by ``combined_score``.
        """
        candidate_pool = top_k * 2

        # ── semantic leg ─────────────────────────────────────────────────────
        semantic_results = self.faiss_store.similarity_search(
            query, self.embedder, top_k=candidate_pool
        )
        semantic_map: Dict[str, float] = {
            r["text"]: r["score"] for r in semantic_results
        }

        # ── keyword leg ──────────────────────────────────────────────────────
        tokenized_query: List[str] = query.lower().split()
        bm25_raw: List[float] = self.bm25.get_scores(tokenized_query).tolist()
        bm25_normalised: List[float] = self._normalize(bm25_raw)

        # keep only the top candidate_pool BM25 results to bound union size
        top_bm25_indices = sorted(
            range(len(bm25_normalised)),
            key=lambda i: bm25_normalised[i],
            reverse=True,
        )[:candidate_pool]
        bm25_map: Dict[str, float] = {
            self.chunk_texts[i]: bm25_normalised[i] for i in top_bm25_indices
        }

        # ── fusion ───────────────────────────────────────────────────────────
        all_texts = set(semantic_map.keys()) | set(bm25_map.keys())

        fused: List[dict] = []
        for text in all_texts:
            sem_score = semantic_map.get(text, 0.0)
            kw_score = bm25_map.get(text, 0.0)
            combined = 0.7 * sem_score + 0.3 * kw_score
            fused.append(
                {
                    "text": text,
                    "metadata": self._text_to_metadata.get(text, {}),
                    "semantic_score": round(sem_score, 4),
                    "keyword_score": round(kw_score, 4),
                    "combined_score": round(combined, 4),
                }
            )

        fused.sort(key=lambda x: x["combined_score"], reverse=True)
        top_results = fused[:top_k]

        if top_results:
            logger.info(
                "Query: '%s' | top combined score: %.4f",
                query,
                top_results[0]["combined_score"],
            )

        return top_results
