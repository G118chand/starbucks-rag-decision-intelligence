"""Embedding engine wrapping SentenceTransformer with process-level model caching."""

import logging
from typing import ClassVar, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates dense vector embeddings using a local SentenceTransformer model.

    The underlying model is loaded once per process and shared across all instances
    via a class-level cache, avoiding redundant disk I/O and GPU/CPU initialisation.
    """

    _model_instance: ClassVar[Optional[SentenceTransformer]] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load or reuse the SentenceTransformer model.

        Args:
            model_name: Hugging Face model identifier or local path.
                        Defaults to ``all-MiniLM-L6-v2`` (384-dimensional, no API key needed).
        """
        self.model_name = model_name

        if EmbeddingEngine._model_instance is None:
            logger.info("Loading embedding model: %s", model_name)
            try:
                EmbeddingEngine._model_instance = SentenceTransformer(model_name)
            except Exception as exc:
                logger.error("Failed to load embedding model %s: %s", model_name, exc)
                raise
        else:
            logger.info("Using cached embedding model: %s", model_name)

        self._model: SentenceTransformer = EmbeddingEngine._model_instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
            Returns an empty list when ``texts`` is empty.
        """
        if not texts:
            return []

        try:
            embeddings: np.ndarray = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Failed to embed %d documents: %s", len(texts), exc)
            raise

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Args:
            query: The search query to embed.

        Returns:
            Float vector representation of the query.
        """
        try:
            embedding: np.ndarray = self._model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding.tolist()
        except Exception as exc:
            logger.error("Failed to embed query: %s", exc)
            raise

    def get_model_info(self) -> dict:
        """Return metadata about the loaded embedding model.

        Returns:
            Dictionary with ``model_name`` and ``dimensions``.
        """
        return {
            "model_name": self.model_name,
            "dimensions": self._model.get_sentence_embedding_dimension(),
        }
