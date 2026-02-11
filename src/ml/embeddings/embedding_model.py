"""
Embedding model wrapper for generating text embeddings.

Uses sentence-transformers library for generating high-quality
semantic embeddings from text content.
"""

from typing import Optional

import numpy as np

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding models.

    Provides a unified interface for generating text embeddings
    with support for batching and device selection.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config setting.
            device: Device to run model on ('cpu', 'cuda', 'mps').
                   Defaults to config setting.
        """
        settings = get_settings()
        self.model_name = model_name or settings.ml.embedding_model
        self.device = device or settings.ml.device
        self.dimension = settings.ml.embedding_dimension
        self.batch_size = settings.ml.batch_size

        self._model = None
        self._initialized = False

    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._initialized = True
            logger.info(f"Embedding model loaded on device: {self.device}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def model(self):
        """Get the underlying sentence-transformer model."""
        if not self._initialized:
            self._load_model()
        return self._model

    def encode(
        self,
        texts: str | list[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text string or list of texts to encode.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).
            show_progress: Whether to show progress bar for large batches.

        Returns:
            numpy array of shape (n_texts, embedding_dim) or (embedding_dim,)
            for single text input.
        """
        if not self._initialized:
            self._load_model()

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        if single_input:
            return embeddings[0]

        return embeddings

    def encode_resume(self, resume_text: str) -> np.ndarray:
        """
        Generate embedding for resume content.

        Args:
            resume_text: Full resume text content.

        Returns:
            Normalized embedding vector.
        """
        return self.encode(resume_text, normalize=True)

    def encode_job_description(self, jd_text: str) -> np.ndarray:
        """
        Generate embedding for job description.

        Args:
            jd_text: Full job description text.

        Returns:
            Normalized embedding vector.
        """
        return self.encode(jd_text, normalize=True)

    def encode_skills(self, skills: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of skills.

        Args:
            skills: List of skill names/descriptions.

        Returns:
            Array of embedding vectors, one per skill.
        """
        if not skills:
            return np.array([])
        return self.encode(skills, normalize=True)

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        # For normalized vectors, cosine similarity = dot product
        sim = np.dot(embedding1, embedding2)
        # Clip to valid range (numerical precision issues)
        return float(np.clip(sim, 0.0, 1.0))

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate similarity between a query and multiple corpus embeddings.

        Args:
            query_embedding: Single query embedding vector.
            corpus_embeddings: Array of corpus embedding vectors.

        Returns:
            Array of similarity scores.
        """
        if len(corpus_embeddings) == 0:
            return np.array([])

        similarities = np.dot(corpus_embeddings, query_embedding)
        return np.clip(similarities, 0.0, 1.0)


# Singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get the embedding model singleton instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
