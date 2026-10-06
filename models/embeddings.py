import logging
from typing import List, Optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service optimized for Hugging Face Spaces (16GB RAM environment).
    Loads SentenceTransformer models lazily with CPU/GPU memory optimizations.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self._model: Optional[SentenceTransformer] = None
        
        # Optimize CPU threads for 16GB RAM Hugging Face CPU Spaces
        if self.device == "cpu" and torch is not None:
            # Avoid thread over-subscription on shared virtual cores
            torch.set_num_threads(max(1, torch.get_num_threads() // 2))


    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model upon first request."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. Please run: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model '{self.model_name}' successfully loaded.")
        return self._model

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query."""
        if not text.strip():
            return []
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embedding vectors for a list of document chunks."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return [emb.tolist() for emb in embeddings]


_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Singleton getter for the EmbeddingService."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
