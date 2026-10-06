import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from config import settings
from models.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    ChromaDB Vector Store Manager for AI-ki-shan.
    Handles embedding persistence and similarity searches.
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.persist_dir = settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_service = get_embedding_service()
        self._client = None
        self._collection = None

    def _init_client(self):
        """Initializes ChromaDB Persistent Client."""
        if self._client is None:
            if chromadb is None:
                raise ImportError("chromadb is not installed. Please run: pip install chromadb")
            
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing ChromaDB PersistentClient at '{self.persist_dir}'")
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    @property
    def collection(self):
        self._init_client()
        return self._collection

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Adds structured chunks with embeddings to ChromaDB.
        Each document should have 'text' and 'metadata'.
        """
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        metadatas = []
        for doc in documents:
            meta = doc.get("metadata", {})
            # Sanitize metadata values for ChromaDB (scalars only)
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self.embedding_service.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully indexed {len(documents)} document chunks into ChromaDB.")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        n_results: int = 4,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches ChromaDB using cosine similarity against query embeddings.
        """
        query_embedding = self.embedding_service.embed_query(query)
        if not query_embedding:
            return []

        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            search_kwargs["where"] = where_filter

        results = self.collection.query(**search_kwargs)

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc_text,
                    "metadata": meta,
                    "score": dist  # In cosine distance: 0 is identical, higher is further
                })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the vector store."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_dir,
                "embedding_model": settings.EMBEDDING_MODEL_NAME
            }
        except Exception as e:
            return {"error": str(e)}


_vector_store_instances: Dict[str, VectorStoreManager] = {}


def get_vector_store(collection_name: Optional[str] = None) -> VectorStoreManager:
    """Singleton getter for the VectorStoreManager per collection."""
    global _vector_store_instances
    name = collection_name or settings.CHROMA_COLLECTION_NAME
    if name not in _vector_store_instances:
        _vector_store_instances[name] = VectorStoreManager(collection_name=name)
    return _vector_store_instances[name]
