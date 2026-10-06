import logging
from typing import List, Dict, Any, Optional
from config import settings
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Generic Retriever for RAG.
    Filters and ranks knowledge context using cosine similarity.
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.vector_store = get_vector_store(collection_name)
        self.max_docs = settings.MAX_RETRIEVED_DOCS
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        crop: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant agricultural context for a given query.
        """
        k = top_k or self.max_docs
        where_filter = {}

        if category:
            where_filter["category"] = category
        if crop:
            where_filter["crop"] = crop

        filter_arg = where_filter if where_filter else None
        results = self.vector_store.similarity_search(query=query, n_results=k, where_filter=filter_arg)

        if not results:
            return []

        # Cosine distance: 0.0 is identical, 1.0 is orthogonal.
        # Allow distance up to max_distance (e.g., 0.85 for 0.65 confidence threshold)
        max_allowed_dist = 1.0 - (self.confidence_threshold - 0.5)
        filtered_results = [res for res in results if res.get("score", 1.0) <= max_allowed_dist]

        # If strict filtering eliminated all results, fallback to returning the top closest candidate
        if not filtered_results and results:
            filtered_results = [results[0]]

        return filtered_results
