import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from rag.document_loader import DocumentLoader
from rag.vector_store import get_vector_store
from rag.retriever import RAGRetriever
from models.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-End RAG Pipeline for AI-ki-shan.
    Integrates Document Ingestion, Embedding Indexing, Semantic Retrieval,
    and LLM Advisory Generation.
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name
        self.loader = DocumentLoader()
        self.vector_store = get_vector_store(collection_name)
        self.retriever = RAGRetriever(collection_name)
        self.llm_client = get_llm_client()

    def ingest_file(self, file_path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingests a document (PDF, TXT, MD, JSON, CSV with tables) into ChromaDB.
        """
        logger.info(f"Ingesting file: {file_path}")
        chunks = self.loader.load_file(file_path, extra_metadata)
        count = self.vector_store.add_documents(chunks)
        return {
            "file": file_path,
            "chunks_created": len(chunks),
            "chunks_indexed": count,
            "status": "success"
        }

    async def query(
        self,
        question: str,
        category: Optional[str] = None,
        crop: Optional[str] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Executes an agricultural inquiry through the RAG pipeline.
        """
        # 1. Retrieve relevant verified context
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            category=category,
            crop=crop
        )

        # 2. Generate response via LLM Client
        response_text = await self.llm_client.generate_response(
            query=question,
            context_chunks=retrieved_chunks,
            language=language
        )

        # 3. Format sources metadata
        sources = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            sources.append({
                "title": meta.get("title", meta.get("source", "Knowledge Base")),
                "category": meta.get("category", "Agriculture"),
                "crop": meta.get("crop", "General"),
                "score": chunk.get("score")
            })

        return {
            "query": question,
            "answer": response_text,
            "retrieved_sources_count": len(retrieved_chunks),
            "sources": sources
        }

    def seed_initial_knowledge(self, sample_data_path: str = "./data/sample_crop_guide.json"):
        """Seeds sample agriculture data if the vector store is currently empty."""
        try:
            stats = self.vector_store.get_stats()
            if stats.get("document_count", 0) == 0 and os.path.exists(sample_data_path):
                logger.info(f"Seeding initial agriculture knowledge from {sample_data_path}...")
                self.ingest_file(sample_data_path, extra_metadata={"source": "National Agriculture Handbook"})
                logger.info("Default agriculture knowledge successfully seeded.")
        except Exception as e:
            logger.warning(f"Could not seed initial knowledge: {e}")


_rag_pipeline_instances: Dict[str, RAGPipeline] = {}


def get_rag_pipeline(collection_name: Optional[str] = None) -> RAGPipeline:
    """Singleton getter for the RAGPipeline per collection."""
    global _rag_pipeline_instances
    from config import settings
    name = collection_name or settings.CHROMA_COLLECTION_NAME
    if name not in _rag_pipeline_instances:
        _rag_pipeline_instances[name] = RAGPipeline(collection_name=name)
    return _rag_pipeline_instances[name]
