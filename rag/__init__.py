"""
RAG (Retrieval-Augmented Generation) package for AI-ki-shan.
Includes document loaders (unstructured for tables/docs), ChromaDB vector store, and retrieval pipelines.
"""
from rag.document_loader import DocumentLoader
from rag.vector_store import VectorStoreManager, get_vector_store
from rag.retriever import RAGRetriever
from rag.pipeline import RAGPipeline, get_rag_pipeline
from rag.ingest import PDFIngestionPipeline, query_top_k

__all__ = [
    "DocumentLoader",
    "VectorStoreManager",
    "get_vector_store",
    "RAGRetriever",
    "RAGPipeline",
    "get_rag_pipeline",
    "PDFIngestionPipeline",
    "query_top_k",
]

