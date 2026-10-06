import os
import re
import sys
import glob
import uuid
import logging
import argparse
from pathlib import Path
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional, Tuple

try:
    import chromadb
except ImportError:
    chromadb = None

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


# Optional imports for tokenization and PDF partitioning
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Table, Element
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag.ingest")

# Default Constants
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATA_DIR = "./data"
DEFAULT_CHROMA_DIR = "./data/chroma_db"
DEFAULT_COLLECTION_NAME = "agriculture_knowledge_base"
TARGET_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


# =====================================================================
# 1. HTML Table to Markdown Converter
# =====================================================================

class HTMLTableToMarkdownParser(HTMLParser):
    """
    Custom HTML parser that extracts table rows & cells and converts them
    into clean GitHub-flavored Markdown table strings.
    """
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self.current_row: List[str] = []
        self.current_cell: List[str] = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = []
        elif tag == "tr":
            self.current_row = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.in_cell = False
            cell_text = "".join(self.current_cell).strip().replace("\n", " ").replace("|", "\\|")
            self.current_row.append(cell_text or "-")
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""

        # Normalize column count across all rows
        max_cols = max(len(row) for row in self.rows)
        if max_cols == 0:
            return ""

        padded_rows = [row + ["-"] * (max_cols - len(row)) for row in self.rows]

        # First row is treated as header
        header_row = padded_rows[0]
        md_lines = [
            "| " + " | ".join(header_row) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |"
        ]

        # Body rows
        for row in padded_rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)


def html_table_to_markdown(html_str: str) -> str:
    """Converts an HTML table string into a GitHub-flavored Markdown table."""
    if not html_str or "<table" not in html_str.lower():
        return ""
    parser = HTMLTableToMarkdownParser()
    try:
        parser.feed(html_str)
        return parser.to_markdown()
    except Exception as e:
        logger.warning(f"Failed to parse HTML table: {e}")
        return ""


def text_table_to_markdown(raw_text: str) -> str:
    """Fallback converter that formats tab/multi-space delimited text into Markdown table."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return raw_text

    parsed_rows = []
    for line in lines:
        # Split by tabs or 2+ spaces
        cells = [c.strip() for c in re.split(r"\t+|\s{2,}", line) if c.strip()]
        if len(cells) > 1:
            parsed_rows.append(cells)

    if not parsed_rows:
        return raw_text

    max_cols = max(len(r) for r in parsed_rows)
    header = parsed_rows[0] + ["-"] * (max_cols - len(parsed_rows[0]))
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |"
    ]
    for row in parsed_rows[1:]:
        padded = row + ["-"] * (max_cols - len(row))
        md_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(md_lines)


# =====================================================================
# 2. Tokenizer & 500-Token Chunking Engine
# =====================================================================

class TokenChunker:
    """
    Chunks text into ~500-token blocks using the model's tokenizer
    with configurable token overlap.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, chunk_size: int = TARGET_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = None

        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Could not load AutoTokenizer for '{model_name}': {e}. Using word-level token estimator.")

    def count_tokens(self, text: str) -> int:
        """Estimates or calculates exact token count."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Approximation: 1 token ~= 0.75 words / 4 characters
        return max(1, int(len(text.split()) * 1.3))

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into ~500-token chunks with overlap.
        Preserves paragraph boundaries where possible.
        """
        if not text or not text.strip():
            return []

        # If already within target token limit
        if self.count_tokens(text) <= self.chunk_size:
            return [text.strip()]

        if self.tokenizer:
            # Exact token-level chunking
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            start = 0
            stride = max(1, self.chunk_size - self.overlap)
            while start < len(token_ids):
                end = min(start + self.chunk_size, len(token_ids))
                chunk_token_ids = token_ids[start:end]
                chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if end >= len(token_ids):
                    break
                start += stride
            return chunks
        else:
            # Fallback word-level chunking (~380 words for 500 tokens)
            target_words = int(self.chunk_size / 1.3)
            overlap_words = int(self.overlap / 1.3)
            words = text.split()
            chunks = []
            start = 0
            stride = max(1, target_words - overlap_words)
            while start < len(words):
                end = min(start + target_words, len(words))
                chunk_words = words[start:end]
                chunks.append(" ".join(chunk_words).strip())
                if end >= len(words):
                    break
                start += stride
            return chunks


# =====================================================================
# 3. PDF Ingestion with Unstructured
# =====================================================================

class PDFIngestionPipeline:
    """
    Parses PDFs from /data using 'unstructured', converts tables to Markdown,
    chunks text into 500-token blocks, generates embeddings, and persists to ChromaDB.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        self.model_name = model_name
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.chunker = TokenChunker(model_name=model_name, chunk_size=TARGET_CHUNK_TOKENS, overlap=CHUNK_OVERLAP_TOKENS)
        
        # Load embedding model on available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
        self.embedding_model = SentenceTransformer(model_name, device=device)

        # Initialize ChromaDB client and collection
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to ChromaDB collection '{self.collection_name}' at '{self.chroma_dir}'.")

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parses a single PDF file using unstructured:
        - Preserves and converts tables into Markdown format
        - Chunks narrative text into 500-token blocks
        """
        path = Path(pdf_path)
        logger.info(f"Processing PDF: {path.name}")
        chunks: List[Dict[str, Any]] = []

        if not UNSTRUCTURED_AVAILABLE:
            logger.warning(f"Unstructured library not available. Reading '{path.name}' with fallback text loader.")
            return self._fallback_pdf_read(pdf_path)

        try:
            # Partition PDF into structured elements
            elements = partition_pdf(
                filename=str(path),
                strategy="auto",
                infer_table_structure=True
            )
        except Exception as e:
            logger.warning(f"partition_pdf failed for '{path.name}': {e}. Trying partition()...")
            try:
                elements = partition(filename=str(path))
            except Exception as e2:
                logger.error(f"Failed to partition '{path.name}': {e2}. Using fallback reader.")
                return self._fallback_pdf_read(pdf_path)

        # Buffer for accumulating sequential narrative text before 500-token chunking
        narrative_buffer: List[str] = []
        current_page = 1

        def flush_narrative_buffer():
            """Flushes buffered narrative text into 500-token blocks."""
            if not narrative_buffer:
                return
            full_text = "\n\n".join(narrative_buffer).strip()
            narrative_buffer.clear()
            if not full_text:
                return

            text_chunks = self.chunker.chunk_text(full_text)
            for chunk_idx, text_chunk in enumerate(text_chunks):
                chunks.append({
                    "text": text_chunk,
                    "metadata": {
                        "source": path.name,
                        "file_path": str(path),
                        "is_table": False,
                        "page_number": current_page,
                        "chunk_index": chunk_idx,
                        "type": "narrative_text"
                    }
                })

        for element in elements:
            elem_page = getattr(element.metadata, "page_number", current_page) or current_page
            if elem_page != current_page:
                flush_narrative_buffer()
                current_page = elem_page

            # Check if element is a Table
            is_table_elem = isinstance(element, Table) or getattr(element, "category", "") == "Table"
            
            if is_table_elem:
                # Flush prior text before handling the table
                flush_narrative_buffer()

                # Extract and convert table to Markdown format
                table_md = ""
                html_repr = getattr(element.metadata, "text_as_html", None)
                if html_repr:
                    table_md = html_table_to_markdown(html_repr)

                if not table_md:
                    # Fallback to formatting raw table text
                    table_md = text_table_to_markdown(str(element))

                if table_md.strip():
                    # Format as Markdown Table chunk with metadata
                    formatted_table_chunk = f"### Table from {path.name} (Page {current_page})\n\n{table_md}"
                    chunks.append({
                        "text": formatted_table_chunk,
                        "metadata": {
                            "source": path.name,
                            "file_path": str(path),
                            "is_table": True,
                            "page_number": current_page,
                            "type": "table_markdown"
                        }
                    })
            else:
                elem_text = str(element).strip()
                if elem_text:
                    narrative_buffer.append(elem_text)

        # Flush any remaining text in buffer
        flush_narrative_buffer()

        logger.info(f"Extracted {len(chunks)} chunks ({sum(1 for c in chunks if c['metadata']['is_table'])} tables) from '{path.name}'.")
        return chunks

    def _fallback_pdf_read(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback for extracting text if unstructured partitioning is unavailable."""
        path = Path(pdf_path)
        chunks = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_chunks = self.chunker.chunk_text(page_text)
                    for idx, ch in enumerate(page_chunks):
                        chunks.append({
                            "text": ch,
                            "metadata": {
                                "source": path.name,
                                "file_path": str(path),
                                "is_table": False,
                                "page_number": page_num,
                                "chunk_index": idx,
                                "type": "fallback_pdf_text"
                            }
                        })
        except Exception as e:
            logger.error(f"Fallback PDF reader error for '{path.name}': {e}")
        return chunks

    def ingest_directory(self, data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
        """
        Finds and ingests all PDFs in the data directory into ChromaDB.
        """
        pdf_pattern = os.path.join(data_dir, "**", "*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=True)

        if not pdf_files:
            # Also check flat directory
            pdf_pattern_flat = os.path.join(data_dir, "*.pdf")
            pdf_files = glob.glob(pdf_pattern_flat)

        if not pdf_files:
            logger.warning(f"No PDF files found in '{data_dir}'.")
            return {"total_pdfs": 0, "total_chunks_indexed": 0, "status": "no_files_found"}

        logger.info(f"Found {len(pdf_files)} PDF files to ingest: {[Path(p).name for p in pdf_files]}")

        all_chunks: List[Dict[str, Any]] = []
        for pdf_file in pdf_files:
            pdf_chunks = self.parse_pdf(pdf_file)
            all_chunks.extend(pdf_chunks)

        if not all_chunks:
            logger.warning("No chunks were extracted from PDFs.")
            return {"total_pdfs": len(pdf_files), "total_chunks_indexed": 0, "status": "no_chunks"}

        # Generate embeddings in batches
        texts = [chunk["text"] for chunk in all_chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks using '{self.model_name}'...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        ).tolist()

        # Sanitize metadata values for ChromaDB
        metadatas = []
        for chunk in all_chunks:
            meta = {}
            for k, v in chunk.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        ids = [str(uuid.uuid4()) for _ in all_chunks]

        # Insert into ChromaDB collection in batches to prevent memory spikes
        batch_size = 100
        total_indexed = 0
        for i in range(0, len(all_chunks), batch_size):
            end = i + batch_size
            self.collection.add(
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            total_indexed += len(texts[i:end])

        logger.info(f"Successfully stored {total_indexed} chunks in ChromaDB collection '{self.collection_name}'.")
        return {
            "total_pdfs": len(pdf_files),
            "pdf_files": [Path(p).name for p in pdf_files],
            "total_chunks_indexed": total_indexed,
            "collection_name": self.collection_name,
            "chroma_dir": self.chroma_dir,
            "status": "success"
        }


# =====================================================================
# 4. Top-5 Query Function
# =====================================================================

_cached_models: Dict[str, SentenceTransformer] = {}


def _get_query_embedding_model(model_name: str) -> SentenceTransformer:
    """Retrieves or creates a cached SentenceTransformer model instance."""
    if model_name not in _cached_models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{model_name}' on device '{device}' for retrieval...")
        _cached_models[model_name] = SentenceTransformer(model_name, device=device)
    return _cached_models[model_name]


def query_top_k(
    query_text: str,
    top_k: int = 5,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant results from the persistent ChromaDB collection.
    
    Args:
        query_text: The user's question or search query.
        top_k: Number of top results to retrieve (default: 5).
        chroma_dir: Path to persistent ChromaDB storage.
        collection_name: Name of ChromaDB collection.
        model_name: Hugging Face model ID for embeddings.

    Returns:
        List of dicts containing 'rank', 'text', 'score', 'is_table', and 'metadata'.
    """
    if not query_text or not query_text.strip():
        logger.warning("Empty query provided.")
        return []

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Collection '{collection_name}' not found at '{chroma_dir}': {e}")
        return []

    # Generate query embedding using cached model
    embed_model = _get_query_embedding_model(model_name)
    query_embedding = embed_model.encode(
        query_text,
        convert_to_tensor=False,
        show_progress_bar=False,
        normalize_embeddings=True
    ).tolist()


    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    top_results = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            # Cosine distance: 0 is exact match, 1 is orthogonal
            # Similarity score: 1 - distance
            similarity = round(1.0 - float(dist), 4)
            is_table = meta.get("is_table", False)
            source = meta.get("source", "Unknown PDF")
            page = meta.get("page_number", "-")

            top_results.append({
                "rank": rank,
                "text": doc,
                "similarity_score": similarity,
                "distance": round(float(dist), 4),
                "is_table": is_table,
                "source": source,
                "page_number": page,
                "metadata": meta
            })

    return top_results


def print_query_results(query: str, results: List[Dict[str, Any]]):
    """Pretty prints query results to terminal."""
    print("\n" + "=" * 80)
    print(f"🔍 Top {len(results)} Results for Query: '{query}'")
    print("=" * 80)

    for res in results:
        table_badge = " [📊 TABLE]" if res["is_table"] else " [📄 TEXT]"
        print(f"\n#{res['rank']}{table_badge} Source: {res['source']} (Page {res['page_number']}) | Similarity: {res['similarity_score']:.2%}")
        print("-" * 80)
        # Indent content slightly
        lines = res["text"].splitlines()
        for line in lines:
            print(f"  {line}")
    print("\n" + "=" * 80 + "\n")


# =====================================================================
# 5. CLI Entrypoint
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI-ki-shan PDF & Table Ingestion and Top-5 Query Tool"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest all PDFs from the data directory into ChromaDB"
    )
    parser.add_argument(
        "--finance",
        action="store_true",
        help="Ingest into the finance knowledge base instead of the agriculture knowledge base"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing PDF files (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a query to retrieve the top 5 most relevant results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=DEFAULT_CHROMA_DIR,
        help=f"ChromaDB persistent directory (default: {DEFAULT_CHROMA_DIR})"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME})"
    )

    args = parser.parse_args()

    # Determine target collection based on --finance flag
    from config import settings
    target_collection = settings.CHROMA_FINANCE_COLLECTION_NAME if args.finance else args.collection

    # If --ingest or no specific query supplied, run ingestion
    if args.ingest or (not args.query and len(sys.argv) == 1):
        pipeline = PDFIngestionPipeline(
            chroma_dir=args.chroma_dir,
            collection_name=target_collection
        )
        result = pipeline.ingest_directory(data_dir=args.data_dir)
        print("\n" + "=" * 50)
        print("📁 Ingestion Summary:")
        print(f"  • Total PDFs Processed: {result.get('total_pdfs', 0)}")
        print(f"  • Total Chunks Indexed: {result.get('total_chunks_indexed', 0)}")
        print(f"  • Collection: {result.get('collection_name')}")
        print(f"  • Storage Directory: {result.get('chroma_dir')}")
        print("=" * 50 + "\n")

    # If query supplied (or test query on default run)
    query_text = args.query or "What are the eligibility criteria and financial benefits under PM-Kisan scheme?"
    results = query_top_k(
        query_text=query_text,
        top_k=args.top_k,
        chroma_dir=args.chroma_dir,
        collection_name=target_collection
    )
    print_query_results(query_text, results)


if __name__ == "__main__":
    main()
