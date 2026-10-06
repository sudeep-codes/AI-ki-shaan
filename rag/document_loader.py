import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# Try to import unstructured for table and document parsing
try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Table, Element
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("unstructured library not installed or partially missing. Fallbacks will be used.")


class _HTMLTableParser(HTMLParser):
    """Parses HTML tables into markdown table strings."""
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
        max_cols = max(len(row) for row in self.rows)
        if max_cols == 0:
            return ""
        padded_rows = [row + ["-"] * (max_cols - len(row)) for row in self.rows]
        header_row = padded_rows[0]
        md_lines = [
            "| " + " | ".join(header_row) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |"
        ]
        for row in padded_rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(md_lines)


def html_table_to_markdown(html_str: str) -> str:
    """Converts HTML table string to markdown table."""
    if not html_str or "<table" not in html_str.lower():
        return ""
    parser = _HTMLTableParser()
    try:
        parser.feed(html_str)
        return parser.to_markdown()
    except Exception as e:
        logger.warning(f"Failed to parse HTML table to markdown: {e}")
        return ""


class DocumentLoader:
    """
    Agricultural Document Loader using 'unstructured' for advanced table & document parsing.
    Supports CSV, JSON, PDF, TXT, and Markdown files.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Parses a document file and returns structured chunks with metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        metadata = extra_metadata or {}
        metadata.setdefault("source", path.name)
        metadata.setdefault("file_type", ext)

        if ext == ".json":
            return self._load_json(file_path, metadata)
        elif ext in [".txt", ".md"]:
            return self._load_text(file_path, metadata)
        elif UNSTRUCTURED_AVAILABLE:
            return self._load_with_unstructured(file_path, metadata)
        else:
            return self._load_text(file_path, metadata)

    def _load_with_unstructured(self, file_path: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Uses unstructured to parse tables and documents into semantic elements.
        """
        chunks: List[Dict[str, Any]] = []
        try:
            elements: List[Element] = partition(filename=file_path)
            for idx, element in enumerate(elements):
                raw_text = str(element).strip()
                if not raw_text:
                    continue

                is_table = isinstance(element, Table) or getattr(element, "category", "") == "Table"
                elem_meta = dict(base_metadata)
                elem_meta.update({
                    "element_id": f"{Path(file_path).stem}_{idx}",
                    "is_table": is_table,
                    "element_type": element.__class__.__name__
                })

                chunk_text = raw_text
                # Convert table HTML to Markdown if available
                if is_table:
                    html_content = getattr(element.metadata, "text_as_html", None)
                    if html_content:
                        table_md = html_table_to_markdown(html_content)
                        if table_md:
                            chunk_text = f"### Table from {Path(file_path).name}\n\n{table_md}"
                            elem_meta["table_html"] = html_content

                chunks.append({
                    "text": chunk_text,
                    "metadata": elem_meta
                })
        except Exception as e:
            logger.error(f"Error parsing with unstructured: {e}. Falling back to standard text loader.")
            return self._load_text(file_path, base_metadata)

        return chunks

    def _load_json(self, file_path: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Loads JSON agricultural knowledge files (lists of articles/crop guides)."""
        chunks: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    title = item.get("title") or item.get("crop") or item.get("scheme_name", f"Item {idx+1}")
                    content = item.get("content") or item.get("description") or item.get("details", "")
                    
                    # Merge item keys into metadata
                    item_meta = dict(base_metadata)
                    item_meta.update({
                        "title": str(title),
                        "category": item.get("category", "General Agriculture"),
                        "crop": item.get("crop", "All"),
                        "source_index": idx
                    })

                    formatted_text = f"Title: {title}\nCategory: {item_meta['category']}\nDetails:\n{content}"
                    if "recommendations" in item:
                        formatted_text += f"\nRecommendations: {item['recommendations']}"
                    if "dosage" in item:
                        formatted_text += f"\nDosage / Application: {item['dosage']}"
                    if "eligibility" in item:
                        formatted_text += f"\nEligibility: {item['eligibility']}"

                    chunks.append({
                        "text": formatted_text.strip(),
                        "metadata": item_meta
                    })
        elif isinstance(data, dict):
            chunks.append({
                "text": json.dumps(data, indent=2),
                "metadata": base_metadata
            })

        return chunks

    def _load_text(self, file_path: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Loads and splits plain text or markdown files."""
        chunks: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split into manageable chunks by paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for idx, para in enumerate(paragraphs):
            meta = dict(base_metadata)
            meta["chunk_id"] = idx
            chunks.append({
                "text": para,
                "metadata": meta
            })

        return chunks
