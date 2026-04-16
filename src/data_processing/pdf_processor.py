"""PDF document processor using PyMuPDF (fitz)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from src.data_processing.base_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFProcessor:
    """Extracts text from PDF files and converts to Document objects.

    Handles both structured product-sheet PDFs (with key-value pairs) and
    freeform text PDFs.  Each page becomes one Document, unless the page
    is very short — in which case consecutive short pages are merged.
    """

    _MIN_CONTENT_LEN = 30  # skip near-empty pages

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)

    def process(self) -> list[Document]:
        """Extract text from every page and return Document objects.

        Returns:
            List of Document objects, roughly one per page.
        """
        if not self.file_path.exists():
            logger.error("PDF file not found: %s", self.file_path)
            return []

        try:
            doc = fitz.open(str(self.file_path))
        except Exception as e:
            logger.error("Failed to open PDF %s: %s", self.file_path, e)
            return []

        pages: list[str] = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
        doc.close()

        if not pages:
            logger.warning("No text extracted from PDF: %s", self.file_path)
            return []

        # Attempt to detect a product name from the first page
        product_name = self._detect_product_name(pages[0])

        # Merge all pages into a single content block if total is short,
        # otherwise keep per-page chunking.
        full_text = "\n\n".join(pages)
        documents: list[Document] = []

        if len(full_text) <= 2000:
            # Treat as a single document
            documents.append(
                Document(
                    content=self._format_content(full_text, product_name),
                    metadata={
                        "product": product_name,
                        "source": self.file_path.name,
                        "type": "pdf_upload",
                    },
                )
            )
        else:
            # Chunk per page, merging very short pages with the next
            buffer = ""
            for i, page_text in enumerate(pages):
                buffer = f"{buffer}\n\n{page_text}".strip() if buffer else page_text
                if len(buffer) >= self._MIN_CONTENT_LEN and (
                    len(buffer) >= 300 or i == len(pages) - 1
                ):
                    documents.append(
                        Document(
                            content=self._format_content(buffer, product_name),
                            metadata={
                                "product": product_name,
                                "source": self.file_path.name,
                                "type": "pdf_upload",
                                "page": i + 1,
                            },
                        )
                    )
                    buffer = ""
            # Flush remaining buffer
            if buffer and len(buffer) >= self._MIN_CONTENT_LEN:
                documents.append(
                    Document(
                        content=self._format_content(buffer, product_name),
                        metadata={
                            "product": product_name,
                            "source": self.file_path.name,
                            "type": "pdf_upload",
                        },
                    )
                )

        logger.info(
            "PDF processed: %s → %d document(s), product='%s'",
            self.file_path.name,
            len(documents),
            product_name,
        )
        return documents

    @staticmethod
    def _detect_product_name(first_page: str) -> str:
        """Try to extract a product name from the first page.

        Looks for common patterns:
        - "Product Name\\nSOMETHING"
        - Lines containing "NUST ..."
        """
        # Pattern: "Product Name" followed by value on next line
        m = re.search(r"Product\s+Name\s*\n+\s*(.+)", first_page, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Fallback: first line containing "NUST"
        for line in first_page.split("\n"):
            line = line.strip()
            if "NUST" in line and len(line) < 80:
                return line
        return "Uploaded PDF Document"

    @staticmethod
    def _format_content(text: str, product_name: str) -> str:
        """Prefix content with product name for retrieval context."""
        if product_name and product_name != "Uploaded PDF Document":
            return f"[{product_name}]\n{text}"
        return text
