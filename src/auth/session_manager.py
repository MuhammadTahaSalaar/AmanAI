"""Session-level document manager for runtime document uploads."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data_processing.base_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SessionDocumentManager:
    """Manages documents uploaded during the session (stored in memory only)."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        logger.info("SessionDocumentManager initialized")

    def add_document(self, content: str, source: str = "uploaded", 
                     metadata_overrides: dict | None = None) -> tuple[bool, str]:
        """Add a document from uploaded content.

        Args:
            content: The document text content
            source: The source identifier (default: 'uploaded')
            metadata_overrides: Optional dict to override/extend default metadata

        Returns:
            Tuple of (success, message)
        """
        if not content or not content.strip():
            return False, "Document content is empty"

        try:
            # Create a Document object
            metadata = {
                "source": source,
                "type": "session_upload",
                "product": "User Uploaded Document",
            }
            # Override with any provided metadata
            if metadata_overrides:
                metadata.update(metadata_overrides)
            
            doc = Document(
                content=content.strip(),
                metadata=metadata,
            )
            self._documents.append(doc)
            logger.info("Added session document from %s", source)
            return True, f"Document added successfully ({len(content)} characters)"
        except Exception as e:
            logger.error("Failed to add session document: %s", str(e))
            return False, f"Error adding document: {str(e)}"

    def parse_and_add_file(self, file_path: str) -> tuple[bool, str]:
        """Parse a file and add documents from it.

        Args:
            file_path: Path to the file (.txt, .json, .pdf, .xlsx, .csv)

        Returns:
            Tuple of (success, message with document count or error)
        """
        path = Path(file_path)

        if not path.exists():
            return False, f"File not found: {file_path}"

        try:
            suffix = path.suffix.lower()
            if suffix == ".txt":
                return self._parse_txt_file(path)
            elif suffix == ".json":
                return self._parse_json_file(path)
            elif suffix == ".pdf":
                return self._parse_pdf_file(path)
            elif suffix == ".xlsx":
                return self._parse_excel_file(path)
            elif suffix == ".csv":
                return self._parse_csv_file(path)
            else:
                return False, f"Unsupported file format: {path.suffix}. Use .json, .txt, .pdf, .xlsx, or .csv"
        except Exception as e:
            logger.error("Error parsing file %s: %s", file_path, str(e))
            return False, f"Error parsing file: {str(e)}"

    def _parse_txt_file(self, path: Path) -> tuple[bool, str]:
        """Parse a .txt file and add its content as a document."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        success, msg = self.add_document(content, source=path.name)
        if success:
            return True, f"Added TXT document: {path.name}"
        return False, msg

    def _parse_json_file(self, path: Path) -> tuple[bool, str]:
        """Parse a .json file and extract documents.

        Expects either:
        - Direct text content: {"content": "text"} or {"text": "text"}
        - FAQ format: {"categories": [...]}
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs_added = 0

        # Handle direct text content
        if isinstance(data, dict):
            if "content" in data:
                success, _ = self.add_document(data["content"], source=path.name)
                if success:
                    docs_added += 1
            elif "text" in data:
                success, _ = self.add_document(data["text"], source=path.name)
                if success:
                    docs_added += 1
            elif "categories" in data:
                # Handle FAQ format with categories
                docs_added += self._extract_faq_docs(data, path.name)

        if docs_added == 0:
            return False, "No extractable content found in JSON file"

        return True, f"Added {docs_added} document(s) from JSON file: {path.name}"

    def _extract_faq_docs(self, faq_data: dict, source: str) -> int:
        """Extract documents from FAQ-format JSON."""
        count = 0
        categories = faq_data.get("categories", [])

        for category in categories:
            category_name = category.get("category", "FAQ")
            questions = category.get("questions", [])

            for qa in questions:
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if question and answer:
                    content = f"Q: {question}\nA: {answer}"
                    
                    # Extract product name from question if available
                    import re
                    product_match = re.search(
                        r"(NUST\s+[\w\s]+(?:Account|Finance|Deposit|Card|Remittance))",
                        question,
                        re.IGNORECASE
                    )
                    product_name = product_match.group(1) if product_match else category_name
                    
                    success, _ = self.add_document(
                        content, 
                        source=f"{source}/{category_name}",
                        metadata_overrides={"product": product_name}
                    )
                    if success:
                        count += 1

        return count

    def get_documents(self) -> list[Document]:
        """Get all session-level documents.

        Returns:
            List of Document objects added in this session
        """
        return self._documents

    def _parse_pdf_file(self, path: Path) -> tuple[bool, str]:
        """Parse a PDF file and add extracted text as documents."""
        from src.data_processing.pdf_processor import PDFProcessor

        processor = PDFProcessor(file_path=path)
        docs = processor.process()
        if not docs:
            return False, "No extractable text found in PDF"

        for doc in docs:
            self._documents.append(doc)

        logger.info("Added %d document(s) from PDF: %s", len(docs), path.name)
        return True, f"Added {len(docs)} document(s) from PDF: {path.name}"

    def _parse_excel_file(self, path: Path) -> tuple[bool, str]:
        """Parse an Excel (.xlsx) file — each row becomes a document."""
        import pandas as pd

        try:
            xls = pd.ExcelFile(path)
        except Exception as e:
            return False, f"Cannot open Excel file: {e}"

        total = 0
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            if df.empty:
                continue
            for _, row in df.iterrows():
                parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                if not parts:
                    continue
                content = "\n".join(parts)
                self._documents.append(
                    Document(
                        content=content,
                        metadata={
                            "product": sheet,
                            "source": path.name,
                            "type": "excel_upload",
                        },
                    )
                )
                total += 1

        if total == 0:
            return False, "No data rows found in Excel file"
        logger.info("Added %d document(s) from Excel: %s", total, path.name)
        return True, f"Added {total} document(s) from Excel: {path.name}"

    def _parse_csv_file(self, path: Path) -> tuple[bool, str]:
        """Parse a CSV file — each row becomes a document."""
        from src.data_processing.csv_processor import CSVProcessor

        processor = CSVProcessor(file_path=path)
        docs = processor.process()
        if not docs:
            return False, "No data rows found in CSV file"

        for doc in docs:
            self._documents.append(doc)

        logger.info("Added %d document(s) from CSV: %s", len(docs), path.name)
        return True, f"Added {len(docs)} document(s) from CSV: {path.name}"

    def clear_documents(self) -> None:
        """Clear all session documents.

        Note: This should only be called on logout or session reset.
        """
        self._documents.clear()
        logger.info("Session documents cleared")

    def get_document_count(self) -> int:
        """Get the count of session documents.

        Returns:
            Number of documents in the session
        """
        return len(self._documents)
