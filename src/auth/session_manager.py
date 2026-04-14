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
        """Parse a .txt or .json file and add documents from it.

        Args:
            file_path: Path to the file (.txt or .json)

        Returns:
            Tuple of (success, message with document count or error)
        """
        path = Path(file_path)

        if not path.exists():
            return False, f"File not found: {file_path}"

        try:
            if path.suffix == ".txt":
                return self._parse_txt_file(path)
            elif path.suffix == ".json":
                return self._parse_json_file(path)
            else:
                return False, f"Unsupported file format: {path.suffix}. Use .txt or .json"
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
