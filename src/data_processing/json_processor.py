"""JSON FAQ processor for the funds transfer app features FAQ file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data_processing.base_processor import BaseProcessor, Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JSONProcessor(BaseProcessor):
    """Processes the funds_transfer_app_features_faq.json file.

    Parses the structured JSON with categories and Q&A pairs into
    Document objects.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    def process(self) -> list[Document]:
        """Parse the JSON FAQ file and return Document objects.

        Returns:
            List of Document objects, one per Q&A pair.
        """
        with open(self._file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: list[Document] = []

        for category_block in data.get("categories", []):
            category = category_block.get("category", "Unknown")

            for qa in category_block.get("questions", []):
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()

                if question and answer:
                    content = f"Q: {question}\nA: {answer}"
                    documents.append(
                        Document(
                            content=content,
                            metadata={
                                "category": category,
                                "source": "funds_transfer_app_features_faq.json",
                                "type": "qa",
                            },
                        )
                    )

        logger.info(
            "JSON FAQ processed: %d documents from '%s'",
            len(documents),
            self._file_path.name,
        )
        return documents
