"""ETL Pipeline orchestrator.

Coordinates all data processors, runs the full extract-transform-load
pipeline, and saves processed data for inspection and reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import openpyxl

import config
from src.data_processing.base_processor import Document
from src.data_processing.rate_sheet_processor import RateSheetProcessor
from src.data_processing.faq_sheet_processor import FAQSheetProcessor
from src.data_processing.json_processor import JSONProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ETLPipeline:
    """Orchestrates the full ETL process for NUST Bank datasets.

    Runs all processors, merges results, and saves processed data.
    """

    def __init__(
        self,
        excel_path: Path | None = None,
        json_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._excel_path = excel_path or config.EXCEL_FILE
        self._json_path = json_path or config.FAQ_JSON_FILE
        self._output_dir = output_dir or config.PROCESSED_DATA_DIR

    def run(self) -> list[Document]:
        """Execute the full ETL pipeline.

        Returns:
            Merged list of all processed Document objects.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        all_documents: list[Document] = []

        # 1. Rate Sheet
        rate_docs = self._process_rate_sheet()
        all_documents.extend(rate_docs)
        self._save(rate_docs, "rate_sheet_documents.json")

        # 2. Product FAQ Sheets
        faq_docs = self._process_faq_sheets()
        all_documents.extend(faq_docs)
        self._save(faq_docs, "product_faq_documents.json")

        # 3. JSON FAQs
        json_docs = self._process_json_faq()
        all_documents.extend(json_docs)
        self._save(json_docs, "app_faq_documents.json")

        # NOTE: Runtime documents (data/runtime_document/) are NOT loaded at
        # launch.  They are meant to be uploaded per-session via the admin
        # document-upload interface, so they only affect the uploading user's
        # session and never pollute the base knowledge base.

        # 4. Merged output
        self._save(all_documents, "all_documents.json")

        logger.info(
            "ETL complete: %d total documents (rate=%d, faq=%d, json=%d)",
            len(all_documents),
            len(rate_docs),
            len(faq_docs),
            len(json_docs),
        )
        return all_documents

    def _process_rate_sheet(self) -> list[Document]:
        """Run the rate sheet processor."""
        processor = RateSheetProcessor(
            file_path=self._excel_path,
            sheet_name=config.RATE_SHEET_NAME,
            effective_date=config.RATE_SHEET_EFFECTIVE_DATE,
        )
        return processor.process()

    def _process_faq_sheets(self) -> list[Document]:
        """Run the FAQ sheet processor for all relevant sheets."""
        wb = openpyxl.load_workbook(self._excel_path, data_only=True)
        sheet_names = wb.sheetnames
        wb.close()

        processor = FAQSheetProcessor(
            file_path=self._excel_path,
            sheet_names=sheet_names,
            skip_sheets=config.SKIP_SHEETS,
            rate_sheet_name=config.RATE_SHEET_NAME,
        )
        return processor.process()

    def _process_json_faq(self) -> list[Document]:
        """Run the JSON FAQ processor."""
        if not self._json_path.exists():
            logger.warning("JSON FAQ file not found: %s", self._json_path)
            return []
        processor = JSONProcessor(file_path=self._json_path)
        return processor.process()

    def _process_runtime_documents(self) -> list[Document]:
        """Process runtime documents from data/runtime_document directory.
        
        Runtime documents are special offers, dynamic content, and other
        documents that are loaded at runtime but should be part of the
        knowledge base.
        
        Returns:
            List of Document objects from runtime documents.
        """
        documents: list[Document] = []
        runtime_dir = Path("data/runtime_document")
        
        if not runtime_dir.exists():
            logger.info("Runtime document directory not found: %s", runtime_dir)
            return documents
        
        # Process all JSON files in the runtime_document directory
        for json_file in runtime_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle FAQ-format JSON with categories
                if isinstance(data, dict) and "categories" in data:
                    for category in data.get("categories", []):
                        category_name = category.get("category", "Runtime Document")
                        for qa in category.get("questions", []):
                            question = qa.get("question", "").strip()
                            answer = qa.get("answer", "").strip()
                            
                            if question and answer:
                                content = f"Q: {question}\nA: {answer}"
                                # Prefix with product/category name
                                content = f"[{category_name}]\n{content}"
                                documents.append(
                                    Document(
                                        content=content,
                                        metadata={
                                            "product": category_name,
                                            "source": json_file.name,
                                            "type": "runtime_qa",
                                        },
                                    )
                                )
                
                logger.info("Processed runtime document: %s (%d Q&As)", json_file.name, len(documents))
            except Exception as e:
                logger.error("Failed to process runtime document %s: %s", json_file.name, e)
                continue
        
        return documents

    def _save(self, documents: list[Document], filename: str) -> None:
        """Save processed documents to a JSON file."""
        output_path = self._output_dir / filename
        data = [doc.to_dict() for doc in documents]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d documents to %s", len(documents), output_path)


if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = ETLPipeline()
    documents = pipeline.run()
    
    # Print a final success message to the console
    print(f"\nETL Pipeline completed successfully!")
    print(f"Processed {len(documents)} total documents.")
    # Assuming config.PROCESSED_DATA_DIR is available, otherwise just mention the directory
    print("Check your output directory for the generated JSON files.")
