"""CSV data processor with PII anonymization."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data_processing.base_processor import BaseProcessor, Document
from src.guardrails.pii_anonymizer import PIIAnonymizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CSVProcessor(BaseProcessor):
    """Processes CSV files with automatic row-level PII anonymization.

    Reads CSV exports, anonymizes sensitive columns, and converts each row
    to a Document chunk with metadata preservation.
    """

    # Columns commonly containing PII that should be anonymized
    PII_COLUMNS = {
        "customer_name", "name", "full_name",
        "email", "email_address",
        "phone", "phone_number", "mobile",
        "account_number", "account",
        "cnic", "id_number", "national_id",
        "iban", "bank_account",
    }

    def __init__(self, file_path: Path | str) -> None:
        """Initialize CSV processor.

        Args:
            file_path: Path to the CSV file.
        """
        self.file_path = Path(file_path)
        self._anonymizer = PIIAnonymizer()
        logger.info("CSVProcessor initialized for file: %s", self.file_path)

    def process(self) -> list[Document]:
        """Process CSV file with PII anonymization.

        Returns:
            List of Document objects, one per row.
        """
        if not self.file_path.exists():
            logger.error("CSV file not found: %s", self.file_path)
            return []

        try:
            df = pd.read_csv(self.file_path)
            logger.info(
                "Loaded CSV with %d rows, %d columns",
                len(df),
                len(df.columns),
            )

            # Anonymize PII columns
            df = self._anonymize_pii_columns(df)

            # Convert each row to a Document
            documents: list[Document] = []
            for idx, row in df.iterrows():
                # Build content from row data
                row_content = self._row_to_text(row)
                
                doc = Document(
                    content=row_content,
                    metadata={
                        "source": self.file_path.name,
                        "type": "csv_row",
                        "row_id": int(idx),
                        "original_columns": list(df.columns),
                    },
                )
                documents.append(doc)

            logger.info(
                "Processed %d rows from CSV, created %d documents",
                len(df),
                len(documents),
            )
            return documents

        except Exception as e:
            logger.error("Error processing CSV file: %s", str(e))
            return []

    def _anonymize_pii_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize PII in identified columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with PII anonymized.
        """
        df_copy = df.copy()
        anonymized_count = 0

        for col in df_copy.columns:
            col_lower = col.lower()
            
            # Check if this is a PII column
            if any(pii_col in col_lower for pii_col in self.PII_COLUMNS):
                logger.debug("Anonymizing PII column: %s", col)
                
                for idx in df_copy.index:
                    value = df_copy.at[idx, col]
                    if pd.notna(value) and isinstance(value, str):
                        anonymized = self._anonymizer.anonymize(value)
                        df_copy.at[idx, col] = anonymized
                        anonymized_count += 1

        logger.info("Anonymized %d PII values", anonymized_count)
        return df_copy

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        """Convert a DataFrame row to readable text.

        Args:
            row: A row from the DataFrame.

        Returns:
            Formatted text representation of the row.
        """
        lines = []
        for col, value in row.items():
            if pd.notna(value):
                lines.append(f"{col}: {value}")
        return "\n".join(lines)
