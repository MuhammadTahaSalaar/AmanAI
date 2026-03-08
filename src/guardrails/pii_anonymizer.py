"""PII anonymization using Microsoft Presidio with custom Pakistani recognizers."""

from __future__ import annotations

import logging
import re

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PIIAnonymizer:
    """Detects and redacts Personally Identifiable Information from text.

    Uses Microsoft Presidio with custom recognizers for Pakistani-specific
    identifiers (CNIC, IBAN) alongside standard entity detection.
    """

    # Entities to detect in USER INPUT (includes PERSON — real names users type)
    _INPUT_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "PK_CNIC",
        "PK_IBAN",
        "PK_PHONE",
        "CREDIT_CARD",
    ]

    # Entities to detect in LLM OUTPUT — exclude PERSON because product/place names
    # (e.g. "NUST Maximiser", "NUST Sahar") are mis-tagged as persons by SpacyRecognizer
    # PERSON is re-added at a higher score threshold via a separate analysis pass
    _OUTPUT_ENTITIES = [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "PK_CNIC",
        "PK_IBAN",
        "PK_PHONE",
        "CREDIT_CARD",
    ]

    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._register_custom_recognizers()
        logger.info("PIIAnonymizer initialized with custom recognizers")

    def _register_custom_recognizers(self) -> None:
        """Register custom regex recognizers for Pakistani identifiers."""
        # CNIC: XXXXX-XXXXXXX-X
        cnic_recognizer = PatternRecognizer(
            supported_entity="PK_CNIC",
            patterns=[
                Pattern(
                    name="pk_cnic",
                    regex=r"\b\d{5}-\d{7}-\d{1}\b",
                    score=0.95,
                )
            ],
        )
        self._analyzer.registry.add_recognizer(cnic_recognizer)

        # IBAN: PK + 2 digits + 4 char bank code + 16 digits
        iban_recognizer = PatternRecognizer(
            supported_entity="PK_IBAN",
            patterns=[
                Pattern(
                    name="pk_iban",
                    regex=r"\bPK\d{2}[A-Z0-9]{4}\d{16}\b",
                    score=0.95,
                )
            ],
        )
        self._analyzer.registry.add_recognizer(iban_recognizer)

        # Pakistani phone numbers
        phone_recognizer = PatternRecognizer(
            supported_entity="PK_PHONE",
            patterns=[
                Pattern(
                    name="pk_phone",
                    regex=r"\b(?:\+92|0)\d{3}[\s-]?\d{7}\b",
                    score=0.85,
                )
            ],
        )
        self._analyzer.registry.add_recognizer(phone_recognizer)

    def anonymize(self, text: str) -> str:
        """Detect and redact PII from the given text.

        Args:
            text: Input text that may contain PII.

        Returns:
            Text with PII replaced by redaction tokens.
        """
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._INPUT_ENTITIES,
            score_threshold=0.6,
        )

        if results:
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
            logger.debug("Anonymized %d PII entities", len(results))
            return anonymized.text

        return text

    def anonymize_output(self, text: str) -> str:
        """Detect and redact PII from LLM-generated output.

        Uses a narrower entity set than anonymize() — excludes PERSON to prevent
        false positives on product names like 'NUST Maximiser' or 'NUST Sahar'.
        PERSON is re-detected at a high score threshold (0.85+) so only confident
        full person names (e.g. 'Arif Alvi') are redacted, not product names.

        Args:
            text: LLM-generated text that may contain leaked PII.

        Returns:
            Text with PII replaced by redaction tokens.
        """
        # Pass 1: standard entities (no PERSON) at normal threshold
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._OUTPUT_ENTITIES,
            score_threshold=0.6,
        )

        # Pass 2: PERSON at high threshold to catch real names but not product names
        person_results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=["PERSON"],
            score_threshold=0.85,
        )
        results = results + person_results

        if results:
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
            logger.debug("Sanitized %d PII entities from output", len(results))
            return anonymized.text

        return text

    def contains_pii(self, text: str) -> bool:
        """Check if the text contains any PII without modifying it.

        Args:
            text: Text to scan.

        Returns:
            True if PII is detected.
        """
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._INPUT_ENTITIES,
            score_threshold=0.6,
        )
        return len(results) > 0
