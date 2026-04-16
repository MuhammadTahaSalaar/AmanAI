"""PII anonymization using Microsoft Presidio with custom Pakistani recognizers."""

from __future__ import annotations

import logging
import re
import warnings

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PIIAnonymizer:
    """Detects and redacts Personally Identifiable Information from text.

    Uses Microsoft Presidio with custom recognizers for Pakistani-specific
    identifiers (CNIC, IBAN) alongside standard entity detection.
    
    Bank contact information (emails, phone numbers) is whitelisted to ensure
    official NUST Bank contact details are never redacted from responses.
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

    # Whitelist of bank contact information that should NOT be redacted
    # These are official NUST Bank public contact details
    _BANK_CONTACT_WHITELIST = {
        # Phone numbers (exact matches and normalized variations)
        "+92 (51) 111 000 494",
        "+92(51)111000494",
        "0511110004 94",
        "+92 51 111 000 494",
        "92 (51) 111 000 494",
        # Email addresses
        "support@NUSTbank.com.pk",
        "support@nustbank.com.pk",
    }

    # Regex matching NUST Bank product names — entities overlapping these should
    # NOT be redacted as PERSON (avoids false positives like "Digital Account"
    # or "Freelancer Digital" being tagged as person names).
    _PRODUCT_NAME_RE = re.compile(
        r"NUST\s+(?:Freelancer|Asaan|Waqaar|Sahar|Maximiser|Bachat|"
        r"Imarat|Ujala|Champs|Little|Roshan|Pensioner|Digital|FYP|"
        r"4Car|Hunarmand|Special|Mortgage|Personal|Mastercard|PayPak)"
        r"(?:\s+\w+)*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        # Suppress Presidio's verbose logging during initialization
        presidio_logger = logging.getLogger("presidio-analyzer")
        old_level = presidio_logger.level
        presidio_logger.setLevel(logging.ERROR)
        
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._register_custom_recognizers()
        
        # Restore original log level
        presidio_logger.setLevel(old_level)
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

        # Filter out PERSON entities that overlap with NUST Bank product names
        results = [
            r for r in results
            if r.entity_type != "PERSON"
            or not self._PRODUCT_NAME_RE.search(text[max(0, r.start - 30):r.end + 10])
        ]

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

        Bank contact information (emails, phone numbers) is restored after
        anonymization to ensure official NUST Bank contact details are visible.

        Args:
            text: LLM-generated text that may contain leaked PII.

        Returns:
            Text with PII replaced by redaction tokens, except for whitelisted
            bank contact information.
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
        # Filter out short PERSON entities (< 4 chars) to avoid false positives
        # on banking abbreviations like "SE" (Small Enterprise), "ME" (Medium Enterprise)
        person_results = [
            r for r in person_results
            if (r.end - r.start) >= 4
        ]
        # Filter out PERSON entities that overlap with NUST Bank product names
        # (e.g. "NUST Freelancer Digital Account" should not have "Digital Account" tagged as PERSON)
        person_results = [
            r for r in person_results
            if not self._PRODUCT_NAME_RE.search(text[max(0, r.start - 30):r.end + 10])
        ]
        results = results + person_results

        if results:
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
            sanitized_text = anonymized.text
            logger.debug("Sanitized %d PII entities from output", len(results))
            
            # Restore whitelisted bank contact information
            restored_text = self._restore_bank_contacts(sanitized_text, text)
            return restored_text

        return text

    def _restore_bank_contacts(self, anonymized_text: str, original_text: str) -> str:
        """Restore whitelisted bank contact information in anonymized text.

        This method finds all bank contact details in the original text and
        ensures they appear unredacted in the output, even if the anonymizer
        flagged them as PII.

        Args:
            anonymized_text: Text with PII redacted by Presidio.
            original_text: Original text before anonymization.

        Returns:
            Text with bank contacts restored to their original form.
        """
        restored = anonymized_text

        # Find all bank contacts in the original text
        for contact in self._BANK_CONTACT_WHITELIST:
            if contact in original_text:
                # Find the redacted version in the anonymized text
                # Presidio typically replaces with <ENTITY_TYPE> or similar
                # We'll use a regex to find patterns matching the entity type
                if "@" in contact:  # email
                    # Email addresses are often redacted; try to restore them
                    # Look for redaction patterns like <EMAIL_ADDRESS> or [REDACTED]
                    restored = self._restore_contact_in_text(
                        restored, contact, ["<EMAIL_ADDRESS>", "[REDACTED]", "<EMAIL>"]
                    )
                elif "+" in contact or any(c.isdigit() for c in contact):  # phone
                    # Phone numbers redaction patterns
                    restored = self._restore_contact_in_text(
                        restored,
                        contact,
                        ["<PHONE_NUMBER>", "<PK_PHONE>", "[REDACTED]", "<PHONE>"],
                    )

        return restored

    @staticmethod
    def _restore_contact_in_text(
        text: str, original_contact: str, redaction_patterns: list[str]
    ) -> str:
        """Replace the first occurrence of a redaction pattern with the original contact.

        Args:
            text: Text that may contain redactions.
            original_contact: The original contact information.
            redaction_patterns: List of redaction patterns to search for.

        Returns:
            Text with the first matching redaction pattern replaced by the contact.
        """
        for pattern in redaction_patterns:
            if pattern in text:
                # Replace only the first occurrence
                return text.replace(pattern, original_contact, 1)
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
