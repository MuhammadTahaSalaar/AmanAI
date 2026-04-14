"""Safety manager orchestrating all input and output guardrails."""

from __future__ import annotations

import logging
import re

import config
from src.guardrails.pii_anonymizer import PIIAnonymizer
from src.guardrails.jailbreak_detector import JailbreakDetector
from src.guardrails.semantic_safety_detector import SemanticSafetyDetector
from src.guardrails.audit_logger import AuditLogger
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafetyManager:
    """Orchestrates all safety guardrails for input sanitization and output filtering.

    Applies PII anonymization, jailbreak detection, semantic safety, and input length
    validation in sequence before queries reach the RAG pipeline.
    """

    def __init__(
        self,
        pii_anonymizer: PIIAnonymizer | None = None,
        jailbreak_detector: JailbreakDetector | None = None,
        semantic_detector: SemanticSafetyDetector | None = None,
        audit_logger: AuditLogger | None = None,
        max_input_length: int | None = None,
        user_id: str = "unknown",
    ) -> None:
        self._pii = pii_anonymizer or PIIAnonymizer()
        self._jailbreak = jailbreak_detector or JailbreakDetector()
        self._semantic = semantic_detector or SemanticSafetyDetector()
        self._audit = audit_logger or AuditLogger()
        self._max_length = max_input_length or config.MAX_INPUT_LENGTH
        self._user_id = user_id

    def validate_input(self, user_input: str) -> tuple[bool, str, str]:
        """Validate and sanitize user input through all guardrails.

        Args:
            user_input: Raw user input text.

        Returns:
            Tuple of (is_safe, sanitized_text, rejection_reason).
            If is_safe is False, rejection_reason contains the user-facing message.
        """
        # 1. Strip control characters
        cleaned = self._strip_control_chars(user_input)

        # 2. Check length
        if len(cleaned) > self._max_length:
            self._audit.log_input_validation_failed(self._user_id, "too_long")
            return (
                False,
                "",
                f"Your message is too long. Please limit it to {self._max_length} characters.",
            )

        # 3. Empty check
        if not cleaned.strip():
            return False, "", "Please enter a question to get started."

        # 4. Jailbreak detection (pattern-based)
        if self._jailbreak.is_jailbreak(cleaned):
            self._audit.log_jailbreak_attempt(
                self._user_id,
                "regex_pattern",
                cleaned
            )
            return False, "", self._jailbreak.get_rejection_message()

        # 5. Semantic safety detection
        is_safe, issue = self._semantic.detect_safety_issue(cleaned)
        if not is_safe:
            self._audit.log_guardrail_blocked_query(self._user_id, "semantic_safety", issue)
            return False, "", "I can only assist with NUST Bank-related banking queries. Please ask about our products and services."

        # 6. PII anonymization (on query before it reaches the LLM)
        sanitized = self._pii.anonymize(cleaned)

        return True, sanitized, ""

    def sanitize_output(self, response: str) -> str:
        """Sanitize LLM output before displaying to the user.

        Args:
            response: Raw LLM-generated response.

        Returns:
            Sanitized response with any leaked PII redacted.
        """
        return self._pii.anonymize_output(response)

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        """Remove control characters except newlines and tabs."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
