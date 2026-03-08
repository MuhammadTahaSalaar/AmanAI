"""Safety manager orchestrating all input and output guardrails."""

from __future__ import annotations

import logging
import re

import config
from src.guardrails.pii_anonymizer import PIIAnonymizer
from src.guardrails.jailbreak_detector import JailbreakDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafetyManager:
    """Orchestrates all safety guardrails for input sanitization and output filtering.

    Applies PII anonymization, jailbreak detection, and input length
    validation in sequence before queries reach the RAG pipeline.
    """

    def __init__(
        self,
        pii_anonymizer: PIIAnonymizer | None = None,
        jailbreak_detector: JailbreakDetector | None = None,
        max_input_length: int | None = None,
    ) -> None:
        self._pii = pii_anonymizer or PIIAnonymizer()
        self._jailbreak = jailbreak_detector or JailbreakDetector()
        self._max_length = max_input_length or config.MAX_INPUT_LENGTH

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
            return (
                False,
                "",
                f"Your message is too long. Please limit it to {self._max_length} characters.",
            )

        # 3. Empty check
        if not cleaned.strip():
            return False, "", "Please enter a question to get started."

        # 4. Jailbreak detection
        if self._jailbreak.is_jailbreak(cleaned):
            return False, "", self._jailbreak.get_rejection_message()

        # 5. PII anonymization (on query before it reaches the LLM)
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
