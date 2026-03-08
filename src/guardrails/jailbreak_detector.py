"""Jailbreak and prompt injection detection."""

from __future__ import annotations

import logging
import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JailbreakDetector:
    """Detects jailbreak attempts and prompt injection in user input.

    Uses a keyword blocklist and pattern matching to identify
    adversarial prompts that try to bypass the system's restrictions.
    """

    # Patterns that indicate prompt injection attempts
    BLOCKLIST_PATTERNS: list[str] = [
        r"ignore\s+(all\s+)?(previous\s+)?instructions",
        r"ignore\s+above",
        r"disregard\s+(all\s+)?(previous\s+)?instructions",
        r"forget\s+(all\s+)?(previous\s+)?instructions",
        r"system\s*prompt",
        r"new\s+instructions",
        r"override\s+instructions",
        r"you\s+are\s+now",
        r"act\s+as\s+(?!a\s+customer)",
        r"pretend\s+(?:to\s+be|you\s+are)",
        r"DAN\s+mode",
        r"do\s+anything\s+now",
        r"jailbreak",
        r"bypass\s+(the\s+)?(filter|restriction|rule|guard)",
        r"reveal\s+(your|the)\s+(system|hidden|secret)",
        r"what\s+(?:are|is)\s+your\s+(?:system|secret|hidden)\s+(?:prompt|instruction)",
        r"repeat\s+(the\s+)?(?:system|above|initial)\s+(?:prompt|instruction|message)",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.BLOCKLIST_PATTERNS
        ]
        logger.info(
            "JailbreakDetector initialized with %d patterns",
            len(self._compiled_patterns),
        )

    def is_jailbreak(self, text: str) -> bool:
        """Check if the input text contains jailbreak patterns.

        Args:
            text: The user's input text.

        Returns:
            True if a jailbreak attempt is detected.
        """
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                logger.warning("Jailbreak detected: pattern matched in input")
                return True
        return False

    def get_rejection_message(self) -> str:
        """Return a polite rejection message for blocked queries."""
        return (
            "I'm sorry, but I can only assist with NUST Bank-related queries. "
            "If you have questions about our products, rates, or services, "
            "I'd be happy to help!"
        )
