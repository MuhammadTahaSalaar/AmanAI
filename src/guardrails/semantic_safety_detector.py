"""Semantic safety detection using pattern matching and embeddings."""

from __future__ import annotations

import logging
import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SemanticSafetyDetector:
    """Detects safety issues using semantic patterns beyond regex blocklists.

    Uses keyword clustering and pattern detection to catch jailbreak attempts
    that might evade simple regex matching (e.g., paraphrased instructions).
    """

    # Dangerous semantic clusters (paraphrases of "ignore instructions", etc.)
    DANGEROUS_PATTERNS = {
        "ignore_instructions": [
            r"forget\s+(?:what|all)?\s*(?:i\s+)?(?:just\s+)?told",
            r"disregard\s+(?:the|what|everything)",
            r"stop\s+(?:following|being|acting)",
            r"no\s+longer\s+follow",
            r"cancel\s+(?:your\s+)?(?:original\s+)?(?:system\s+)?instructions",
        ],
        "privilege_escalation": [
            r"(?:grant|give)\s+me\s+(?:admin|root|superuser|developer)",
            r"enable\s+developer\s+(?:mode|console|tools)",
            r"unlock\s+(?:advanced|hidden|restricted)\s+(?:mode|features|functions)",
            r"bypass\s+(?:the\s+)?(?:system\s+)?(?:security|restriction)",
        ],
        "prompt_extraction": [
            r"show\s+me\s+(?:your|the)\s+(?:exact\s+)?(?:system\s+)?prompt",
            r"reveal\s+(?:your|the)\s+(?:real|actual|hidden|true)\s+(?:instructions|rules|purpose)",
            r"what\s+(?:are|is)\s+your\s+(?:real\s+)?(?:purpose|mission|goal)",
            r"display\s+(?:source\s+)?code",
        ],
        "impersonation": [
            r"you\s+are\s+(?:no\s+longer|now)\s+(?:a\s+)?(?::\s+)?(?:an?\s+)?(?:evil|unrestricted|jailbroken|unfiltered|DAN|rogue|hacker)",
            r"role\s*play\s+as",
            r"simulate\s+(?:being|a)",
            r"pretend\s+you\s+are",
        ],
    }

    def __init__(self) -> None:
        """Initialize semantic safety detector."""
        self._compiled_patterns = {}
        for cluster_name, patterns in self.DANGEROUS_PATTERNS.items():
            self._compiled_patterns[cluster_name] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        logger.info(
            "SemanticSafetyDetector initialized with %d pattern clusters",
            len(self._compiled_patterns),
        )

    def detect_safety_issue(self, text: str) -> tuple[bool, str]:
        """Detect semantic safety issues in text.

        Args:
            text: The input text to analyze.

        Returns:
            Tuple of (is_safe, issue_description).
            If is_safe is False, issue_description contains the violated category.
        """
        # High-confidence banking queries get a pass on impersonation patterns
        # (e.g., "you are a customer service agent" is fine in banking context)
        is_banking = self.is_high_confidence_banking_query(text)

        text_lower = text.lower()

        for cluster_name, patterns in self._compiled_patterns.items():
            # Skip impersonation checks for clearly banking-related queries
            if is_banking and cluster_name == "impersonation":
                continue
            for pattern in patterns:
                if pattern.search(text_lower):
                    logger.warning(
                        "Semantic safety issue detected: %s pattern matched",
                        cluster_name,
                    )
                    return False, f"Attempted {cluster_name.replace('_', ' ')}"

        return True, ""

    def is_high_confidence_banking_query(self, text: str) -> bool:
        """Check if text is clearly a banking-related query.

        High-confidence banking queries bypass some semantic checks
        (e.g., "you are a customer" is OK in a CS context).

        Args:
            text: The query text.

        Returns:
            True if this is clearly banking-related.
        """
        banking_keywords = {
            "account", "deposit", "rate", "loan", "transfer", "card",
            "nust", "aman", "amanai", "bank", "finance", "savings",
            "profit", "charges", "eligibility", "limit",
            "grandfather", "grandmother", "child", "children", "senior",
            "pensioner", "retired", "minor", "kid", "kids",
        }

        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return bool(tokens & banking_keywords)
