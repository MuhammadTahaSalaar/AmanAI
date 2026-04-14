"""Query expansion for enhanced banking query coverage."""

from __future__ import annotations

import logging
import re

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryExpander:
    """Expands banking queries with synonyms and semantic variants.

    Helps catch variations in terminology (e.g., 'transfer' vs 'remittance')
    and improves retrieval coverage for semantically similar concepts.
    """

    # Banking terminology synonyms and variants
    SYNONYMS = {
        "transfer": ["remittance", "fund transfer", "send money", "payment"],
        "account": ["account type", "product", "account product"],
        "rate": ["profit rate", "return", "profit margin", "interest rate"],
        "loan": ["financing", "credit facility", "advance", "finance"],
        "deposit": ["savings", "term deposit", "fixed deposit"],
        "charges": ["fee", "fees", "cost", "penalty"],
        "credit": ["credit limit", "credit facility"],
        "debit": ["debit card", "debit facility"],
        "card": ["debit card", "credit card"],
        "crypto": ["cryptocurrency", "bitcoin", "blockchain"],
        "freelancer": ["freelance", "self-employed", "independent"],
        "usd": ["dollar", "foreign currency", "international"],
        "easy": ["simple", "convenient", "hassle-free"],
        "young": ["youth", "young professional", "teenager"],
        "senior": ["elderly", "pensioner", "retired"],
        "maximum": ["max", "highest", "upper limit"],
        "minimum": ["min", "lowest", "lower limit"],
    }

    def __init__(self) -> None:
        """Initialize query expander."""
        logger.debug(f"QueryExpander initialized with {len(self.SYNONYMS)} synonym groups")

    def expand(self, query: str) -> list[str]:
        """Generate expanded query variants with synonyms.

        Args:
            query: The original user query.

        Returns:
            List starting with original query, followed by semantic variants (up to 5 total).

        Examples:
            >>> expander.expand("What is the transfer limit?")
            [
                "What is the transfer limit?",
                "What is the remittance limit?",
                "What is the fund transfer limit?",
                "What is the send money limit?",
            ]
        """
        variants = [query]  # Always include original first
        query_lower = query.lower()

        # Find all matched keywords in query
        matched_keywords = []
        for keyword, synonyms in self.SYNONYMS.items():
            # Check if keyword appears as a word boundary
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, query_lower):
                matched_keywords.append((keyword, synonyms))

        # Generate variants by substituting synonyms
        for keyword, synonyms in matched_keywords:
            for synonym in synonyms:
                variant = re.sub(
                    rf"\b{re.escape(keyword)}\b",
                    synonym,
                    query,
                    flags=re.IGNORECASE,
                    count=1,
                )
                if variant not in variants:  # Avoid duplicates
                    variants.append(variant)

                if len(variants) >= 5:  # Limit to 5 variants
                    break
            if len(variants) >= 5:
                break

        logger.debug(f"Expanded query '{query[:50]}...' to {len(variants)} variants")
        return variants
