"""Audit logging for security and compliance events."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import config


class AuditLogger:
    """Logs all security, safety, and compliance events for audit trails."""

    _LOG_DIR = config.PROJECT_ROOT / "logs"
    _SECURITY_LOG_FILE = "security_audit.log"

    def __init__(self) -> None:
        """Initialize audit logger with file handler."""
        self._LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            log_file = self._LOG_DIR / self._SECURITY_LOG_FILE
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_jailbreak_attempt(self, user_id: str, pattern_matched: str, input_preview: str) -> None:
        """Log a detected jailbreak attempt.

        Args:
            user_id: The user attempting the jailbreak.
            pattern_matched: The regex pattern that triggered detection.
            input_preview: First 100 chars of the input.
        """
        self.logger.warning(
            f"JAILBREAK_ATTEMPT | user={user_id} | pattern={pattern_matched} | "
            f"input_preview={input_preview[:100]}"
        )

    def log_pii_detected_and_redacted(self, user_id: str, pii_type: str, count: int) -> None:
        """Log PII detection and redaction.

        Args:
            user_id: The user whose input contained PII.
            pii_type: Type of PII detected (e.g., 'CNIC', 'EMAIL').
            count: Number of PII entities redacted.
        """
        self.logger.info(
            f"PII_REDACTED | user={user_id} | type={pii_type} | count={count}"
        )

    def log_input_validation_failed(self, user_id: str, reason: str) -> None:
        """Log input validation failure.

        Args:
            user_id: The user.
            reason: Reason for failure (e.g., 'too_long', 'empty').
        """
        self.logger.warning(
            f"INPUT_VALIDATION_FAILED | user={user_id} | reason={reason}"
        )

    def log_guardrail_blocked_query(self, user_id: str, guardrail: str, reason: str) -> None:
        """Log when a guardrail blocks a query.

        Args:
            user_id: The user.
            guardrail: Name of the guardrail that blocked it.
            reason: Reason for blocking.
        """
        self.logger.warning(
            f"QUERY_BLOCKED | user={user_id} | guardrail={guardrail} | reason={reason}"
        )

    def log_document_uploaded(self, user_id: str, filename: str, doc_count: int) -> None:
        """Log document upload by admin.

        Args:
            user_id: Admin user who uploaded.
            filename: Name of uploaded file.
            doc_count: Number of documents extracted.
        """
        self.logger.info(
            f"DOCUMENT_UPLOADED | user={user_id} | filename={filename} | "
            f"doc_count={doc_count}"
        )

    def log_out_of_domain_query(self, user_id: str, query_preview: str) -> None:
        """Log out-of-domain queries that were rejected.

        Args:
            user_id: The user.
            query_preview: First 100 chars of the query.
        """
        self.logger.info(
            f"OUT_OF_DOMAIN_REJECTED | user={user_id} | query_preview={query_preview[:100]}"
        )
