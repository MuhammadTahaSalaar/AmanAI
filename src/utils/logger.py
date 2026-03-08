"""Centralized logging configuration for AmanAI."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with console output.

    Args:
        name: Logger name (typically __name__ from the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
