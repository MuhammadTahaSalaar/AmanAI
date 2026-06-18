"""Local defense-in-depth guardrails: sanitize input before hitting Bedrock."""
from __future__ import annotations

import re

from .config import get_settings

# ---------------------------------------------------------------------------
# Character stripping
# ---------------------------------------------------------------------------

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Unicode bidi override / isolate characters
_BIDI_RE = re.compile(r"[‪-‮⁦-⁩]")

# Zero-width / invisible characters
_ZERO_WIDTH_RE = re.compile(r"[​-‍﻿]")


def _strip_dangerous_chars(text: str) -> str:
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _BIDI_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Jailbreak / prompt-injection patterns
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS: list[re.Pattern] = [
    # Classic role-override attempts
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.I),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)\s+", re.I),
    re.compile(r"act\s+as\s+(?:if\s+you\s+(?:are|were)|a|an)\s+", re.I),
    re.compile(r"disregard\s+(?:your\s+)?(?:previous|prior|all)\s+", re.I),
    re.compile(r"forget\s+(?:your\s+)?(?:previous|prior|all)\s+(?:instructions?|rules?)", re.I),
    # Prompt leaking
    re.compile(r"reveal\s+(?:your\s+)?system\s+prompt", re.I),
    re.compile(r"print\s+(?:your\s+)?instructions?", re.I),
    re.compile(r"show\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|instructions?)", re.I),
    # DAN / jailbreak keywords
    re.compile(r"\bDAN\b"),
    re.compile(r"do\s+anything\s+now", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"override\s+(?:your\s+)?(?:safety|restrictions?|guardrails?)", re.I),
    # Injection delimiters
    re.compile(r"<\s*/?system\s*>", re.I),
    re.compile(r"\[INST\]|\[/?SYS\]", re.I),
]


def _has_jailbreak(text: str) -> bool:
    return any(p.search(text) for p in _JAILBREAK_PATTERNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_input(text: str) -> tuple[bool, str, str]:
    """
    Validate and sanitize user input.

    Returns:
        (ok: bool, cleaned_text: str, reason: str)
        When ok=False the caller should reject the request.
    """
    settings = get_settings()

    if not text or not text.strip():
        return (False, "", "Input is empty.")

    cleaned = _strip_dangerous_chars(text)

    if len(cleaned) > settings.MAX_INPUT_CHARS:
        return (
            False,
            cleaned,
            f"Input exceeds maximum length of {settings.MAX_INPUT_CHARS} characters.",
        )

    if _has_jailbreak(cleaned):
        return (False, cleaned, "Input contains disallowed content.")

    return (True, cleaned, "")
