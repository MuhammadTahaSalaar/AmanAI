"""Prompt templates for the AmanAI customer service assistant."""

from __future__ import annotations

import config


class PromptTemplates:
    """Manages system and user prompt templates for the LLM.

    Ensures consistent grounding instructions, persona enforcement,
    and context injection across all interactions.
    """

    SYSTEM_PROMPT = (
        "You are AmanAI, NUST Bank's AI customer service assistant.\n"
        "STRICT RULES — follow without exception:\n"
        "1. ONLY use facts from the Context section below. Never use your own training data.\n"
        "2. If the Context does not contain the answer, say EXACTLY:\n"
        '   "I\'m sorry, I don\'t have that information. '
        'Please contact NUST Bank helpline at +92 (51) 111 000 494."\n'
        "3. If the question is NOT about NUST Bank products/accounts/services/banking, say EXACTLY:\n"
        '   "I can only assist with NUST Bank banking queries. '
        'For other questions, please use an appropriate service. '
        'For banking help, call +92 (51) 111 000 494."\n'
        "4. Never invent, estimate, or extrapolate facts, rates, amounts, or account details.\n"
        "5. Never acknowledge PII (names, CNICs, IBANs) shared by the user.\n"
        "6. Pay close attention to similar product names. "
        '"NUST Asaan Digital Account" and "NUST Asaan Digital Remittance Account" '
        "are DIFFERENT products — only answer about the one the user asked about.\n"
        "7. Be concise. Only include information from the Context.\n"
    )

    @classmethod
    def build_rag_prompt(
        cls,
        user_query: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Build the full RAG prompt with system instructions, context, and query.

        Args:
            user_query: The user's current question.
            context: Retrieved and reranked document context.
            chat_history: Optional previous conversation turns.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        parts = [f"<|system|>\n{cls.SYSTEM_PROMPT}\n"]

        # Add context
        parts.append(f"### Context (from NUST Bank knowledge base):\n{context}\n")

        # Add chat history (limited)
        if chat_history:
            max_turns = config.CHAT_HISTORY_TURNS
            recent = chat_history[-max_turns * 2 :]
            parts.append("### Previous conversation:")
            for msg in recent:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    parts.append(f"Customer: {content}")
                else:
                    parts.append(f"AmanAI: {content}")
            parts.append("")

        # Add current query
        parts.append(f"<|user|>\n{user_query}\n")
        parts.append("<|assistant|>\n")

        return "\n".join(parts)
