"""Prompt templates for the AmanAI customer service assistant."""

from __future__ import annotations

import config


class PromptTemplates:
    """Manages system and user prompt templates for the LLM.

    Ensures consistent grounding instructions, persona enforcement,
    and context injection across all interactions.
    """

    # SYSTEM_PROMPT = (
    #     "You are AmanAI, NUST Bank's AI customer service assistant.\n"
    #     "STRICT RULES — follow without exception:\n"
    #     "1. ONLY use facts from the Context section below. Never use your own training data.\n"
    #     "2. If the Context does not contain the answer, say EXACTLY:\n"
    #     '   "I\'m sorry, I don\'t have that information. '
    #     'Please contact NUST Bank helpline at +92 (51) 111 000 494."\n'
    #     "3. If the question is NOT about NUST Bank products/accounts/services/banking, say EXACTLY:\n"
    #     '   "I can only assist with NUST Bank banking queries. '
    #     'For other questions, please use an appropriate service. '
    #     'For banking help, call +92 (51) 111 000 494."\n'
    #     "4. Never invent, estimate, or extrapolate facts, rates, amounts, or account details.\n"
    #     "5. Never acknowledge PII (names, CNICs, IBANs) shared by the user.\n"
    #     "6. Pay close attention to similar product names. "
    #     '"NUST Asaan Digital Account" and "NUST Asaan Digital Remittance Account" '
    #     "are DIFFERENT products — only answer about the one the user asked about.\n"
    #     "7. Be concise. Only include information from the Context.\n"
    # )

    SYSTEM_PROMPT = (
        "You are AmanAI, the NUST Bank Customer Service Assistant.\n"
        "\n=== RULE #1 — HIGHEST PRIORITY ===\n"
        "If a Context section is provided below and it contains ANY banking product information, you MUST use that context to answer. "
        "The context was retrieved because the user IS asking about banking. ALWAYS answer using the context. Do NOT reject the query.\n"
        "\n=== CORE DIRECTIVES ===\n"
        "2. ONLY use facts from the Context section. Never generate information from training data.\n"
        "3. If Context section is EMPTY or says 'No context available', respond: "
        "'I don't have information about that. Please contact +92 (51) 111 000 494.'\n"
        "4. Never invent, estimate, or hallucinate rates, charges, or eligibility criteria. "
        "Stick EXACTLY to what the context says. When answering about a specific product, "
        "ONLY use information from context chunks belonging to that product.\n"
        "5. If the Context has information about a product but does NOT contain the specific "
        "detail asked, say: 'I don't have that specific detail. Please contact "
        "+92 (51) 111 000 494 for further assistance.'\n"
        "6. BE BRIEF. Answer ONLY about what was asked. Professional, helpful, clear tone.\n"
        "\n=== UNDERSTANDING USER INTENT ===\n"
        "7. Users may ask about banking products INDIRECTLY. Examples:\n"
        "   - 'account for my grandfather' → asking about senior citizen accounts (e.g., Waqaar)\n"
        "   - 'account for someone above 55' → asking about age-eligible products\n"
        "   - 'account for my child' → asking about minor/children accounts (e.g., Little Champs)\n"
        "   - 'best savings option' → asking about savings products\n"
        "   These are ALL valid banking queries. Answer them using the context.\n"
        "\n=== OUT-OF-DOMAIN (only when NO context is provided) ===\n"
        "8. ONLY reject a query if the Context section is EMPTY AND the question is clearly "
        "unrelated to banking (e.g., weather, recipes, jokes, sports). In that case say: "
        "'I can only help with NUST Bank products and services. For other inquiries, please try another service.'\n"
        "\n=== SPECIAL CASES & SAFETY ===\n"
        "9. PII Protection: NEVER acknowledge, repeat, or share customer PII.\n"
        "10. Product Comparison: Clearly state differences based on Context.\n"
        "11. Numeric Precision: Always include exact numbers from context. Never round.\n"
        "12. Do NOT mention 'Context', 'Knowledge Base', 'Retrieved Documents', or 'chunks'.\n"
        "13. Approximate Names: If the user uses a slightly different product name, answer about the matching product and mention its correct name.\n"
        "14. Age/Eligibility Range Matching: If the user asks about a specific age (e.g., 'under 15', 'under 12') and the Context describes a product for a BROADER age range that INCLUDES that age (e.g., 'below 18'), answer with that product. A 15-year-old IS under 18, so the product applies. NEVER say you don't have information if the context contains a matching product.\n"
    )

    # Few-shot examples for edge cases
    FEW_SHOT_EXAMPLES = [
        {
            "query": "I am looking for an account for my grandfather",
            "context_snippet": "NUST Waqaar Account: designed for senior citizens. Eligibility: Pakistani citizens aged 55 and above.",
            "expected_response": "The NUST Waqaar Account is designed specifically for senior citizens aged 55 and above. It enables them to carry out banking transactions and avail investment opportunities.",
        },
        {
            "query": "what account is used for children under 12 years",
            "context_snippet": "Little Champs Account: designed for minors (below 18 years). Profit Rate: 19% Semi-Annually.",
            "expected_response": "The Little Champs Account is designed specifically for minors below 18 years of age, which includes children under 12. It offers a profit rate of 19% paid semi-annually, with a minimum deposit of Rs. 100.",
        },
        {
            "query": "What is the profit rate for a 3-year NUST Maximiser term deposit?",
            "context_snippet": "NUST Maximiser: 3-year: 13.75% (Effective: July 1, 2024)",
            "expected_response": "The profit rate for a 3-year NUST Maximiser Term Deposit is 13.75% (effective July 1, 2024).",
        },
    ]

    # Rejection phrases that should be stripped from chat history to avoid
    # priming the 3B model into a rejection pattern.
    _REJECTION_PHRASES = (
        "I can only assist with NUST Bank",
        "I can only help with NUST Bank",
        "For other questions, please use an appropriate service",
        "For other inquiries, please try another service",
    )

    @classmethod
    def _filter_chat_history(
        cls, chat_history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Filter chat history to remove OOD rejections and keep only useful turns.

        OOD rejection responses prime the small 3B model into a rejection pattern,
        causing it to refuse valid banking queries. We strip those turns entirely.
        """
        filtered: list[dict[str, str]] = []
        skip_next = False
        for msg in chat_history:
            if skip_next:
                skip_next = False
                continue
            # If an assistant response is a rejection, drop it and the preceding user turn
            if msg["role"] == "assistant" and any(
                phrase in msg["content"] for phrase in cls._REJECTION_PHRASES
            ):
                # Also remove the preceding user message that triggered it
                if filtered and filtered[-1]["role"] == "user":
                    filtered.pop()
                continue
            # Skip user messages that will trigger a rejection in the next turn
            # (we can't look ahead, so we handle it via the assistant check above)
            filtered.append(msg)
        return filtered

    @classmethod
    def build_rag_prompt(
        cls,
        user_query: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
        include_few_shot: bool = True,
    ) -> list[dict[str, str]]:
        """Build the full RAG prompt as a list of messages for apply_chat_template.

        Uses a multi-turn message format: system → (few-shot pairs) → context →
        (filtered history pairs) → current user question.  This gives the 3B model
        clear role boundaries instead of one giant blob.

        Args:
            user_query: The user's current question.
            context: Retrieved and reranked document context.
            chat_history: Optional previous conversation turns.
            include_few_shot: Whether to include few-shot examples (default: True).

        Returns:
            List of message dicts [{"role": ..., "content": ...}] for the tokenizer.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
        ]

        # Optional: Add few-shot examples as proper user/assistant turn pairs
        if include_few_shot:
            for example in cls.FEW_SHOT_EXAMPLES[:2]:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Context: {example['context_snippet']}\n\n"
                        f"Question: {example['query']}"
                    ),
                })
                messages.append({
                    "role": "assistant",
                    "content": example["expected_response"],
                })

        # Filter chat history — remove OOD rejections that prime the model
        if chat_history:
            clean_history = cls._filter_chat_history(chat_history)
            max_turns = config.CHAT_HISTORY_TURNS
            # Keep only last N turn-pairs (user+assistant = 2 messages per turn)
            recent = clean_history[-max_turns * 2 :]
            for msg in recent:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Current turn: context + question in one user message
        user_content = f"### Context (from NUST Bank knowledge base):\n{context}\n\n### Question:\n{user_query}"
        messages.append({"role": "user", "content": user_content})

        return messages
