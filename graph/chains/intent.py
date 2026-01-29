"""
graph/chains/intent.py

Structured intent classification chain with Fast/Slow fallback.
Uses the pre-compiled prompt defined in graph/prompts/intent.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from graph.prompts.intent import INTENT_PROMPT
from graph.schemas import IntentMetadata


def build_intent_chain(
    fast_llm: BaseChatModel,
    slow_llm: Optional[BaseChatModel] = None,
    slow_fallback_enabled: bool = True,
    slow_fallback_min_confidence: float = 0.9,
) -> RunnableLambda:
    """
    Builds the intent classification logic with automatic fallback.
    """

    # 1. Bind Schema (Structured Output) to the imported Prompt
    fast_chain = INTENT_PROMPT | fast_llm.with_structured_output(IntentMetadata)
    slow_chain = (
        (INTENT_PROMPT | slow_llm.with_structured_output(IntentMetadata))
        if slow_llm
        else None
    )

    # 2. Define the Execution Logic (The "Router")
    def classify_user_intent(inputs: Dict[str, Any]) -> IntentMetadata:
        messages = inputs.get("messages", [])

        # Step A: Try Fast Model
        result = fast_chain.invoke({"messages": messages})

        # Step B: Check for Fallback
        if (
            slow_fallback_enabled
            and slow_chain is not None
            and result.confidence < slow_fallback_min_confidence
        ):
            # Optimization: Don't use slow model for obvious greetings/exits
            if result.intent not in ["GREETING", "OUT_OF_SCOPE"]:
                result2 = slow_chain.invoke({"messages": messages})

                # Only adopt the new result if it's actually confident
                # (Or strictly better than the fast model)
                if result2.confidence >= result.confidence:
                    return result2

        return result

    # 3. Return as a Runnable
    return RunnableLambda(classify_user_intent)
