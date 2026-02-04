"""
graph/process/agents/suggestor.py

Refactored Suggestor agent builder with structured output support.

Key improvements:
- Uses structured output for reliable parsing
- Proper handling of context serialization
- Compatible with both tool-calling and text-based models
"""

from __future__ import annotations

from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graph.process.prompts.suggestor import SUGGESTOR_SYSTEM_PROMPT


class SuggestionResponse(BaseModel):
    """Structured output from the Suggestor agent."""

    suggestion: str = Field(
        description="A short, actionable coaching nudge (1-3 sentences) that builds on the assistant's answer."
    )
    include_suggestion: bool = Field(
        default=True,
        description="Set to False if no relevant suggestion can be made for this interaction.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this suggestion is relevant (for debugging/logging).",
    )


def build_suggestor_agent(llm_config: Dict[str, Any]) -> Runnable:
    """
    Build the Suggestor agent chain with structured output.

    The agent receives:
    - System prompt with suggestor instructions
    - User context (goals, preferences, memories)
    - Interaction context (user query + assistant's answer)
    - Tone parameter for personalization

    Returns a chain that outputs SuggestionResponse.

    Args:
        llm_config: Config dict for init_chat_model, e.g.:
            {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 256,
            }

    Returns:
        Runnable chain that takes dict with 'user_context', 'interaction', 'tone'
        and returns SuggestionResponse
    """

    llm = init_chat_model(**llm_config)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUGGESTOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "TONE: {tone}\n\n"
                "USER CONTEXT:\n{user_context}\n\n"
                "INTERACTION:\n{interaction}\n\n"
                "Generate a coaching suggestion based on the above context.",
            ),
        ]
    )

    # Bind structured output to the model
    # This works with models that support function calling
    # For models without tool calling, we'll need fallback parsing in the node
    try:
        llm_with_structure = llm.with_structured_output(
            SuggestionResponse,
            method="function_calling",
            include_raw=False,
        )
        chain = prompt | llm_with_structure
    except Exception:
        # Fallback for models without structured output support
        # We'll parse JSON from text in the node
        from langchain_core.output_parsers import StrOutputParser

        chain = prompt | llm | StrOutputParser()

    return chain
