"""
graph/state.py

Complete LangGraph state schema for Fitbit Conversational AI assistant.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from graph.schemas import (ConversationState, CuratedKBQuery, Fact,
                           IntentMetadata, ToolCall, UserProfile)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────


class AssistantState(TypedDict):
    """
    Execution-time container for the Fitbit conversational assistant.

    Attributes:
        messages: Ordered chat history, each item either a LangChain message or dict payload.
        conversation_state: Persisted summary of the dialogue used for follow-up reasoning.
        intent_metadata: Latest routing decision emitted by the intent classification chain.
        plan: Optional high-level response plan shared across nodes in the graph.
        facts: Registry of grounded facts keyed by identifier for evidence tracking.
        tool_calls: Chronological record of structured tool invocations for the current turn.
        curated_kb_queries: Audit trail of curated knowledge-base lookups performed so far.
        user_profile: Hydrated Fitbit user profile supplying personalization context.
        response: Candidate natural-language reply assembled by downstream nodes.
        grounded: Indicates whether the drafted response is supported by retrieved facts.
        safe: Signals that safety and compliance checks have passed for the reply.
        suggestiveness: Numeric guidance for how proactive recommendations should be.
        suggestion_included: Marks whether a coach-like suggestion was added to the response.
    """

    messages: Annotated[List[Any], add_messages]

    conversation_state: Optional[ConversationState]
    intent_metadata: Optional[IntentMetadata]

    # These fields are typically overwritten per turn, so simple types are fine
    plan: Optional[str]  # Or ProcessPlan object
    facts: Dict[str, Fact]
    tool_calls: List[ToolCall]
    curated_kb_queries: List[CuratedKBQuery]
    user_profile: Optional[UserProfile]

    response: Optional[str]
    grounded: bool
    safe: bool
    suggestiveness: float
    suggestion_included: bool

    # Process specific fields (ensure these match what your nodes return)
    process_plan: Optional[Any]
    needs_clarification: bool
    grounding_metadata: Optional[Any]
