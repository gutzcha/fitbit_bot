"""
graph/process/nodes/suggestor.py

Refactored Suggestor node that adds coaching nudges to assistant responses.

Key improvements:
- Structured output with SuggestionResponse
- Proper context serialization (consistent with execution agent)
- Fallback parsing for models without tool calling
- Cleaner separation of concerns
- Better error handling
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from graph.agents.suggestor import SuggestionResponse, build_suggestor_agent
from graph.memory import trim_conversation_history
from graph.state import AssistantState


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def retrieve_memories(user_id: int, query: str) -> str:
    """
    Retrieve relevant user memories for context.

    TODO: Replace with actual memory retrieval system.
    Currently returns placeholder data.
    """
    return (
        "- User prefers outdoor activities (walking, hiking) over gym.\n"
        "- User is trying to improve sleep consistency but struggles with late nights.\n"
        "- User responds well to positive reinforcement (You're doing great!).\n"
        "- User dislikes strict calorie counting."
    )


def _extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from text that might have markdown wrapping.

    This is a fallback for models that don't support proper structured output.
    """
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    try:
        json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_obj_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    # Last resort
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Could not extract JSON from text: {text[:200]}...")


def _safe_float(x: Any, default: float) -> float:
    """Safely convert value to float with fallback."""
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return default
    return default


def _last_human_text(messages: list[BaseMessage]) -> str:
    """Extract the last human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content or "")
    return "User Status Update"


def _serialize_user_context(user_profile: Any) -> str:
    """
    Serialize user context for the suggestor prompt.

    Extracts and formats:
    - Health goals
    - Activity preferences
    - Baseline metrics
    """
    context_parts = []

    # Extract goals
    if hasattr(user_profile, "health_goals"):
        goals = user_profile.health_goals
        if goals:
            goals_dict = goals.model_dump() if hasattr(goals, "model_dump") else goals.dict() if hasattr(goals, "dict") else goals
            context_parts.append(f"Goals: {json.dumps(goals_dict, default=str)}")
    elif isinstance(user_profile, dict):
        goals = user_profile.get("health_goals")
        if goals:
            context_parts.append(f"Goals: {json.dumps(goals, default=str)}")

    # Extract activity profile
    if hasattr(user_profile, "activity_profile"):
        activity = user_profile.activity_profile
        if activity:
            activity_dict = activity.model_dump() if hasattr(activity, "model_dump") else activity.dict() if hasattr(activity, "dict") else activity
            context_parts.append(f"Activity Profile: {json.dumps(activity_dict, default=str)}")
    elif isinstance(user_profile, dict):
        activity = user_profile.get("activity_profile")
        if activity:
            context_parts.append(f"Activity Profile: {json.dumps(activity, default=str)}")

    # Extract baselines
    if hasattr(user_profile, "baselines"):
        baselines = user_profile.baselines
        if baselines:
            baselines_dict = baselines.model_dump() if hasattr(baselines, "model_dump") else baselines.dict() if hasattr(baselines, "dict") else baselines
            context_parts.append(f"Baselines: {json.dumps(baselines_dict, default=str)}")
    elif isinstance(user_profile, dict):
        baselines = user_profile.get("baselines")
        if baselines:
            context_parts.append(f"Baselines: {json.dumps(baselines, default=str)}")

    return "\n".join(context_parts) if context_parts else "No user context available"


def _extract_coaching_preferences(user_profile: Any) -> tuple[float, str]:
    """
    Extract suggestiveness and tone from user profile.

    Returns:
        (suggestiveness: float, tone: str)
    """
    suggestiveness = 0.5  # Default
    tone = "supportive"  # Default

    if hasattr(user_profile, "coaching_preferences"):
        prefs = user_profile.coaching_preferences
        if prefs:
            suggestiveness = float(getattr(prefs, "suggestiveness", 0.5) or 0.5)
            tone = getattr(prefs, "tone", None) or "supportive"
    elif isinstance(user_profile, dict):
        prefs = user_profile.get("coaching_preferences", {}) or {}
        suggestiveness = _safe_float(prefs.get("suggestiveness"), 0.5)
        tone = prefs.get("tone") or "supportive"

    return suggestiveness, tone


def _extract_user_id(user_profile: Any) -> int:
    """Extract user ID from profile."""
    if hasattr(user_profile, "user_id"):
        return int(getattr(user_profile, "user_id", 0) or 0)
    elif isinstance(user_profile, dict):
        try:
            return int(user_profile.get("user_id", 0) or 0)
        except Exception:
            return 0
    return 0


# ============================================================================
# NODE FACTORY
# ============================================================================

def make_suggestor_node(config_dict: Dict[str, Any]) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    Create a suggestor node that adds coaching nudges to responses.

    Config structure:
    {
        "enabled": true,
        "llm": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 256
        },
        "min_suggestiveness": 0.3,
        "max_history_limit": 20
    }

    The node:
    1. Checks if suggestions are appropriate (based on user preferences)
    2. Gathers context (goals, memories, recent interaction)
    3. Invokes the suggestor agent for a coaching nudge
    4. Appends the suggestion to the assistant's response

    Returns:
        Node function for LangGraph
    """

    # Check if suggestor is enabled
    enabled = bool(config_dict.get("enabled", True))
    if not enabled:
        def _noop(_: AssistantState) -> Dict[str, Any]:
            return {}
        return _noop

    # Extract config
    llm_cfg = config_dict.get("llm", {})
    min_suggestiveness = float(config_dict.get("min_suggestiveness", 0.3))
    max_history_limit = int(config_dict.get("max_history_limit", 20))

    # Build the suggestor agent
    chain = build_suggestor_agent(llm_cfg)

    def suggestor_node(state: AssistantState) -> Dict[str, Any]:
        """
        Suggestor node implementation.

        Skips suggestion if:
        - Clarification is needed
        - No user profile available
        - No messages in conversation
        - Last message is not from assistant
        - User's suggestiveness is below threshold
        """

        # Skip if clarification needed
        if state.get("needs_clarification", False):
            return {}

        # Get required state
        user_profile = state.get("user_profile")
        messages = state.get("messages", [])

        if not user_profile or not messages:
            return {}

        # Check that last message is from assistant
        last_ai_message = messages[-1]
        trimmed_messages = trim_conversation_history(messages, config_dict.get("max_history_limit", 5))
        if not isinstance(last_ai_message, AIMessage):
            return {}

        # Extract user preferences
        suggestiveness, tone = _extract_coaching_preferences(user_profile)

        # Skip if user doesn't want suggestions
        if suggestiveness < min_suggestiveness:
            return {}

        # Get user query and ID
        user_query = _last_human_text(messages)
        user_id = _extract_user_id(user_profile)

        # Gather context
        user_context = _serialize_user_context(user_profile)
        relevant_memories = retrieve_memories(user_id, user_query)

        # Build interaction context
        interaction = (
            f"User asked: {user_query}\n"
            f"Relevant memories: {relevant_memories}"
        )

        # Invoke suggestor chain
        try:
            result = chain.invoke({
                "tone": tone,
                "user_context": user_context,
                "interaction": interaction,
                "history": trimmed_messages
            })

            # Handle different result types
            suggestion_response = None

            # Case 1: Proper structured output (SuggestionResponse)
            if isinstance(result, SuggestionResponse):
                suggestion_response = result

            # Case 2: Dict (from structured output validation)
            elif isinstance(result, dict):
                # Check if it's already the right shape
                if "suggestion" in result:
                    suggestion_response = SuggestionResponse.model_validate(result)
                else:
                    # Might be wrapped in 'structured_response'
                    sr = result.get("structured_response")
                    if sr:
                        suggestion_response = SuggestionResponse.model_validate(sr)

            # Case 3: String (fallback for models without tool calling)
            elif isinstance(result, str):
                try:
                    json_data = _extract_json_from_text(result)
                    suggestion_response = SuggestionResponse.model_validate(json_data)
                except (ValueError, Exception) as e:
                    print(f"Warning: Could not parse suggestion from text: {e}")
                    # Last resort: use the raw text as the suggestion
                    suggestion_response = SuggestionResponse(
                        suggestion=result.strip(),
                        include_suggestion=True,
                        reasoning="Fallback to raw text"
                    )

            if not suggestion_response or not suggestion_response.include_suggestion:
                return {}

            suggestion_text = (suggestion_response.suggestion or "").strip()
            if not suggestion_text:
                return {}

        except Exception as e:
            print(f"Error invoking suggestor: {e}")
            return {}

        # Append suggestion to the last AI message
        new_content = f"{last_ai_message.content}\n\n{suggestion_text}"
        updated_last = AIMessage(content=new_content)

        # Update messages
        trimmed_messages = trim_conversation_history(messages, max_messages=max_history_limit)
        trimmed_messages = list(trimmed_messages)

        if trimmed_messages and isinstance(trimmed_messages[-1], AIMessage):
            trimmed_messages[-1] = updated_last
        else:
            trimmed_messages.append(updated_last)

        return {
            "messages": trimmed_messages,
            "response": new_content,
            "suggestion_included": True,
        }

    return suggestor_node