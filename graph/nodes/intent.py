"""
graph/nodes/intent.py
Factory for the Intent Classification graph node.
Orchestrates a configurable dual-LLM chain (fast/slow fallback).
===
"""

import json
from typing import Any, Callable, Dict

from langchain.chat_models import init_chat_model

from graph.chains.intent import build_intent_chain
from graph.defaults.intent import IntentNodeConfig  # Import the schema
from graph.memory import trim_conversation_history
from graph.schemas import IntentMetadata
from graph.state import AssistantState, ConversationState


def make_intent_node(config_dict: Dict[str, Any]) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    Factory that builds the intent node.

    Args:
        config_dict: Raw dictionary from config.json.
                     Validated internally against IntentNodeConfig.
    """

    # 1. Validate Configuration (Fail fast if invalid)
    config = IntentNodeConfig(**config_dict)

    # 2. Instantiate Fast LLM
    # We use the validated 'config' object now
    fast_llm = init_chat_model(**config.llm_fast.model_dump(exclude_none=True))

    # 3. Instantiate Slow LLM
    slow_llm = None
    if config.llm_slow:
        slow_llm = init_chat_model(**config.llm_slow.model_dump(exclude_none=True))

    # 4. Build Chain
    classifier_chain = build_intent_chain(
        fast_llm=fast_llm,
        slow_llm=slow_llm,
        slow_fallback_enabled=(slow_llm is not None),
        slow_fallback_min_confidence=config.confidence_threshold,
    )

    # 5. Define Runtime Node
    def intent_node(state: AssistantState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        trimmed_messages = trim_conversation_history(
            messages, max_messages=config.max_history_limit
        )
        current_conv_state = state.get("conversation_state") or ConversationState()

        user_profile = state.get("user_profile")
        profile_context_str = "No user profile available."

        if user_profile:
            if hasattr(user_profile, "model_dump_json"):
                profile_context_str = user_profile.model_dump_json()
            elif isinstance(user_profile, dict):
                profile_context_str = json.dumps(user_profile)
            else:
                profile_context_str = str(user_profile)

        intent_result: IntentMetadata = classifier_chain.invoke(
            {
                "messages": trimmed_messages,
                "conversation_state": current_conv_state,
                "user_profile": profile_context_str,
            }
        )

        updated_conv_state = current_conv_state.model_copy(deep=True)
        updated_conv_state.turn_count += 1
        updated_conv_state.prior_intent = intent_result.intent
        updated_conv_state.user_explicitly_asked = user_text

        if intent_result.current_topic and intent_result.current_topic != "general":
            updated_conv_state.current_topic = intent_result.current_topic

        if intent_result.mentioned_metrics:
            updated_conv_state.mentioned_metrics.update(intent_result.mentioned_metrics)

        return {
            "intent_metadata": intent_result,
            "conversation_state": updated_conv_state,
        }

    return intent_node