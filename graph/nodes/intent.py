"""
graph/nodes/intent.py
"""

import json
from typing import Any, Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from graph.chains.intent import build_intent_chain
from graph.memory import trim_conversation_history
from graph.schemas import IntentMetadata
from graph.state import AssistantState, ConversationState


def make_intent_node(
    fast_llm: BaseChatModel,
    slow_llm: BaseChatModel | None = None,
    slow_fallback_enabled: bool = True,
    slow_fallback_min_confidence: float = 0.99,
    max_history_limit: int = 10,
) -> Callable[[AssistantState], Dict[str, Any]]:
    classifier_chain = build_intent_chain(
        fast_llm=fast_llm,
        slow_llm=slow_llm,
        slow_fallback_enabled=slow_fallback_enabled,
        slow_fallback_min_confidence=slow_fallback_min_confidence,
    )

    def intent_node(state: AssistantState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}

        # 1. Get User Input
        last_msg = messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # 2. Prepare Conversation Context
        trimmed_messages = trim_conversation_history(
            messages, max_messages=max_history_limit
        )
        current_conv_state = state.get("conversation_state") or ConversationState()

        # 3. Prepare User Profile Context
        # We inject this so the Intent Classifier knows the user's specific context
        # (e.g. knowing "my goal" refers to a 10k step goal)
        user_profile = state.get("user_profile")
        profile_context_str = "No user profile available."

        if user_profile:
            # Handle Pydantic serialization safely
            if hasattr(user_profile, "model_dump_json"):
                profile_context_str = user_profile.model_dump_json()
            elif isinstance(user_profile, dict):
                profile_context_str = json.dumps(user_profile)
            else:
                profile_context_str = str(user_profile)

        # 4. RUN CHAIN
        # We pass the 'user_profile' key which your prompt template must expect
        intent_result: IntentMetadata = classifier_chain.invoke(
            {
                "messages": trimmed_messages,
                "conversation_state": current_conv_state,
                "user_profile": profile_context_str
            }
        )

        # 5. UPDATE STATE
        updated_conv_state = current_conv_state.model_copy(deep=True)
        updated_conv_state.turn_count += 1
        updated_conv_state.prior_intent = intent_result.intent
        updated_conv_state.user_explicitly_asked = user_text

        # ✅ Explicitly update topic if the LLM found one
        if intent_result.current_topic and intent_result.current_topic != "general":
            updated_conv_state.current_topic = intent_result.current_topic

        # ✅ Merge mentioned metrics
        if intent_result.mentioned_metrics:
            updated_conv_state.mentioned_metrics.update(intent_result.mentioned_metrics)

        return {
            "intent_metadata": intent_result,
            "conversation_state": updated_conv_state,
        }

    return intent_node