"""
graph/nodes/request_clarification.py
Node responsible for pausing the conversation to ask for specifics.
Wraps the clarification chain but prioritizes specific questions from the Planner.
===
"""

from typing import Any, Callable, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     trim_messages)

from graph.chains.request_clarification import build_clarification_chain
from graph.defaults.request_clarification import ClarificationNodeConfig
from graph.helpers import serialize_context_to_json
from graph.memory import trim_conversation_history
from graph.prompts.request_clarification import CLARIFICATION_SYSTEM_STRING
from graph.state import AssistantState


def make_clarification_node(
    config_dict: Dict[str, Any],
) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    Factory that creates the clarification node executable using the provided configuration.

    Args:
        config_dict: Raw dictionary from config.json.
                     Validated internally against ClarificationNodeConfig.
    """

    # 1. Validate Configuration (Fail fast if invalid)
    config = ClarificationNodeConfig(**config_dict)

    # 2. Instantiate LLM from Config
    # Unpack the LLMConfig model directly into init_chat_model
    llm = init_chat_model(**config.llm.model_dump(exclude_none=True))

    # 3. Build the chain once at startup
    clarification_chain = build_clarification_chain(llm)

    # 4. Define Runtime Node
    def request_clarification_node(state: AssistantState) -> Dict[str, Any]:
        """
        Executes the clarification logic.

        Logic:
        1. IF the Planner/Executor explicitly set a 'clarification_question', use it.
           (This is high-precision: "Which date range?")
        2. ELSE, run the generic clarification chain on the user's input.
           (This is low-precision fallback: "What did you mean?")
        """

        # --- A. Check for High-Precision Questions from Process ---

        # Check ProcessPlan (from Planner)
        plan = state.get("process_plan")
        conversation_state = state.get("conversation_state", {})
        intent_metadata = state.get("intent_metadata", {})
        user_profile = state.get("user_profile", {})

        messages = state.get("messages", [])
        trimmed_messages = trim_conversation_history(
            messages, max_messages=config.max_history_limit
        )
        context = [SystemMessage(content=CLARIFICATION_SYSTEM_STRING)]
        if plan_msg := serialize_context_to_json(plan, "PlanStep"):
            context += [AIMessage(content=plan_msg)]
        if conversation_state_msg := serialize_context_to_json(
            conversation_state, "ConversationState"
        ):
            context += [AIMessage(content=conversation_state_msg)]
        if intent_metadata_msg := serialize_context_to_json(
            intent_metadata, "IntentMetadata"
        ):
            context += [AIMessage(content=intent_metadata_msg)]
        if user_profile_msg := serialize_context_to_json(user_profile, "UserProfile"):
            context += [SystemMessage(content=user_profile_msg)]
        context += trimmed_messages

        # --- B. Fallback: Generic Clarification ---

        if not trimmed_messages:
            return {"response": "I'm listening. How can I help?"}

        # Invoke the generic chain
        response = clarification_chain.invoke(context)

        return {
            "response": response,
            "safe": True,
        }

    return request_clarification_node
