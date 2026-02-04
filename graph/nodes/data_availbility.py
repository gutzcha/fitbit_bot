"""
graph/nodes/request_data_availability.py
Node responsible for pausing the conversation to ask for specifics.
Wraps the data_availability chain but prioritizes specific questions from the Planner.
===
"""

from typing import Any, Callable, Dict

from langchain.chat_models import init_chat_model

from graph.chains.data_availability import build_data_availability_chain

from graph.state import AssistantState
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, trim_messages
from graph.helpers import serialize_context_to_json
from graph.memory import trim_conversation_history
from graph.prompts.data_availability import DATA_AVAILABILITY_SYSTEM_STRING

from pydantic import BaseModel, Field
from graph.config_schemas import LLMConfig


class DataAvailabilityConfig(BaseModel):
    """Configuration for the data_availability Node."""
    llm: LLMConfig
    max_history_limit: int = Field(default=5, gt=0)


def make_data_availability_node(config_dict: Dict[str, Any]) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    Factory that creates the data_availability node executable using the provided configuration.

    Args:
        config_dict: Raw dictionary from config.json.
                     Validated internally against ClarificationNodeConfig.
    """

    # 1. Validate Configuration (Fail fast if invalid)
    config = DataAvailabilityConfig(**config_dict)

    # 2. Instantiate LLM from Config
    # Unpack the LLMConfig model directly into init_chat_model
    llm = init_chat_model(**config.llm.model_dump(exclude_none=True))

    # 3. Build the chain once at startup
    data_availability_chain = build_data_availability_chain(llm)

    # 4. Define Runtime Node
    def data_availability_node(state: AssistantState) -> Dict[str, Any]:
        """
        Executes the data_availability logic.
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
        context = [SystemMessage(content=DATA_AVAILABILITY_SYSTEM_STRING)]
        if plan_msg := serialize_context_to_json(plan,"PlanStep"):
            context += [AIMessage(content=plan_msg)]
        if conversation_state_msg := serialize_context_to_json(conversation_state, "ConversationState"):
            context += [AIMessage(content=conversation_state_msg)]
        if intent_metadata_msg := serialize_context_to_json(intent_metadata, "IntentMetadata"):
            context += [AIMessage(content=intent_metadata_msg)]
        if user_profile_msg := serialize_context_to_json(user_profile, "UserProfile"):
            context += [SystemMessage(content=user_profile_msg)]
        context += trimmed_messages

        if not trimmed_messages:
            return {"response": "Can you please clarify?"}


        # Invoke the generic chain
        response = data_availability_chain.invoke(context)

        return {
            "response": response,
            "safe": True,
        }

    return data_availability_node