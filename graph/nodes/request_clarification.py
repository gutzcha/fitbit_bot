"""
graph/nodes/request_clarification.py

Node responsible for pausing the conversation to ask for specifics.
Wraps the clarification chain but prioritizes specific questions from the Planner.
"""

from typing import Any, Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from graph.chains.request_clarification import build_clarification_chain
from graph.state import AssistantState


def make_clarification_node(
    llm: BaseChatModel,
) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    Factory that creates the clarification node executable.
    """

    # Build the chain once at startup
    clarification_chain = build_clarification_chain(llm)

    def request_clarification_node(state: AssistantState) -> Dict[str, Any]:
        """
        Executes the clarification logic.

        Logic:
        1. IF the Planner/Executor explicitly set a 'clarification_question', use it.
           (This is high-precision: "Which date range?")
        2. ELSE, run the generic clarification chain on the user's input.
           (This is low-precision fallback: "What did you mean?")
        """

        # --- 1. Check for High-Precision Questions from Process ---

        # Check ProcessPlan (from Planner)
        plan = state.get("process_plan")
        if plan and plan.needs_clarification and plan.clarification_question:
            return {
                "response": plan.clarification_question,
                "safe": True
            }

        # Check GroundingMetadata (from Execution Agent)
        meta = state.get("grounding_metadata")
        # Metadata might be a Pydantic model or a dict depending on upstream processing
        if meta:
            if isinstance(meta, dict):
                question = meta.get("clarification_question")
            else:
                question = getattr(meta, "clarification_question", None)

            if question:
                return {
                    "response": question,
                    "safe": True
                }

        # --- 2. Fallback: Generic Clarification ---
        # If we reached here, it means the INTENT node sent us here directly
        # or the downstream nodes failed to specify *what* was missing.

        messages = state.get("messages", [])
        if not messages:
            return {"response": "I'm listening. How can I help?"}

        last_user_msg = messages[-1]
        user_text = (
            last_user_msg.content
            if hasattr(last_user_msg, "content")
            else str(last_user_msg)
        )

        # Invoke the generic chain
        question = clarification_chain.invoke({"user_input": user_text})

        return {
            "response": question,
            "safe": True,
        }

    return request_clarification_node