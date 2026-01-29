"""
graph/nodes/static_responses.py

Handles:
1. GREETING -> Canned Welcome
2. OUT_OF_SCOPE -> Canned Refusal
3. DATA_AVAILABILITY -> Dynamic Menu of Capabilities
"""

from typing import Any, Dict

# ✅ Import the "Menu" from your new config file
from graph.data_config import SQL_AVAILABLE_METRICS, VECTOR_KNOWLEDGE_TOPICS
from graph.state import AssistantState
from graph.static_responses import (ERROR_RESPONSE, GREETING_RESPONSE,
                                    OUT_OF_SCOPE_RESPONSE)


def make_static_response_node():
    """
    Factory for the static response node.
    """

    def generate_static_response(state: AssistantState) -> Dict[str, Any]:
        intent_data = state.get("intent_metadata")

        if not intent_data:
            return {"response": ERROR_RESPONSE}

        intent = intent_data.intent

        # ─────────────────────────────────────────────────────────────────────
        # 1. DATA AVAILABILITY (The "Menu" Handler)
        # ─────────────────────────────────────────────────────────────────────
        if intent == "DATA_AVAILABILITY":
            # Format the list of metrics nicely
            # e.g. "steps, heart_rate, sleep_score..."
            metrics_list = ", ".join([f"`{k}`" for k in SQL_AVAILABLE_METRICS.keys()])

            # Optional: Add a few examples of knowledge topics
            topics_sample = ", ".join(VECTOR_KNOWLEDGE_TOPICS[:3])

            response_text = (
                f"I can currently analyze data for these metrics:\n\n"
                f"{metrics_list}\n\n"
                f"I can also answer general questions about topics like *{topics_sample}*."
            )

            return {"response": response_text, "safe": True, "grounded": True}

        # ─────────────────────────────────────────────────────────────────────
        # 2. STANDARD STATIC RESPONSES
        # ─────────────────────────────────────────────────────────────────────
        if intent == "GREETING":
            return {"response": GREETING_RESPONSE, "safe": True, "grounded": True}

        if intent == "OUT_OF_SCOPE":
            return {"response": OUT_OF_SCOPE_RESPONSE, "safe": True, "grounded": True}

        # Fallback
        return {
            "response": "I'm not sure how to help with that specific request. Try asking about your steps or sleep.",
            "safe": True,
            "grounded": True,
        }

    return generate_static_response
