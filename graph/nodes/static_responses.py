"""
graph/nodes/static_responses.py

Handles deterministic responses for system intents:
1. GREETING -> Canned Welcome
2. OUT_OF_SCOPE -> Canned Refusal
3. DATA_AVAILABILITY -> Dynamic Menu of Capabilities based on data_config
"""

from typing import Any, Callable, Dict

# Corrected Import Path: graph.data_config (was dataset.data_config)
from graph.data_config import SQL_AVAILABLE_METRICS, VECTOR_KNOWLEDGE_TOPICS
from graph.defaults.static_response import StaticResponseNodeConfig
from graph.state import AssistantState
from graph.static_responses import (
    ERROR_RESPONSE,
    GREETING_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
)


def make_static_response_node(config_dict: Dict[str, Any]) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    Factory for the static response node.

    Args:
        config_dict: Raw dictionary from config.json.
                     Validated internally against StaticResponseNodeConfig.
    """

    # 1. Validate Configuration
    config = StaticResponseNodeConfig(**config_dict)

    def generate_static_response(state: AssistantState) -> Dict[str, Any]:
        # 2. Check if node is enabled using the validated object
        if not config.enabled:
            return {}

        intent_data = state.get("intent_metadata")
        if not intent_data:
            return {"response": ERROR_RESPONSE}

        intent = intent_data.intent

        # # ─────────────────────────────────────────────────────────────────────
        # # 1. DATA AVAILABILITY (The "Menu" Handler)
        # # ─────────────────────────────────────────────────────────────────────
        # if intent == "DATA_AVAILABILITY":
        #     # Extract readable metric names from the keys
        #     # e.g. "daily_activity" -> "Daily Activity"
        #     sql_categories = [k.replace("_", " ").title() for k in SQL_AVAILABLE_METRICS.keys()]
        #     metrics_str = ", ".join(sql_categories)
        #
        #     # Extract knowledge topics (keys from the Dict structure)
        #     topic_keys = VECTOR_KNOWLEDGE_TOPICS
        #     # Take top 3 for brevity
        #     topics_sample = ", ".join(topic_keys[:3])
        #
        #     response_text = (
        #         f"I can analyze data from these categories:\n"
        #         f"**{metrics_str}**\n\n"
        #         f"I can also answer health questions about topics like:\n"
        #         f"_{topics_sample}, and more._"
        #     )
        #
        #     return {"response": response_text, "safe": True, "grounded": True}

        # ─────────────────────────────────────────────────────────────────────
        # 2. STANDARD STATIC RESPONSES
        # ─────────────────────────────────────────────────────────────────────
        if intent == "GREETING":
            return {"response": GREETING_RESPONSE, "safe": True, "grounded": True}

        if intent == "OUT_OF_SCOPE":
            return {"response": OUT_OF_SCOPE_RESPONSE, "safe": True, "grounded": True}

        # Fallback for undefined static intents
        return {
            "response": "I'm not sure how to help with that specific request. Try asking about your heart rate or sport activities.",
            "safe": True,
            "grounded": True,
        }

    return generate_static_response