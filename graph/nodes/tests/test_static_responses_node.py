from graph.nodes import static_responses as static_module
from graph.schemas import IntentMetadata
from graph.static_responses import GREETING_RESPONSE


def test_static_response_node_returns_greeting():
    node = static_module.make_static_response_node({"enabled": True})

    state = {
        "intent_metadata": IntentMetadata(
            intent="GREETING",
            confidence=1.0,
            suggested_sources=["NONE"],
            response_type="HELP_MESSAGE",
            mentioned_metrics=[],
            current_topic="general",
            is_followup=False,
            needs_clarification=False,
        )
    }

    result = node(state)

    assert result["response"] == GREETING_RESPONSE
    assert result["safe"] is True
    assert result["grounded"] is True


def test_static_response_node_respects_disabled_config():
    node = static_module.make_static_response_node({"enabled": False})

    result = node({"intent_metadata": None})

    assert result == {}
