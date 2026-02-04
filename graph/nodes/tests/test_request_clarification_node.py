from langchain_core.messages import HumanMessage

from graph.nodes.request_clarification import make_clarification_node
from tests.live_utils import (
    get_runtime_node,
    load_runtime_config,
    sample_conversation_state,
    sample_intent_metadata,
    sample_user_profile,
)


def test_request_clarification_node_live():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.nodes.request_clarification")

    node = make_clarification_node(node_cfg)

    state = {
        "messages": [HumanMessage(content="Tell me about my progress")],
        "process_plan": None,
        "conversation_state": sample_conversation_state(),
        "intent_metadata": sample_intent_metadata(),
        "user_profile": sample_user_profile(),
    }

    result = node(state)

    assert isinstance(result["response"], str)
    assert result["response"].strip()
    assert "?" in result["response"]
    assert result["safe"] is True


def test_request_clarification_node_empty_messages():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.nodes.request_clarification")

    node = make_clarification_node(node_cfg)

    result = node({"messages": []})

    assert result["response"] == "I'm listening. How can I help?"
