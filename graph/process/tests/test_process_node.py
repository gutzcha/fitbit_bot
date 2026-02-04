from langchain_core.messages import HumanMessage

from graph.process.process import make_process_node
from tests.live_utils import (SAMPLE_QUESTION, load_runtime_config,
                              sample_conversation_state,
                              sample_intent_metadata, sample_user_profile)


def test_process_node_live():
    config = load_runtime_config()
    node = make_process_node(config)

    state = {
        "messages": [HumanMessage(content=SAMPLE_QUESTION)],
        "conversation_state": sample_conversation_state(),
        "intent_metadata": sample_intent_metadata(),
        "user_profile": sample_user_profile(),
    }

    result = node(state)

    assert isinstance(result["response"], str)
    assert result["response"].strip()
    assert "steps" in result["response"].lower()
    assert any(ch.isdigit() for ch in result["response"])
    assert result["needs_clarification"] is False


def test_process_node_empty_messages():
    config = load_runtime_config()
    node = make_process_node(config)

    result = node({"messages": []})

    assert result == {}
