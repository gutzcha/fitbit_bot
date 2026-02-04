from langchain_core.messages import HumanMessage

from graph.nodes.intent import make_intent_node
from tests.live_utils import (
    SAMPLE_QUESTION,
    get_runtime_node,
    load_runtime_config,
    sample_conversation_state,
    sample_user_profile,
)


def test_intent_node_live_updates_state():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.nodes.intent")

    node = make_intent_node(node_cfg)

    conversation_state = sample_conversation_state()
    state = {
        "messages": [HumanMessage(content=SAMPLE_QUESTION)],
        "conversation_state": conversation_state,
        "user_profile": sample_user_profile(),
    }

    result = node(state)

    assert result["intent_metadata"].intent == "METRIC_RETRIEVAL"
    updated = result["conversation_state"]
    assert updated.turn_count == conversation_state.turn_count + 1
    assert updated.prior_intent == "METRIC_RETRIEVAL"
    assert "steps" in updated.mentioned_metrics
