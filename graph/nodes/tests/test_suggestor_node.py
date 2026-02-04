from langchain_core.messages import AIMessage, HumanMessage

from graph.nodes.suggestor import make_suggestor_node
from tests.live_utils import (
    SAMPLE_AI_ANSWER,
    SAMPLE_QUESTION,
    get_runtime_node,
    load_runtime_config,
    sample_user_profile,
)


def test_suggestor_node_live_appends_suggestion():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.process.nodes.suggestor")

    node = make_suggestor_node(node_cfg)

    state = {
        "needs_clarification": False,
        "user_profile": sample_user_profile(),
        "messages": [
            HumanMessage(content=SAMPLE_QUESTION),
            AIMessage(content=SAMPLE_AI_ANSWER),
        ],
    }

    result = node(state)

    assert result["suggestion_included"] is True
    assert SAMPLE_AI_ANSWER in result["response"]
    assert isinstance(result["messages"][-1], AIMessage)


def test_suggestor_node_skips_on_clarification():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.process.nodes.suggestor")

    node = make_suggestor_node(node_cfg)

    result = node({"needs_clarification": True, "messages": []})

    assert result == {}
