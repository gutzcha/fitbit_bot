from langchain_core.messages import HumanMessage, SystemMessage

from graph.chains.request_clarification import build_clarification_chain
from graph.prompts.request_clarification import CLARIFICATION_SYSTEM_STRING
from tests.live_utils import build_llm, get_runtime_node, load_runtime_config


def test_clarification_chain_live():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.nodes.request_clarification")
    llm = build_llm(node_cfg.get("llm", {}))

    chain = build_clarification_chain(llm)

    messages = [
        SystemMessage(content=CLARIFICATION_SYSTEM_STRING),
        HumanMessage(content="How am I doing?"),
    ]

    result = chain.invoke(messages)

    assert isinstance(result, str)
    assert result.strip()
    assert "?" in result
    assert len(result.split()) <= 25
