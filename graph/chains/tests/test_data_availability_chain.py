from langchain_core.messages import HumanMessage, SystemMessage

from graph.chains.data_availability import build_data_availability_chain
from graph.prompts.data_availability import DATA_AVAILABILITY_SYSTEM_STRING
from tests.live_utils import build_llm, get_runtime_node, load_runtime_config


def test_data_availability_chain_live():
    config = load_runtime_config()
    node_cfg = get_runtime_node(config, "graph.nodes.data_availability")
    llm = build_llm(node_cfg.get("llm", {}))

    chain = build_data_availability_chain(llm)

    messages = [
        SystemMessage(content=DATA_AVAILABILITY_SYSTEM_STRING),
        HumanMessage(content="What can you help me with?"),
    ]

    result = chain.invoke(messages)

    assert isinstance(result, str)
    assert result.strip()
