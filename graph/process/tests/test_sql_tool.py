from graph.process.tools.sql_metrics import make_sql_tool
from tests.live_utils import SAMPLE_QUESTION, get_runtime_node, load_runtime_config


def _normalize_tool_output(result):
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return result.content
    raise AssertionError("Unexpected SQL tool output format")


def test_sql_tool_live():
    config = load_runtime_config()
    agent_cfg = get_runtime_node(config, "graph.process.nodes.sql_agent")
    validation_cfg = get_runtime_node(config, "graph.process.nodes.sql_validation")

    tool = make_sql_tool(agent_config=agent_cfg, validation_config=validation_cfg)

    result = tool.invoke({"query": SAMPLE_QUESTION})
    text = _normalize_tool_output(result)

    assert isinstance(text, str)
    assert text.strip()
    assert any(ch.isdigit() for ch in text)
