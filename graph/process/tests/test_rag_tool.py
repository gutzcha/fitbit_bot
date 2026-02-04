from graph.process.tools.rag_retriever import make_rag_tool
from tests.live_utils import get_runtime_node, load_runtime_config


def _normalize_tool_output(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    if isinstance(result, dict) and "content" in result and "artifact" in result:
        return result["content"], result["artifact"]
    if hasattr(result, "content") and hasattr(result, "artifact"):
        return result.content, result.artifact
    raise AssertionError("Unexpected tool output format")


def test_rag_tool_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    tool = make_rag_tool(rag_cfg)

    result = tool.invoke({"question": "What is a normal resting heart rate?"})
    content, artifact = _normalize_tool_output(result)

    assert isinstance(content, str)
    assert content.strip()
    assert artifact["raw_status"] == "success"
    assert artifact["sources"]
