from langchain_core.messages import AIMessage, HumanMessage

from graph.helpers import extract_json_from_markdown
from graph.process.agents.execution import build_execution_agent
from graph.process.schemas import ExecutionResponse
from tests.live_utils import (SAMPLE_QUESTION, build_llm, get_runtime_node,
                              load_runtime_config)


def _extract_execution_response(raw_result) -> ExecutionResponse:
    if isinstance(raw_result, ExecutionResponse):
        return raw_result

    if isinstance(raw_result, dict):
        structured = raw_result.get("structured_response")
        if structured is not None:
            if isinstance(structured, ExecutionResponse):
                return structured
            if isinstance(structured, dict):
                return ExecutionResponse.model_validate(structured)

        messages = raw_result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content or ""
                if not content.strip():
                    continue
                data = extract_json_from_markdown(content)
                return ExecutionResponse.model_validate(data)

    raise AssertionError("Could not extract ExecutionResponse from agent output")


def test_execution_agent_live():
    config = load_runtime_config()
    exec_cfg = get_runtime_node(config, "graph.process.nodes.execution")

    manager_llm = build_llm(exec_cfg.get("llm_manager", {}))
    sql_ref = exec_cfg.get("sql_config_ref", "graph.process.nodes.sql_agent")
    sql_validation_ref = exec_cfg.get(
        "sql_validation_config_ref", "graph.process.nodes.sql_validation"
    )
    rag_ref = exec_cfg.get("rag_config_ref", "graph.process.rag_retriever")

    sql_cfg = get_runtime_node(config, sql_ref)
    sql_validation_cfg = get_runtime_node(config, sql_validation_ref)
    rag_cfg = get_runtime_node(config, rag_ref)

    agent = build_execution_agent(
        manager_llm=manager_llm,
        sql_config=sql_cfg,
        sql_validation_config=sql_validation_cfg,
        rag_config=rag_cfg,
    )

    raw_result = agent.invoke(
        {"messages": [HumanMessage(content=SAMPLE_QUESTION)]},
        config={"recursion_limit": int(exec_cfg.get("max_iterations", 5))},
    )

    exec_result = _extract_execution_response(raw_result)

    assert exec_result.answer.strip()
    assert 0.0 <= float(exec_result.confidence) <= 1.0
    assert exec_result.needs_clarification is False
