from langchain_core.messages import AIMessage, HumanMessage

from graph.helpers import extract_json_from_markdown
from graph.process.agents.sql_agent import build_sql_agent
from graph.process.schemas import SQLAgentResponse
from tests.live_utils import (
    SAMPLE_QUESTION,
    get_runtime_node,
    load_runtime_config,
)


def _extract_sql_response(raw_result) -> SQLAgentResponse:
    if isinstance(raw_result, SQLAgentResponse):
        return raw_result

    if isinstance(raw_result, dict):
        structured = raw_result.get("structured_response")
        if structured is not None:
            if isinstance(structured, SQLAgentResponse):
                return structured
            if isinstance(structured, dict):
                return SQLAgentResponse.model_validate(structured)

        messages = raw_result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content or ""
                if not content.strip():
                    continue
                data = extract_json_from_markdown(content)
                return SQLAgentResponse.model_validate(data)

    raise AssertionError("Could not extract SQLAgentResponse from agent output")


def test_sql_agent_live():
    config = load_runtime_config()
    agent_cfg = get_runtime_node(config, "graph.process.nodes.sql_agent")
    validation_cfg = get_runtime_node(config, "graph.process.nodes.sql_validation")

    agent = build_sql_agent(agent_config=agent_cfg, validation_config=validation_cfg)

    prompt = (
        "User ID: 1503960366. "
        f"{SAMPLE_QUESTION} "
        "Answer using the database and keep it concise."
    )

    raw_result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    sql_result = _extract_sql_response(raw_result)

    assert sql_result.answer.strip()
    assert sql_result.sql_queries
    assert sql_result.table_names
    assert 0.0 <= float(sql_result.confidence) <= 1.0
