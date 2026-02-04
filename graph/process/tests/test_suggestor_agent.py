import json

from graph.agents.suggestor import SuggestionResponse, build_suggestor_agent
from graph.nodes.suggestor import _extract_json_from_text
from tests.live_utils import (
    SAMPLE_AI_ANSWER,
    SAMPLE_QUESTION,
    get_runtime_node,
    load_runtime_config,
)


def _parse_suggestor_result(result) -> SuggestionResponse:
    if isinstance(result, SuggestionResponse):
        return result

    if isinstance(result, dict):
        return SuggestionResponse.model_validate(result)

    if isinstance(result, str):
        try:
            parsed = _extract_json_from_text(result)
            return SuggestionResponse.model_validate(parsed)
        except Exception:
            return SuggestionResponse(
                suggestion=result.strip(),
                include_suggestion=True,
                reasoning="Fallback to raw text",
            )

    raise AssertionError("Unexpected suggestor output format")


def test_suggestor_agent_live():
    config = load_runtime_config()
    suggestor_cfg = get_runtime_node(config, "graph.process.nodes.suggestor")

    chain = build_suggestor_agent(suggestor_cfg.get("llm", {}))

    result = chain.invoke(
        {
            "tone": "energetic",
            "user_context": "Goals: {\"daily_steps_goal\": 12000}",
            "interaction": (
                f"User asked: {SAMPLE_QUESTION}\n"
                f"Assistant answered: {SAMPLE_AI_ANSWER}"
            ),
            "history": [],
        }
    )

    parsed = _parse_suggestor_result(result)

    assert parsed.suggestion.strip()
    assert parsed.include_suggestion is True
