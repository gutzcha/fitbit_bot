from langchain_core.messages import HumanMessage

from graph.chains.intent import build_intent_chain
from graph.schemas import IntentMetadata
from tests.live_utils import (SAMPLE_QUESTION, build_llm, get_runtime_node,
                              load_runtime_config)


def _build_intent_chain_from_config(config: dict, slow_threshold: float | None = None):
    node_cfg = get_runtime_node(config, "graph.nodes.intent")
    fast_llm = build_llm(node_cfg.get("llm_fast", {}))
    slow_cfg = node_cfg.get("llm_slow")
    slow_llm = build_llm(slow_cfg) if slow_cfg else None

    threshold = slow_threshold
    if threshold is None:
        threshold = node_cfg.get("confidence_threshold", 0.7)

    return build_intent_chain(
        fast_llm=fast_llm,
        slow_llm=slow_llm,
        slow_fallback_enabled=slow_llm is not None,
        slow_fallback_min_confidence=float(threshold),
    )


def test_intent_chain_metric_retrieval_live():
    config = load_runtime_config()
    chain = _build_intent_chain_from_config(config)

    result = chain.invoke({"messages": [HumanMessage(content=SAMPLE_QUESTION)]})

    assert isinstance(result, IntentMetadata)
    assert result.intent == "METRIC_RETRIEVAL"
    assert "USER_METRICS" in result.suggested_sources
    assert result.needs_clarification is False


def test_intent_chain_greeting_live():
    config = load_runtime_config()
    chain = _build_intent_chain_from_config(config)

    result = chain.invoke({"messages": [HumanMessage(content="Hello there!")]})

    assert isinstance(result, IntentMetadata)
    assert result.intent == "GREETING"


def test_intent_chain_out_of_scope_live():
    config = load_runtime_config()
    chain = _build_intent_chain_from_config(config)

    result = chain.invoke(
        {"messages": [HumanMessage(content="Write a Python script to sort a list.")]}
    )

    assert isinstance(result, IntentMetadata)
    assert result.intent == "OUT_OF_SCOPE"


def test_intent_chain_forces_slow_fallback_live():
    config = load_runtime_config()
    chain = _build_intent_chain_from_config(config, slow_threshold=1.1)

    result = chain.invoke({"messages": [HumanMessage(content=SAMPLE_QUESTION)]})

    assert isinstance(result, IntentMetadata)
    assert result.intent == "METRIC_RETRIEVAL"
