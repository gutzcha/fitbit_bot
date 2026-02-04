from graph.process.rag_retriever.nodes import (
    make_generate_node,
    make_grade_documents_node,
    make_no_available_data_node,
    make_retrieve_node,
)
from tests.live_utils import get_runtime_node, load_runtime_config


def test_rag_retrieve_node_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    node = make_retrieve_node(rag_cfg.get("retriever", {}))

    result = node({"question": "What is a normal resting heart rate?"})

    assert result["documents"]
    assert result["question"] == "What is a normal resting heart rate?"


def test_rag_grade_documents_node_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    retrieve_node = make_retrieve_node(rag_cfg.get("retriever", {}))
    docs_state = retrieve_node({"question": "What is a normal resting heart rate?"})

    grade_node = make_grade_documents_node(rag_cfg.get("grade_documents", {}))
    result = grade_node(
        {"question": docs_state["question"], "documents": docs_state["documents"]}
    )

    assert result["no_data_available"] is False
    assert result["documents"]


def test_rag_generate_node_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    retrieve_node = make_retrieve_node(rag_cfg.get("retriever", {}))
    docs_state = retrieve_node({"question": "What is a normal resting heart rate?"})

    generate_node = make_generate_node(rag_cfg.get("generate", {}))
    result = generate_node(
        {"question": docs_state["question"], "documents": docs_state["documents"]}
    )

    assert isinstance(result["generation"], str)
    assert result["generation"].strip()


def test_rag_no_available_data_node():
    node = make_no_available_data_node()

    result = node({"question": "", "documents": [], "generation": "", "no_data_available": True})

    assert result["no_data_available"] is True
    assert "could not find" in result["generation"].lower()
