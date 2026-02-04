from langchain_core.documents import Document

from graph.process.rag_retriever.chains.generation import make_generation_chain
from graph.process.rag_retriever.chains.retrieval_grader import make_grader_chain
from graph.process.rag_retriever.chains.retriever import make_retriever
from tests.live_utils import get_runtime_node, load_runtime_config


def test_rag_retriever_chain_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    retriever = make_retriever(rag_cfg.get("retriever", {}))
    docs = retriever.invoke("What is a normal resting heart rate?")

    assert docs
    assert hasattr(docs[0], "page_content")


def test_rag_grader_chain_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    grader = make_grader_chain(rag_cfg.get("grade_documents", {}))
    question = "What is a normal resting heart rate?"

    relevant = Document(page_content="Normal resting heart rate is often 60 to 100 bpm for adults.")
    irrelevant = Document(page_content="This document is about banana bread recipes and baking tips.")

    score_relevant = grader.invoke({"question": question, "document": relevant.page_content})
    score_irrelevant = grader.invoke({"question": question, "document": irrelevant.page_content})

    assert score_relevant.binary_score.lower() == "true"
    assert score_irrelevant.binary_score.lower() == "false"


def test_rag_generation_chain_live():
    config = load_runtime_config()
    rag_cfg = get_runtime_node(config, "graph.process.rag_retriever")

    chain = make_generation_chain(rag_cfg.get("generate", {}))

    context = "XQZ123 is a unique test token used for validation."
    result = chain.invoke({"context": context, "question": "Explain XQZ123."})

    assert isinstance(result, str)
    assert result.strip()
    assert "xqz123" in result.lower()
