# graph/process/rag_retriever/nodes/grade_documents.py

from typing import Any, Callable, Dict

from graph.process.rag_retriever.chains.retrieval_grader import \
    make_grader_chain
from graph.process.rag_retriever.rag_state import GraphState


def make_grade_documents_node(
    config: Dict[str, Any],
) -> Callable[[GraphState], Dict[str, Any]]:
    """
    Factory for the GRADE_DOCUMENTS node.
    """
    # 1. Initialize the chain using the factory
    retrieval_grader = make_grader_chain(config)

    def grade_documents(state: GraphState) -> Dict[str, Any]:
        """
        Determines whether the retrieved documents are relevant to the question.
        Filters out irrelevant documents.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []

        # Default to True. If we find AT LEAST ONE relevant doc, this flips to False.
        no_data_available = True

        for d in documents:
            # Invoke chain
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )

            # 2. Check Score (String Comparison)
            # The chain returns "true" or "false" (string) as defined in GradeDocuments
            grade = score.binary_score

            if grade.lower() == "true":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
                no_data_available = False
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {
            "documents": filtered_docs,
            "question": question,
            "no_data_available": no_data_available,
        }

    return grade_documents
